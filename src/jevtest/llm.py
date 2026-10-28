"""Frozen backbone access: weights source, direct Hugging Face model and the AnyJev zero-shot runtime.

Ported from neurojev.evaluation.m3_learning (model_source, LLM, anyjev_runtime). Inference is local.
"""
from pathlib import Path
import json, os

import numpy as np

from jevtest.config import GateError

LETTERS = "ABCDEFGHIJ"


def orders(k):
    """Option orders read for every prompt: the k cyclic rotations (forward and reversed when k = 2)."""
    return [list(np.roll(np.arange(k), -s)) for s in range(k)]


def answer_instruction(k):
    letters = list(LETTERS[:k])
    return f"Answer with {', '.join(letters[:-1])} or {letters[-1]}."


def options_block(question, classes, order):
    lines = "\n".join(f"{LETTERS[j]}. {classes[c]}" for j, c in enumerate(order))
    return f"Question: {question}\n{lines}\n{answer_instruction(len(classes))}"


def model_source(model_id, revision):
    """Plain-file local copy when HF_HOME is on the /workspace network volume (see neurojev HANDOFF pitfalls)."""
    home = os.environ.get("HF_HOME", "")
    if not home.startswith("/workspace"):
        return model_id, revision
    from huggingface_hub import snapshot_download

    target = Path(home) / "local" / f"{model_id.replace('/', '--')}@{revision}"

    def truncated(folder):
        bad = []
        for f in Path(folder).glob("*.safetensors"):
            size = f.stat().st_size
            try:
                with f.open("rb") as handle:
                    n = int.from_bytes(handle.read(8), "little")
                    header = json.loads(handle.read(n)) if 0 < n < size else {}
                end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=-1)
                complete = size == 8 + n + end
            except (OSError, ValueError):
                complete = False
            if not complete:
                bad.append(f)
        return bad

    index = target / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if all((target / s).exists() for s in shards) and not truncated(target):
            return str(target), None
    for attempt in range(5):
        try:
            path = snapshot_download(model_id, revision=revision, local_dir=target, max_workers=2)
        except OSError as error:
            print(f"download pass {attempt + 1} failed ({error}); retrying", flush=True)
            continue
        bad = truncated(path)
        if not bad:
            return str(path), None
        for f in bad:
            f.unlink()
            for record in (target / ".cache/huggingface/download").glob(f"{f.name}.*"):
                record.unlink()
    raise RuntimeError(f"weights in {target} still incomplete after 5 download passes")


def tokenizer(study):
    from transformers import AutoTokenizer

    source, revision = model_source(study["backbone"]["model_id"], study["backbone"]["revision"])
    return AutoTokenizer.from_pretrained(source, revision=revision)


def deterministic(study, warn_only):
    """Seeded, deterministic kernels. `backbone.deterministic_warn_only: true` downgrades the error raised by
    kernels without a deterministic implementation (seen with sharded MoE backbones) to a warning."""
    import torch
    from transformers import set_seed

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(study["seed"])
    warn_only = bool(warn_only or study["backbone"].get("deterministic_warn_only", False))
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def placement_kwargs(study):
    """`from_pretrained` keyword arguments that place the backbone: empty for one GPU (then `.cuda()`),
    `device_map` (+ optional `max_memory`) to shard across the visible GPUs (needs `accelerate`)."""
    backbone = study["backbone"]
    # Optional extra `from_pretrained` kwargs (e.g. `use_kernels: true`); numerically neutral, so not in provenance.
    kwargs = dict(backbone.get("model_kwargs") or {})
    device_map = backbone.get("device_map")
    if not device_map:
        return kwargs
    try:
        import accelerate  # noqa: F401
    except ImportError as error:
        raise GateError("backbone.device_map needs the `accelerate` package (pip install -e '.[inference]')") from error
    kwargs["device_map"] = device_map
    if backbone.get("max_memory"):
        kwargs["max_memory"] = {(int(k) if str(k).isdigit() else k): v for k, v in backbone["max_memory"].items()}
    return kwargs


def chat(tok, content):
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


class LLM:
    """The pinned backbone for hidden states and answer-letter readouts; weights frozen."""

    def __init__(self, study, rope_scaling=None):
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise GateError("CUDA unavailable; LLM stages run on GPU only")
        deterministic(study, warn_only=True)
        self.torch, self.batch = torch, study["backbone"]["llm_batch"]
        source, revision = model_source(study["backbone"]["model_id"], study["backbone"]["revision"])
        self.tokenizer = tokenizer(study)
        self.tokenizer.padding_side = "left"
        placement = placement_kwargs(study)
        self.model = AutoModelForCausalLM.from_pretrained(
            source, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa",
            **({"rope_scaling": rope_scaling} if rope_scaling else {}), **placement,
        )
        if "device_map" not in placement:
            self.model = self.model.cuda()
        self.model.eval()
        # Inputs go to the device of the first layer; a sharded model moves activations between GPUs itself.
        self.device = self.model.get_input_embeddings().weight.device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        letters = [self.tokenizer.encode(x, add_special_tokens=False) for x in LETTERS]
        assert all(len(x) == 1 for x in letters)
        self.letters = [x[0] for x in letters]

    def prompt(self, text, question, classes, order):
        return chat(self.tokenizer, f"{text}\n\n{options_block(question, classes, order)}")

    def encode(self, prompts):
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        # Left padding: positions must start at the first real token.
        batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        return batch

    def embed(self, texts, question, classes, layers):
        """Last-token hidden states of the forward-order prompt at the given layers."""
        order = list(range(len(classes)))
        out = {f"layer{k}": [] for k in layers}
        with self.torch.no_grad():
            for offset in range(0, len(texts), self.batch):
                prompts = [self.prompt(t, question, classes, order) for t in texts[offset:offset + self.batch]]
                states = self.model(**self.encode(prompts), output_hidden_states=True,
                                    logits_to_keep=1, use_cache=False).hidden_states
                for k in layers:
                    out[f"layer{k}"].append(states[k][:, -1].float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    def prefix_scores(self, prefix, suffixes, k, reuse="crop", progress=None):
        """Softmax over the k answer letters for each suffix after one shared, once-encoded prefix."""
        import copy
        from transformers import DynamicCache

        torch = self.torch
        ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        cache, scores = DynamicCache(), []
        with torch.no_grad():
            self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            for completed, suffix in enumerate(suffixes, 1):
                tail = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                own = copy.deepcopy(cache) if reuse == "copy" else cache
                out = self.model(input_ids=tail, past_key_values=own, use_cache=True, logits_to_keep=1)
                scores.append(out.logits[0, -1, self.letters[:k]].float().softmax(-1).cpu().numpy())
                if reuse == "copy":
                    del own
                else:
                    cache.crop(ids.shape[1])
                if progress is not None and (completed % 25 == 0 or completed == len(suffixes)):
                    progress(completed, len(suffixes))
        return scores, ids.shape[1]


def check_anyjev(study):
    from importlib.metadata import distribution

    direct = json.loads(distribution("anyjev").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != study["backbone"]["anyjev_commit"]:
        raise GateError("Installed AnyJev commit does not match the pinned commit in configs/study.yaml")


def anyjev_runtime(study):
    """Frozen backbone, AnyJev L0 readout with the content-free prior (as neurojev jev_improvement.runtime)."""
    import torch
    from anyjev import Decider
    from anyjev.backends.hf import HFBackend

    if not torch.cuda.is_available():
        raise GateError("CUDA unavailable; LLM stages run on GPU only")
    check_anyjev(study)
    deterministic(study, warn_only=False)
    source, revision = model_source(study["backbone"]["model_id"], study["backbone"]["revision"])
    # With device_map the backend leaves placement to accelerate and reads inputs' device from the model.
    backend = HFBackend(source, revision=revision, device="cuda", dtype="bfloat16",
                        batch_size=study["backbone"]["anyjev_batch"], trust_remote_code=False,
                        attn_implementation="sdpa", **placement_kwargs(study))
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    # `backbone.anyjev_prior` (content_free | batch | none; default content_free) and the optional
    # `backbone.anyjev_prior_strength` change the readout, so both stay in the prediction provenance.
    prior = study["backbone"].get("anyjev_prior", "content_free")
    strength = study["backbone"].get("anyjev_prior_strength")
    extra = {} if strength is None else {"prior_strength": float(strength)}
    return backend, Decider(backend, level="L0", prior=prior, adaptive_shifts=False, shared_prefix=False, **extra)
