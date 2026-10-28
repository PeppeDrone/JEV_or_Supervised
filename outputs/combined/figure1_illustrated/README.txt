Illustrated manuscript Figure 1

An icon-led, continuous study flow for the active 45-task benchmark.
Final dimensions: 183 × 96 mm. Smallest label: 7.8 pt at full width.
Publication exports live alongside the other combined figures in ../figures/.
This directory contains the editable source, generated assets, caption and proofs.

Deliverables
  ../figures/Figure1.pdf        Publication export: embedded illustrations and vector text/arrows.
  ../figures/Figure1.svg        Editable composition: separate images, live labels and paths.
  ../figures/Figure1.png        600 dpi manuscript raster export.
  make_figure1.py    Reproducible figure layout.
  caption.txt        Figure caption grounded in the active benchmark code/reports.
  assets/           Six original generated PNG icons with transparent backgrounds.
  asset_prompts.json Full prompts, generation mode and original asset locations.
  proof_183mm.png    Full-width reproduction proof.
  proof_160mm.png    Reduced-width reproduction proof.
  validation.json   Text/illustration bounds and overlap checks.
  validation/       Export/content checks and review records.

Editing and reproduction
  Open ../figures/Figure1.svg in a vector editor to reposition illustrations, edit labels and
  alter the connecting paths. The illustrations are embedded raster objects;
  their individual painted strokes are not vector-editable. Labels and arrows
  remain editable, and the original high-resolution icon assets are included.
  Use Arial to maintain the supplied typography. The PDF embeds its font subsets.
  Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python outputs/combined/figure1_illustrated/make_figure1.py

Generation
  The six icon illustrations were generated with the built-in image_gen tool,
  following the imagegen skill. The full prompt set is in asset_prompts.json.
  The final page was assembled with scientific plotting tools to preserve exact
  labels and editable connectors. The generated source pixels were not altered.

Scope
  The current benchmark code and reports are the scientific source, as confirmed
  by the user. The removed manuscript folder is unrelated and was not used.
  The frozen language model does not undergo supervised fitting in outer folds.
  Complexity and codebook analyses are associations across tasks. The illustration
  does not imply causation, clinical correctness of codebooks, or model superiority.
