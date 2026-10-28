"""Output-level ensembles of the member models' out-of-fold posteriors, with equal weights."""
import numpy as np


def mean(posteriors):
    return np.mean(posteriors, axis=0)


def mode(posteriors):
    """Vote shares of the members' argmax labels. Ties go to the tied class with the highest mean posterior."""
    posteriors = np.asarray(posteriors)
    m, n, k = posteriors.shape
    votes = np.zeros((n, k))
    for P in posteriors:
        votes[np.arange(n), P.argmax(axis=1)] += 1
    shares = votes / m
    winner = np.where(votes == votes.max(axis=1, keepdims=True), posteriors.mean(axis=0), -np.inf).argmax(axis=1)
    return shares, winner


def run_ensembles(run):
    members = run.study["ensemble"]["members"]
    found = {name: run.predictions(name) for name in members}
    missing = [name for name, P in found.items() if P is None]
    if missing:
        print(f"ensembles skipped: missing member predictions {missing}", flush=True)
        return False
    posteriors = np.stack([found[name] for name in members])
    run.save_predictions("Mean", mean(posteriors))
    shares, winner = mode(posteriors)
    run.save_predictions("Mode", shares, labels=winner)
    return True
