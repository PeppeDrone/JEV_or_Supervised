from pathlib import Path

from jevtest.benchmark import ALL_PAIRS, discover_pairs, missing_from_yaml

PROJECT = Path(__file__).resolve().parents[1]


def test_all_pairs_cover_every_prepared_dataset():
    discovered = set(discover_pairs(PROJECT))
    listed = set(ALL_PAIRS)
    assert listed == discovered, (
        f"ALL_PAIRS missing {sorted(discovered - listed)}; extra {sorted(listed - discovered)}"
    )
    assert not missing_from_yaml(PROJECT)
    assert len(ALL_PAIRS) == len(listed)
