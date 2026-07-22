from __future__ import annotations

import json
from pathlib import Path

from migrate_legacy import collect_sort_profiles


def test_collect_sort_profiles_ignores_stale_paths_without_losing_profile(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    sort_root = tmp_path / "library"
    target = sort_root / "targets"
    control = sort_root / "controls"
    target.mkdir(parents=True)
    control.mkdir()
    legacy.mkdir()
    (legacy / "sorter_profiles.json").write_text(
        json.dumps(
            {
                "Useful": {
                    "target": str(target),
                    "controls": [
                        str(control),
                        str(sort_root / "missing"),
                        str(tmp_path / "outside"),
                    ],
                },
                "Missing target": {"target": str(sort_root / "gone")},
                "Bad/name": {"target": str(target)},
            }
        ),
        encoding="utf-8",
    )

    profiles, skipped_profiles, stale_controls = collect_sort_profiles(
        legacy, sort_root
    )

    assert profiles == [
        {
            "name": "Useful",
            "target_directory": "targets",
            "control_directories": ["controls"],
            "mode": "stem",
            "threshold_seconds": 50,
            "add_ids": False,
        }
    ]
    assert skipped_profiles == 2
    assert stale_controls == 2
