"""Unified component classes shared by data prep, training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from . import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "classes.yaml"


@dataclass(frozen=True)
class Taxonomy:
    classes: list[str]
    source_map: dict[str, dict[str, tuple[str | None, str | None]]]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "Taxonomy":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        classes = list(cfg["classes"])
        source_map = {
            src: {raw: (m[0], m[1]) for raw, m in labels.items()}
            for src, labels in cfg["source_map"].items()
        }
        for src, labels in source_map.items():
            for raw, (name, _) in labels.items():
                if name is not None and name not in classes:
                    raise ValueError(f"{src}:{raw} maps to unknown class {name!r}")
        return cls(classes, source_map)

    def index(self, name: str) -> int:
        return self.classes.index(name)

    def map(self, source: str, raw_label: str) -> tuple[str | None, str | None] | None:
        """Return (class, subtype) for a raw label, or None if the label is unmapped."""
        return self.source_map[source].get(raw_label)
