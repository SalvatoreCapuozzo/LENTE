"""Descriptor schema and class atlas.

The schema is the controlled vocabulary used to describe an object (material,
shape, optics, ...). The atlas is a text file that describes each class in that
vocabulary. Neither contributes a parameter to the model: the schema values are
phrases scored through a shared text path, the atlas is read at inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class Field:
    name: str
    kind: str            # "single" | "multi"
    scope: str           # "intrinsic" | "contextual"
    weight: float
    templates: list[str]
    values: list[str]


@dataclass
class Schema:
    fields: list[Field]
    # flat index over every (field, value) pair, in field order
    flat: list[tuple[str, str]] = field(default_factory=list)
    slices: dict[str, slice] = field(default_factory=dict)

    def __post_init__(self):
        self._reindex()

    def _reindex(self):
        self.flat, self.slices, start = [], {}, 0
        for f in self.fields:
            self.slices[f.name] = slice(start, start + len(f.values))
            self.flat += [(f.name, v) for v in f.values]
            start += len(f.values)

    @property
    def n_values(self) -> int:
        return len(self.flat)

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"field {name!r} not in schema")

    def field_index(self) -> torch.Tensor:
        """(V,) long: which field each flat value belongs to."""
        names = [f.name for f in self.fields]
        return torch.tensor([names.index(fn) for fn, _ in self.flat], dtype=torch.long)

    def phrases(self) -> list[list[str]]:
        """Prompt-ensembled phrases for every flat value (list of templates each)."""
        out = []
        for fn, v in self.flat:
            f = self.field(fn)
            out.append([t.format(v) for t in f.templates])
        return out

    def extended(self, atlases: list["Atlas"]) -> "Schema":
        """Copy of the schema with every value that an atlas uses but the schema
        lacks appended to its field. Those values get no parameter of their own:
        they are scored with the field's temperature and bias."""
        fields = [Field(f.name, f.kind, f.scope, f.weight, list(f.templates), list(f.values))
                  for f in self.fields]
        by_name = {f.name: f for f in fields}
        for atlas in atlases:
            for e in atlas.entries:
                for fn, vals in e.descriptors.items():
                    if fn not in by_name:
                        raise KeyError(f"atlas {atlas.name!r} entry {e.name!r} uses unknown field {fn!r}")
                    for v in _as_list(vals):
                        if v not in by_name[fn].values:
                            by_name[fn].values.append(v)
        return Schema(fields)

    @staticmethod
    def load(path: str | Path) -> "Schema":
        d = yaml.safe_load(Path(path).read_text())
        fields = []
        for name, spec in d["fields"].items():
            fields.append(Field(name=name, kind=spec["kind"], scope=spec["scope"],
                                weight=float(spec.get("weight", 1.0)),
                                templates=list(spec["templates"]), values=[str(v) for v in spec["values"]]))
            if spec["kind"] not in ("single", "multi"):
                raise ValueError(f"field {name}: kind must be single|multi")
            if spec["scope"] not in ("intrinsic", "contextual"):
                raise ValueError(f"field {name}: scope must be intrinsic|contextual")
        return Schema(fields)


def _as_list(v) -> list[str]:
    if v is None:
        return []
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]


@dataclass
class Entry:
    name: str
    descriptors: dict
    prose: str = ""
    stream: str | None = None


@dataclass
class Atlas:
    name: str
    entries: list[Entry]

    def __post_init__(self):
        self._by_name = {e.name: e for e in self.entries}
        if len(self._by_name) != len(self.entries):
            raise ValueError(f"atlas {self.name!r} has duplicate entry names")

    def __contains__(self, name):
        return name in self._by_name

    def __getitem__(self, name) -> Entry:
        return self._by_name[name]

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    @staticmethod
    def load(path: str | Path) -> "Atlas":
        d = yaml.safe_load(Path(path).read_text())
        entries = [Entry(name=str(e["name"]), descriptors=dict(e.get("descriptors") or {}),
                         prose=str(e.get("prose", "")), stream=e.get("stream")) for e in d["entries"]]
        return Atlas(name=str(d.get("name", Path(path).stem)), entries=entries)

    @staticmethod
    def merge(atlases: list["Atlas"], name="merged") -> "Atlas":
        seen, entries = set(), []
        for a in atlases:
            for e in a.entries:
                if e.name in seen:
                    raise ValueError(f"entry {e.name!r} defined in more than one atlas")
                seen.add(e.name)
                entries.append(e)
        return Atlas(name, entries)


def entry_target(entry: Entry | None, schema: Schema) -> tuple[np.ndarray, np.ndarray]:
    """Descriptor target of one atlas entry over the flat schema index.

    Returns (target, mask), both (V,). `mask` is False for every value of a field
    the entry does not describe, so an unknown field is never read as negative.
    Single-valued fields listing several values get a uniform soft target.
    Values outside `schema` are ignored here (they only matter at inference).
    """
    V = schema.n_values
    tgt = np.zeros(V, dtype=np.float32)
    msk = np.zeros(V, dtype=bool)
    if entry is None:
        return tgt, msk
    for f in schema.fields:
        if f.name not in entry.descriptors:
            continue
        vals = [v for v in _as_list(entry.descriptors[f.name]) if v in f.values]
        if not vals:
            continue
        sl = schema.slices[f.name]
        msk[sl] = True
        for v in vals:
            tgt[sl.start + f.values.index(v)] = 1.0
        if f.kind == "single":
            tgt[sl] /= tgt[sl].sum()
    return tgt, msk


def context_target(value: str | None, schema: Schema, field_name="surroundings") -> tuple[np.ndarray, np.ndarray]:
    """Target for a contextual field known at image level (e.g. dataset scene)."""
    V = schema.n_values
    tgt = np.zeros(V, dtype=np.float32)
    msk = np.zeros(V, dtype=bool)
    if value is None or field_name not in schema.slices:
        return tgt, msk
    f = schema.field(field_name)
    if value not in f.values:
        raise KeyError(f"{value!r} is not a value of field {field_name!r}")
    sl = schema.slices[field_name]
    msk[sl] = True
    tgt[sl.start + f.values.index(value)] = 1.0
    return tgt, msk
