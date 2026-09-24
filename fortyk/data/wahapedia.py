"""Lecture brute des CSV de l'export Wahapedia.

Particularités du format (cf. ``Export Data Specs.xlsx``) :

* délimiteur ``|``, aucun échappement ni guillemet (``csv.QUOTE_NONE``) ;
* UTF-8 avec BOM ;
* chaque ligne se termine par un ``|`` final, donc une colonne vide surnuméraire ;
* les champs texte (``description``…) contiennent du HTML.

Le dossier des CSV est, par ordre de priorité : l'argument explicite, la variable
d'environnement ``FORTYK_WAHAPEDIA_DIR``, puis ``<racine du dépôt>/data/wahapedia/raw``.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

__all__ = ["RawTables", "default_raw_dir", "TABLE_NAMES"]

csv.field_size_limit(1 << 30)

TABLE_NAMES = (
    "Factions",
    "Source",
    "Datasheets",
    "Datasheets_abilities",
    "Datasheets_keywords",
    "Datasheets_models",
    "Datasheets_options",
    "Datasheets_wargear",
    "Datasheets_unit_composition",
    "Datasheets_models_cost",
    "Datasheets_stratagems",
    "Datasheets_enhancements",
    "Datasheets_detachment_abilities",
    "Datasheets_leader",
    "Stratagems",
    "Abilities",
    "Enhancements",
    "Detachment_abilities",
    "Detachments",
    "Detachments_chapter_dp",
    "Last_update",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def default_raw_dir() -> Path:
    env = os.environ.get("FORTYK_WAHAPEDIA_DIR")
    return Path(env).expanduser() if env else _REPO_ROOT / "data" / "wahapedia" / "raw"


Row = Dict[str, str]


class RawTables:
    """Accès paresseux et mis en cache aux tables CSV, chaque ligne étant un ``dict``."""

    def __init__(self, raw_dir: Optional[os.PathLike] = None):
        self.raw_dir = Path(raw_dir) if raw_dir is not None else default_raw_dir()
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier des CSV Wahapedia introuvable : {self.raw_dir}\n"
                "Lance `python3 scripts/fetch_wahapedia.py` ou définis FORTYK_WAHAPEDIA_DIR."
            )

    def path(self, table: str) -> Path:
        return self.raw_dir / f"{table}.csv"

    def available(self) -> List[str]:
        return [t for t in TABLE_NAMES if self.path(t).exists()]

    @lru_cache(maxsize=None)
    def read(self, table: str) -> tuple:
        """Toutes les lignes de ``table`` sous forme de tuple de dicts (clé = nom de colonne)."""
        path = self.path(table)
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh, delimiter="|", quoting=csv.QUOTE_NONE)
            try:
                header = next(reader)
            except StopIteration:
                return ()
            # Colonne vide finale due au « | » de fin de ligne.
            while header and header[-1] == "":
                header.pop()
            n = len(header)
            rows = []
            for raw in reader:
                if not any(raw):
                    continue
                cells = raw[:n]
                if len(cells) < n:
                    cells = cells + [""] * (n - len(cells))
                rows.append(dict(zip(header, cells)))
        return tuple(rows)

    def by_key(self, table: str, key: str) -> Dict[str, List[Row]]:
        """Regroupe les lignes d'une table par la valeur d'une colonne (ex. ``datasheet_id``)."""
        groups: Dict[str, List[Row]] = {}
        for row in self.read(table):
            groups.setdefault(row[key], []).append(row)
        return groups

    def index(self, table: str, key: str = "id") -> Dict[str, Row]:
        """Première ligne pour chaque valeur de ``key`` (les tables Abilities / Stratagems
        dupliquent certains identifiants entre factions : la première occurrence est gardée)."""
        out: Dict[str, Row] = {}
        for row in self.read(table):
            out.setdefault(row[key], row)
        return out

    def last_update(self) -> Optional[str]:
        rows = self.read("Last_update")
        return rows[0]["last_update"] if rows else None

    def __repr__(self) -> str:
        return f"RawTables({str(self.raw_dir)!r}, {len(self.available())} tables)"
