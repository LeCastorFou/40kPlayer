#!/usr/bin/env python3
"""Télécharge l'export CSV de Wahapedia (Warhammer 40,000 V11) dans data/wahapedia/raw/.

Usage :
    python3 scripts/fetch_wahapedia.py              # tout télécharger
    python3 scripts/fetch_wahapedia.py --check      # juste comparer Last_update local/distant
    python3 scripts/fetch_wahapedia.py --only Datasheets_models Datasheets_wargear

Ne dépend que de la bibliothèque standard. Les fichiers sont écrits de façon atomique
(téléchargement dans un .part puis renommage), l'ancien fichier est conservé en cas d'erreur.

Format des fichiers (cf. Export Data Specs.xlsx) :
  - CSV, délimiteur "|" (barre verticale), UTF-8 (parfois avec BOM)
  - les champs description / abilities contiennent du HTML
  - les booléens sont les chaînes "true" / "false"
  - chaque ligne se termine généralement par un "|" final (colonne vide à ignorer au chargement)

Données © Games Workshop, compilées par Wahapedia. Créditer « powered by Wahapedia »
dans tout travail dérivé publié.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EDITION = "wh40k11ed"
BASE_URL = f"https://wahapedia.ru/{EDITION}/"

# Les 21 fichiers listés dans Export Data Specs.xlsx (feuille EN).
FILES: tuple[str, ...] = (
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

# Wahapedia refuse parfois les clients sans User-Agent de navigateur.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "wahapedia" / "raw"


def fetch(url: str, timeout: float = 60.0, retries: int = 3) -> bytes:
    """GET avec quelques tentatives (Wahapedia est parfois lent)."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
            last_err = err
            if attempt < retries:
                time.sleep(2.0 * attempt)
    assert last_err is not None
    raise last_err


def decode(raw: bytes) -> str:
    """UTF-8, en tolérant un BOM éventuel."""
    return raw.decode("utf-8-sig")


def sanity_check(name: str, text: str) -> list[str]:
    """Vérifications minimales sur le contenu téléchargé. Retourne des avertissements."""
    warnings: list[str] = []
    lines = text.splitlines()
    if not lines:
        warnings.append("fichier vide")
        return warnings
    header = lines[0]
    if "|" not in header:
        warnings.append(f"pas de délimiteur '|' dans l'en-tête : {header[:80]!r}")
    lowered = text[:2000].lower()
    if "<!doctype html" in lowered or "<html" in lowered:
        warnings.append("le contenu ressemble à une page HTML, pas à un CSV (blocage / URL changée ?)")
    if name != "Last_update" and len(lines) < 2:
        warnings.append("aucune ligne de données")
    return warnings


def read_last_update(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = decode(path.read_bytes()).splitlines()
    if len(lines) < 2:
        return None
    return lines[1].strip("| \r")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"dossier de destination (défaut : {DEFAULT_OUT})")
    parser.add_argument("--only", nargs="+", metavar="FILE", help="ne télécharger que ces fichiers (sans .csv)")
    parser.add_argument("--check", action="store_true", help="comparer Last_update local et distant, sans télécharger")
    args = parser.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.check:
        remote = decode(fetch(BASE_URL + "Last_update.csv")).splitlines()
        remote_ts = remote[1].strip("| \r") if len(remote) > 1 else "?"
        local_ts = read_last_update(out / "Last_update.csv") or "(absent)"
        print(f"Last_update distant : {remote_ts}")
        print(f"Last_update local   : {local_ts}")
        return 0 if remote_ts == local_ts else 1

    names = tuple(args.only) if args.only else FILES
    unknown = [n for n in names if n not in FILES]
    if unknown:
        print(f"Fichiers inconnus (pas dans la spec) : {unknown}", file=sys.stderr)
        return 2

    print(f"Source : {BASE_URL}   ->   {out}\n")
    failures: list[str] = []
    total_bytes = 0
    for name in names:
        url = f"{BASE_URL}{name}.csv"
        target = out / f"{name}.csv"
        part = target.with_suffix(".csv.part")
        try:
            raw = fetch(url)
        except Exception as err:  # noqa: BLE001 - on veut continuer sur les autres fichiers
            print(f"  ✗ {name:34s} ÉCHEC : {err}")
            failures.append(name)
            continue

        text = decode(raw)
        warnings = sanity_check(name, text)
        n_lines = max(text.count("\n") - 1, 0)  # lignes de données (hors en-tête)
        part.write_bytes(raw)
        part.replace(target)
        total_bytes += len(raw)
        flag = "⚠" if warnings else "✓"
        print(f"  {flag} {name:34s} {len(raw)/1024:8.1f} ko  {n_lines:6d} lignes")
        for w in warnings:
            print(f"      -> {w}")

    print(f"\n{len(names) - len(failures)}/{len(names)} fichiers, {total_bytes/1024/1024:.2f} Mo")
    ts = read_last_update(out / "Last_update.csv")
    if ts:
        print(f"Export Wahapedia daté du {ts} (GMT+3)")
    if failures:
        print(f"Échecs : {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
