#!/usr/bin/env python3
"""Exporte les parties sauvegardées par le service web en données d'entraînement (JSON Lines).

    python3 scripts/export_games.py data/games parties.jsonl            # une partie par ligne
    python3 scripts/export_games.py /data/games parties.jsonl --finished # seulement les parties terminées
    python3 scripts/export_games.py data/games actions.jsonl --no-states # sans les états (plus léger)

Chaque ligne est un export ``fortyk-training/1`` (voir ``fortyk/training.py``) : pour chaque décision,
l'état compact du plateau avant la décision, la décision posée et l'action choisie (humain ou bot),
puis les événements du moteur, le résultat et les branches annulées.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import load_catalog  # noqa: E402
from fortyk.training import training_export  # noqa: E402
from fortyk.web.rooms import FORMAT, public_record  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("games_dir", help="dossier des parties (FORTYK_DATA_DIR du service)")
    p.add_argument("out", help="fichier JSON Lines produit")
    p.add_argument("--finished", action="store_true", help="ignorer les parties en cours")
    p.add_argument("--no-states", action="store_true", help="ne pas inclure l'état avant chaque décision")
    args = p.parse_args(argv)
    cat = load_catalog()
    n = skipped = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for path in sorted(Path(args.games_dir).glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                if doc.get("format") != FORMAT or (args.finished and doc.get("status") != "finished"):
                    skipped += 1
                    continue
                ex = training_export(cat, public_record(doc), states=not args.no_states)
            except Exception as err:  # noqa: BLE001
                print(f"{path.name} : ignorée ({type(err).__name__}: {err})", file=sys.stderr)
                skipped += 1
                continue
            out.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
            print(f"{path.name} : {len(ex['samples'])} décisions — {ex['game']['title']}")
    print(f"{n} partie(s) exportée(s) dans {args.out}" + (f", {skipped} ignorée(s)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
