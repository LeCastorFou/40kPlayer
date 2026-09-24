#!/usr/bin/env python3
"""Rend une carte de bataille en PNG pour vérification visuelle.

    python3 scripts/render_layout.py layout_a                 # → data/layouts/layout_a.png
    python3 scripts/render_layout.py layout_a --out /tmp/a.png --scale 20

Nécessite Pillow (``pip install pillow``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.engine.layout import LAYOUTS_DIR, load_layout  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("layout", help="nom court (layout_a) ou chemin d'un JSON")
    p.add_argument("--out", help="fichier PNG de sortie")
    p.add_argument("--scale", type=float, default=16.0, help="pixels par pouce")
    args = p.parse_args(argv)

    layout = load_layout(args.layout)
    out = Path(args.out) if args.out else LAYOUTS_DIR / f"{Path(args.layout).stem}.png"
    issues = layout.symmetry_report()
    for i in issues:
        print("! " + i, file=sys.stderr)
    layout.render(out, scale=args.scale)
    print(f"{layout.name} : {len(layout.terrain)} décors, {len(layout.objectives)} objectifs → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
