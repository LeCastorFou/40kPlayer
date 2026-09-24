"""Couverture de la traduction automatique des règles Wahapedia (stratagèmes de détachement).

    python3 scripts/rules_coverage.py                 # par faction
    python3 scripts/rules_coverage.py --faction EC    # détail d'une faction (identifiant Wahapedia)
    python3 scripts/rules_coverage.py --unparsed 40   # phrases non traduites les plus fréquentes
"""

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import load_catalog  # noqa: E402
from fortyk.rules_compiler import compile_stratagem  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faction", help="identifiant Wahapedia de la faction (SM, EC, GC…) : détail stratagème par stratagème")
    ap.add_argument("--unparsed", type=int, default=0, help="afficher les N phrases non traduites les plus fréquentes")
    args = ap.parse_args()
    cat = load_catalog()
    rules = cat.rules
    std = [s for s in rules.stratagems if s.detachment_id and rules.detachments.get(s.detachment_id) and not rules.detachments[s.detachment_id].type]
    by_faction = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    unparsed = collections.Counter()
    moments = collections.Counter()
    for st in std:
        c = compile_stratagem(st)
        by_faction[st.faction_id][c.status] += 1
        total[c.status] += 1
        moments[c.timing.moment] += 1
        for u in c.unparsed:
            unparsed[re.sub(r"\d+", "N", re.sub(r"\b[A-Z][A-Z’' -]{2,}\b", "KW", u))[:120]] += 1
        if args.faction and st.faction_id == args.faction:
            print(f"[{c.status:7}] {st.name} ({st.detachment}) — {c.summary() or '—'}")
            for u in c.unparsed:
                print(f"           à la main : {u}")
    n = sum(total.values())
    print(f"\nStratagèmes de détachement : {n} — auto {total['auto']} ({100 * total['auto'] / n:.0f} %), "
          f"partiel {total['partial']} ({100 * total['partial'] / n:.0f} %), à la main {total['manual']} ({100 * total['manual'] / n:.0f} %)")
    if not args.faction:
        names = {f.id: f.name for f in cat.factions.values()} if isinstance(cat.factions, dict) else {}
        for fid, cnt in sorted(by_faction.items(), key=lambda kv: -sum(kv[1].values())):
            k = sum(cnt.values())
            print(f"  {names.get(fid, fid):32} {k:4}  auto {cnt['auto']:3}  partiel {cnt['partial']:3}  main {cnt['manual']:3}")
        print("moments :", dict(moments.most_common()))
    for s, k in unparsed.most_common(args.unparsed):
        print(f"{k:4}  {s}")


if __name__ == "__main__":
    main()
