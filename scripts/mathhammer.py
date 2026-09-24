#!/usr/bin/env python3
"""Math-hammer : espérance d'une arme d'une fiche contre une autre fiche, avec vérification aux dés.

    python3 scripts/mathhammer.py "Intercessor Squad" "Bolt rifle" "Infractors" --count 5
    python3 scripts/mathhammer.py "Intercessor Squad" "Bolt rifle" "Infractors" --count 5 --cover --reroll-hits fails
    python3 scripts/mathhammer.py "Infractors" "Duelling sabre" "Intercessor Squad" --count 4 --reroll-wounds ones
    python3 scripts/mathhammer.py "Intercessor Squad" "Hand flamer" "Infractors"

Le contexte (couvert, Heavy, relances, +A de Hail of Bolts…) se donne à la main : c'est
la couche combat qui le déduira des positions plus tard. Les mots-clés lus sur l'arme
sont appliqués automatiquement : Torrent, Lethal Hits, Sustained Hits, Devastating
Wounds, Twin-linked, Anti-X (si la cible a le mot-clé X).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import Datasheet, Weapon, load_catalog  # noqa: E402
from fortyk.data.parse import DiceExpr  # noqa: E402
from fortyk.engine.attack import (  # noqa: E402
    AttackProfile,
    Defender,
    estimate_models_slain,
    expected_attacks,
    expected_hazardous_mortal_wounds,
    hit_roll_needed,
    resolve_attacks,
    wound_roll_needed_for,
)
from fortyk.engine.rules import DEFAULT_RULES, save_needed  # noqa: E402


def profile_from_weapon(
    weapon: Weapon,
    target: Datasheet,
    count: int,
    extra_attacks: int = 0,
    hit_modifier: int = 0,
    wound_modifier: int = 0,
    reroll_hits: str = "none",
    reroll_wounds: str = "none",
    blast_target_models: int = 0,
) -> AttackProfile:
    """Traduit un profil d'arme Wahapedia en :class:`AttackProfile` pour une cible donnée."""
    attacks = weapon.attacks
    bonus = extra_attacks
    if weapon.has("blast") and blast_target_models:
        bonus += blast_target_models // DEFAULT_RULES.blast_models_per_extra_attack
    if bonus:
        attacks = DiceExpr(attacks.n_dice, attacks.sides, attacks.flat + bonus)
    if weapon.has("twin-linked"):
        reroll_wounds = "fails"
    sustained = weapon.keyword("sustained hits")
    sustained_value = None
    if sustained is not None:
        v = sustained.value if sustained.value is not None else 1
        sustained_value = v if isinstance(v, DiceExpr) else DiceExpr.fixed(int(v))
    critical_wound_on = 6
    anti = weapon.keyword("anti")
    if anti is not None and anti.target and target.has_keyword(anti.target) and isinstance(anti.value, int):
        critical_wound_on = anti.value
    return AttackProfile(
        attacks=attacks,
        skill=None if weapon.has("torrent") else weapon.skill,
        strength=weapon.strength,
        ap=weapon.ap,
        damage=weapon.damage,
        count=count,
        hit_modifier=hit_modifier,
        wound_modifier=wound_modifier,
        reroll_hits=reroll_hits,
        reroll_wounds=reroll_wounds,
        lethal_hits=weapon.has("lethal hits"),
        sustained_hits=sustained_value,
        devastating_wounds=weapon.has("devastating wounds"),
        critical_wound_on=critical_wound_on,
        label=f"{weapon.name} ×{count}",
    )


def defender_from_datasheet(target: Datasheet) -> Defender:
    m = target.models[0]
    fnp = None
    for a in target.core_abilities:
        if a.name == "Feel No Pain" and a.parameter.rstrip("+").isdigit():
            fnp = int(a.parameter.rstrip("+"))
    return Defender(
        toughness=m.toughness,
        save=m.save,
        invuln=m.invuln,
        feel_no_pain=fnp,
        is_vehicle_or_monster=target.has_keyword("Vehicle") or target.has_keyword("Monster"),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("attacker", help="fiche qui attaque")
    p.add_argument("weapon", help="nom de l'arme (profil) sur la fiche")
    p.add_argument("target", help="fiche visée")
    p.add_argument("--count", "-n", type=int, default=1, help="figurines tirant avec cette arme")
    p.add_argument("--attacker-faction")
    p.add_argument("--target-faction")
    p.add_argument("--target-models", type=int, default=5, help="effectif de la cible (Blast, estimation des pertes)")
    p.add_argument("--extra-attacks", type=int, default=0, help="+A (ex. Hail of Bolts : 2)")
    p.add_argument("--cover", action="store_true", help="la cible a le couvert (-1 touche)")
    p.add_argument("--heavy-bonus", action="store_true", help="Heavy actif (+1 touche, unité ayant bougé < 3\")")
    p.add_argument("--hit-mod", type=int, default=0)
    p.add_argument("--wound-mod", type=int, default=0)
    p.add_argument("--reroll-hits", choices=["none", "ones", "fails"], default="none")
    p.add_argument("--reroll-wounds", choices=["none", "ones", "fails"], default="none")
    p.add_argument("--sims", type=int, default=20000, help="répétitions Monte-Carlo (0 pour désactiver)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cat = load_catalog()
    attacker = cat.get(args.attacker, args.attacker_faction)
    target = cat.get(args.target, args.target_faction)
    weapon = attacker.weapon(args.weapon)
    if weapon is None:
        print(f"Arme {args.weapon!r} introuvable sur {attacker.name}. Armes : " + ", ".join(w.name for w in attacker.weapons), file=sys.stderr)
        return 1

    hit_mod = args.hit_mod - (1 if args.cover else 0) + (DEFAULT_RULES.heavy_hit_bonus if args.heavy_bonus and weapon.has("heavy") else 0)
    if args.cover and weapon.has("ignores cover"):
        hit_mod += 1  # Ignores Cover annule le malus de couvert
    profile = profile_from_weapon(
        weapon, target, args.count, args.extra_attacks, hit_mod, args.wound_mod, args.reroll_hits, args.reroll_wounds, args.target_models
    )
    defender = defender_from_datasheet(target)
    tm = target.models[0]

    print(f"{attacker.name} — {weapon}")
    print(f"  → {target.name} : T{tm.toughness} Sv{tm.save}+" + (f" Inv{tm.invuln}+" if tm.invuln else "") + f" W{tm.wounds}" + (f" FNP{defender.feel_no_pain}+" if defender.feel_no_pain else ""))
    hn = hit_roll_needed(profile)
    sv = save_needed(defender.save, profile.ap, defender.invuln)
    print(f"  touche {'auto' if hn is None else f'{hn}+'} | blesse {wound_roll_needed_for(profile, defender)}+ | sauvegarde {'aucune' if sv is None else f'{sv}+'}"
          + (f" | critique blessure {profile.critical_wound_on}+" if profile.critical_wound_on != 6 else "")
          + (f" | relance touches: {profile.reroll_hits}" if profile.reroll_hits != "none" else "")
          + (f" | relance blessures: {profile.reroll_wounds}" if profile.reroll_wounds != "none" else ""))

    e = expected_attacks(profile, defender)
    print(f"\n  Espérance : {e.attacks:.2f} attaques → {e.hits:.2f} touches → {e.wounds:.2f} blessures"
          + (f" (dont {e.devastating:.2f} dévastatrices)" if e.devastating else "")
          + f" → {e.unsaved + e.devastating:.2f} passent → {e.damage:.2f} dégâts")
    slain = estimate_models_slain(e.unsaved + e.devastating, profile.damage, tm.wounds, defender.feel_no_pain)
    print(f"  ≈ {slain:.2f} figurine(s) de {tm.wounds} PV tuée(s) sur {args.target_models}")
    if weapon.has("hazardous"):
        print(f"  Hazardous : {expected_hazardous_mortal_wounds(args.count, attacker.has_keyword('Vehicle') or attacker.has_keyword('Monster')):.2f} blessure(s) mortelle(s) attendue(s) pour le tireur")

    if args.sims > 0:
        rng = random.Random(args.seed)
        tot_dmg = tot_unsaved = 0.0
        kills = 0
        for _ in range(args.sims):
            out = resolve_attacks(profile, defender, rng)
            tot_dmg += out.total_damage
            tot_unsaved += out.unsaved + out.devastating
            # allocation simple : une figurine à la fois, pas de débordement
            hp, dead = tm.wounds, 0
            for d in out.damage:
                hp -= d
                if hp <= 0:
                    dead += 1
                    hp = tm.wounds
            kills += min(dead, args.target_models)
        print(f"\n  Monte-Carlo ({args.sims} tirs) : {tot_unsaved / args.sims:.2f} passent, {tot_dmg / args.sims:.2f} dégâts, {kills / args.sims:.2f} figurine(s) tuée(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
