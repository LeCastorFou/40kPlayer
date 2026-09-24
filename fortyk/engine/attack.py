"""Séquence d'attaque V11 : jet de touche → jet de blessure → sauvegarde → dégâts.

Deux implémentations de la même séquence, testées l'une contre l'autre :

* :func:`resolve_attacks` lance réellement les dés (pour jouer) ;
* :func:`expected_attacks` calcule les espérances en fermé (pour que l'IA évalue
  une action sans simuler).

L'unité de travail est l':class:`AttackProfile` : un lot d'attaques homogènes
(``count`` figurines identiques, une arme, un même contexte : modificateurs, relances,
mots-clés actifs). C'est la couche « combat » (à venir) qui construira ces profils
à partir des unités, des positions, des portées et du couvert évalué tireur par
tireur ; ici on ne manipule que des nombres.

Mots-clés pris en charge dans la séquence : Torrent (``skill=None``), Lethal Hits,
Sustained Hits N, Devastating Wounds, Anti-X (``critical_wound_on``), Twin-linked
(``reroll_wounds="fails"``), Feel No Pain côté défenseur. Heavy, Blast, Rapid Fire,
Melta, Ignores Cover, Pistol, Assault, Hazardous et Precision agissent *avant* ou
*après* la séquence (modificateur, nombre d'attaques, dégâts, allocation) : voir
:func:`hazardous_test` et la couche combat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..data.parse import DiceExpr
from .rules import DEFAULT_RULES, RulesConfig, clamp_modifier, clamp_roll, save_needed, wound_roll_needed

__all__ = [
    "AttackProfile",
    "Defender",
    "AttackOutcome",
    "ExpectedOutcome",
    "hit_roll_needed",
    "wound_roll_needed_for",
    "resolve_attacks",
    "expected_attacks",
    "estimate_models_slain",
    "hazardous_test",
    "expected_hazardous_mortal_wounds",
]

REROLL_NONE = "none"
REROLL_ONES = "ones"
REROLL_FAILS = "fails"
_REROLLS = (REROLL_NONE, REROLL_ONES, REROLL_FAILS)


@dataclass(frozen=True)
class AttackProfile:
    """Un lot d'attaques homogènes."""

    attacks: DiceExpr  #: attaques par figurine, déjà ajustées (Blast, Rapid Fire, Hail of Bolts…)
    skill: Optional[int]  #: BS ou WS ; None = touche automatique (Torrent), sans critique possible
    strength: int
    ap: int  #: 0, -1, -2… (signé comme sur la fiche)
    damage: DiceExpr
    count: int = 1  #: nombre de figurines identiques utilisant ce profil
    hit_modifier: int = 0  #: net avant plafonnement (couvert -1, Heavy +1…)
    wound_modifier: int = 0
    reroll_hits: str = REROLL_NONE  #: "none" | "ones" | "fails"
    reroll_wounds: str = REROLL_NONE
    lethal_hits: bool = False
    sustained_hits: Optional[DiceExpr] = None  #: touches supplémentaires par touche critique
    devastating_wounds: bool = False
    critical_wound_on: int = 6  #: Anti-X N+ contre une cible X : N
    min_unmodified_hit: int = 0  #: Indirect Fire (10.07) : un jet non modifié inférieur rate (6 ou 4)
    precision: bool = False  #: [PRECISION] (24.28) : l'attaquant peut choisir un groupe PERSONNAGE visible
    critical_hit_on: int = 6  #: touche critique sur N+ (effets « critical hit on 5+ »)
    label: str = ""  #: pour les journaux (« Bolt rifle ×5 »)

    def __post_init__(self):
        if self.reroll_hits not in _REROLLS or self.reroll_wounds not in _REROLLS:
            raise ValueError(f"relance inconnue : {self.reroll_hits!r} / {self.reroll_wounds!r}")
        if self.count < 0:
            raise ValueError("count doit être ≥ 0")


@dataclass(frozen=True)
class Defender:
    """Caractéristiques défensives de la cible (identiques pour toutes ses figurines)."""

    toughness: int
    save: int
    invuln: Optional[int] = None
    feel_no_pain: Optional[int] = None
    is_vehicle_or_monster: bool = False
    damage_reduction: int = 0  #: « subtract 1 from the Damage characteristic » (Duty Eternal…) ; les dégâts ne descendent pas sous 1


@dataclass
class AttackOutcome:
    """Résultat d'une résolution aux dés. ``damage`` liste les dégâts de chaque blessure
    non sauvegardée ou dévastatrice, après Feel No Pain, avant allocation aux figurines."""

    attacks: int = 0
    hits: int = 0
    critical_hits: int = 0
    wounds: int = 0  #: blessures réussies (dont automatiques via Lethal Hits)
    critical_wounds: int = 0
    devastating: int = 0  #: blessures critiques passées sans sauvegarde (Devastating Wounds)
    saves_attempted: int = 0
    unsaved: int = 0
    damage: List[int] = field(default_factory=list)

    @property
    def total_damage(self) -> int:
        return sum(self.damage)


@dataclass(frozen=True)
class ExpectedOutcome:
    """Espérances de la même séquence."""

    attacks: float
    hits: float
    critical_hits: float
    wounds: float
    critical_wounds: float
    devastating: float
    saves_attempted: float
    unsaved: float
    damage: float  #: dégâts totaux attendus après FNP, avant allocation
    damage_per_wound: float  #: dégâts attendus d'une blessure passée, après FNP


# ------------------------------------------------------------------ seuils


def hit_roll_needed(profile: AttackProfile, rules: RulesConfig = DEFAULT_RULES) -> Optional[int]:
    """Résultat de touche requis après modificateur plafonné ; None pour Torrent."""
    if profile.skill is None:
        return None
    return max(clamp_roll(profile.skill - clamp_modifier(profile.hit_modifier, rules)), profile.min_unmodified_hit)


def wound_roll_needed_for(profile: AttackProfile, defender: Defender, rules: RulesConfig = DEFAULT_RULES) -> int:
    base = wound_roll_needed(profile.strength, defender.toughness)
    return clamp_roll(base - clamp_modifier(profile.wound_modifier, rules))


# ------------------------------------------------------------- probabilités


def _p_at_least(needed: int) -> float:
    """P(D6 ≥ needed) pour needed dans 1..7."""
    needed = max(1, min(7, needed))
    return (7 - needed) / 6


def _with_reroll(p_success: float, p_crit: float, reroll: str):
    """Applique une politique de relance à (P(succès), P(critique)) d'un jet."""
    if reroll == REROLL_ONES:
        again = 1 / 6
    elif reroll == REROLL_FAILS:
        again = 1 - p_success
    else:
        again = 0.0
    return p_success + again * p_success, p_crit + again * p_crit


def _p_fnp(defender: Defender) -> float:
    return _p_at_least(defender.feel_no_pain) if defender.feel_no_pain else 0.0


# ------------------------------------------------------------- aux dés


def _d6(rng: random.Random) -> int:
    return rng.randint(1, 6)


def _roll_with_reroll(rng: random.Random, is_success, reroll: str) -> int:
    roll = _d6(rng)
    if reroll == REROLL_ONES and roll == 1:
        roll = _d6(rng)
    elif reroll == REROLL_FAILS and not is_success(roll):
        roll = _d6(rng)
    return roll


def resolve_attacks(
    profile: AttackProfile,
    defender: Defender,
    rng: random.Random,
    rules: RulesConfig = DEFAULT_RULES,
    defer_saves: bool = False,
) -> AttackOutcome:
    """Résout aux dés toutes les attaques d'un profil contre un défenseur. ``defer_saves`` : s'arrête
    après les blessures (``saves_attempted`` blessures à sauvegarder, ``devastating`` blessures
    dévastatrices) ; l'appelant fait les sauvegardes groupe d'allocation par groupe (V11 05.03)."""
    out = AttackOutcome()
    crit = min(rules.critical_roll, profile.critical_hit_on)
    hit_needed = hit_roll_needed(profile, rules)
    wound_needed = wound_roll_needed_for(profile, defender, rules)
    crit_wound_on = profile.critical_wound_on
    save = save_needed(defender.save, profile.ap, defender.invuln, rules)
    fnp = defender.feel_no_pain

    n_attacks = sum(profile.attacks.roll(rng) for _ in range(profile.count))
    out.attacks = n_attacks

    # --- touches
    hits_to_wound = 0  # touches (normales) qui iront au jet de blessure
    auto_wounds = 0  # Lethal Hits
    for _ in range(n_attacks):
        if hit_needed is None:  # Torrent : touche automatique, jamais critique
            out.hits += 1
            hits_to_wound += 1
            continue
        roll = _roll_with_reroll(rng, lambda r: r >= hit_needed or r >= crit, profile.reroll_hits)
        if roll == 1 or (roll < hit_needed and roll < crit):
            continue
        out.hits += 1
        if roll >= crit:
            out.critical_hits += 1
            if profile.sustained_hits is not None:
                extra = profile.sustained_hits.roll(rng)
                out.hits += extra
                hits_to_wound += extra
            if profile.lethal_hits:
                auto_wounds += 1
                continue
        hits_to_wound += 1

    # --- blessures
    to_save = auto_wounds
    out.wounds = auto_wounds
    for _ in range(hits_to_wound):
        roll = _roll_with_reroll(rng, lambda r: r >= wound_needed or r >= crit_wound_on, profile.reroll_wounds)
        if roll == 1 or (roll < wound_needed and roll < crit_wound_on):
            continue
        out.wounds += 1
        if roll >= crit_wound_on:
            out.critical_wounds += 1
            if profile.devastating_wounds:
                out.devastating += 1
                continue
        to_save += 1

    # --- sauvegardes
    out.saves_attempted = to_save
    if defer_saves:
        return out
    for _ in range(to_save):
        if save is None or _d6(rng) < save:
            out.unsaved += 1

    # --- dégâts (réduction de dégâts, puis FNP point par point)
    for _ in range(out.unsaved + out.devastating):
        dmg = profile.damage.roll(rng)
        if defender.damage_reduction:
            dmg = max(1, dmg - defender.damage_reduction)
        if fnp:
            dmg = sum(1 for _ in range(dmg) if _d6(rng) < fnp)
        out.damage.append(dmg)
    return out


# ------------------------------------------------------------- en espérance


def expected_attacks(
    profile: AttackProfile,
    defender: Defender,
    rules: RulesConfig = DEFAULT_RULES,
) -> ExpectedOutcome:
    """Espérances de la séquence complète, en fermé (même logique que :func:`resolve_attacks`)."""
    n_attacks = profile.count * profile.attacks.mean

    # --- touches
    hit_needed = hit_roll_needed(profile, rules)
    if hit_needed is None:
        p_hit, p_crit_hit = 1.0, 0.0
    else:
        crit = max(2, min(rules.critical_roll, profile.critical_hit_on))
        p_hit, p_crit_hit = _with_reroll(_p_at_least(min(hit_needed, crit)), _p_at_least(crit), profile.reroll_hits)
    hits = n_attacks * p_hit
    crit_hits = n_attacks * p_crit_hit
    extra_hits = crit_hits * profile.sustained_hits.mean if profile.sustained_hits is not None else 0.0
    hits_total = hits + extra_hits
    auto_wounds = crit_hits if profile.lethal_hits else 0.0
    hits_to_wound = hits_total - auto_wounds

    # --- blessures
    wound_needed = wound_roll_needed_for(profile, defender, rules)
    p_wound_base = _p_at_least(min(wound_needed, profile.critical_wound_on))
    p_crit_wound_base = _p_at_least(profile.critical_wound_on)
    p_wound, p_crit_wound = _with_reroll(p_wound_base, p_crit_wound_base, profile.reroll_wounds)
    rolled_wounds = hits_to_wound * p_wound
    crit_wounds = hits_to_wound * p_crit_wound
    wounds = auto_wounds + rolled_wounds
    devastating = crit_wounds if profile.devastating_wounds else 0.0
    to_save = wounds - devastating

    # --- sauvegardes
    save = save_needed(defender.save, profile.ap, defender.invuln, rules)
    p_save = _p_at_least(save) if save is not None else 0.0
    unsaved = to_save * (1 - p_save)

    # --- dégâts
    dmg_per_wound = _expected_damage(profile.damage, defender.damage_reduction) * (1 - _p_fnp(defender))
    return ExpectedOutcome(
        attacks=n_attacks,
        hits=hits_total,
        critical_hits=crit_hits,
        wounds=wounds,
        critical_wounds=crit_wounds,
        devastating=devastating,
        saves_attempted=to_save,
        unsaved=unsaved,
        damage=(unsaved + devastating) * dmg_per_wound,
        damage_per_wound=dmg_per_wound,
    )


def _damage_distribution(damage: DiceExpr) -> Dict[int, float]:
    """Loi exacte d'une caractéristique de dégâts (convolution des dés)."""
    dist = {0: 1.0}
    for _ in range(damage.n_dice):
        nxt: Dict[int, float] = {}
        for v, p in dist.items():
            for face in range(1, damage.sides + 1):
                nxt[v + face] = nxt.get(v + face, 0.0) + p / damage.sides
        dist = nxt
    return {v + damage.flat: p for v, p in dist.items()}


def _expected_damage(damage: DiceExpr, reduction: int = 0) -> float:
    """E[max(1, D − réduction)] ; sans réduction, simplement la moyenne."""
    if not reduction:
        return damage.mean
    return sum(p * max(1, v - reduction) for v, p in _damage_distribution(damage).items())


def estimate_models_slain(wounds_through: float, damage: DiceExpr, model_wounds: int, fnp: Optional[int] = None) -> float:
    """Estimation continue du nombre de figurines tuées par ``wounds_through`` blessures passées.

    Les dégâts d'une blessure ne débordent pas sur une autre figurine : chaque blessure
    inflige au plus ``model_wounds``. Approximation « fluide » (pas d'effet de seuil),
    suffisante pour une fonction d'évaluation ; les bots à simulation n'en ont pas besoin.
    """
    if model_wounds <= 0:
        return 0.0
    p_keep = 1 - (_p_at_least(fnp) if fnp else 0.0)
    if damage.is_fixed:
        eff = min(damage.flat * p_keep, model_wounds)
    else:
        # E[min(D, W)] sur la distribution de D (uniforme par dé, convolution exacte)
        dist = {0: 1.0}
        for _ in range(damage.n_dice):
            nxt = {}
            for v, p in dist.items():
                for face in range(1, damage.sides + 1):
                    nxt[v + face] = nxt.get(v + face, 0.0) + p / damage.sides
            dist = nxt
        eff = sum(p * min((v + damage.flat) * p_keep, model_wounds) for v, p in dist.items())
    return wounds_through * eff / model_wounds


# ------------------------------------------------------------- Hazardous


def hazardous_test(
    rng: random.Random,
    n_weapons: int,
    defender_is_vehicle_or_monster: bool,
    rules: RulesConfig = DEFAULT_RULES,
) -> int:
    """Blessures mortelles subies par l'unité tireuse après avoir tiré ``n_weapons`` armes Hazardous."""
    per_fail = rules.hazardous_mortal_wounds_vehicle_monster if defender_is_vehicle_or_monster else rules.hazardous_mortal_wounds_infantry
    return sum(per_fail for _ in range(n_weapons) if _d6(rng) in rules.hazardous_fail_on)


def expected_hazardous_mortal_wounds(
    n_weapons: int,
    defender_is_vehicle_or_monster: bool,
    rules: RulesConfig = DEFAULT_RULES,
) -> float:
    per_fail = rules.hazardous_mortal_wounds_vehicle_monster if defender_is_vehicle_or_monster else rules.hazardous_mortal_wounds_infantry
    return n_weapons * len(rules.hazardous_fail_on) / 6 * per_fail
