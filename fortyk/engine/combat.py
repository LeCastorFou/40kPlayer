"""Couche combat : du plateau aux profils d'attaque, puis aux dégâts alloués.

Tir (:func:`resolve_shooting`) et mêlée (:func:`resolve_fight`) construisent des
:class:`~fortyk.engine.attack.AttackProfile` par groupe de figurines identiques (même
arme, même couvert), les résolvent aux dés et allouent les dégâts à l'unité visée.
Les mêmes constructeurs servent à l'évaluation en espérance pour l'IA
(:func:`expected_shooting`, :func:`expected_fight`).

Règles appliquées ici (V11, validées) :

* portée mesurée du tireur à la figurine visée la plus proche (socle à socle) ; visibilité
  figurine à figurine (vraie ligne de vue, décors denses opaques) ;
* couvert = -1 à la touche pour un tireur si chaque figurine de la cible est soit dans un décor
  (infanterie : le socle touche une empreinte ; monstre / véhicule : entièrement dans une ruine),
  soit pas pleinement visible de ce tireur ; Ignores Cover annule ;
* Heavy : +1 à la touche si l'unité tireuse a bougé de moins de 3" ; Blast : +1 attaque par
  tranche de 5 figurines dans la cible, interdit sur une unité à portée d'engagement d'une
  unité amie ; Pistol : seules armes utilisables par une unité engagée, sur une cible
  engagée ; une figurine tire soit ses pistolets soit ses autres armes ; Assault : peut tirer
  après une Advance ; Hazardous : test après le tir, 1-2 → blessures mortelles ;
* Hail of Bolts (Intercessors) : +2 A aux bolt rifles sur la cible désignée ;
  Oath of Moment : relance des touches contre la cible du serment ;
  Excessive Assault (Infractors) : en mêlée, relance des blessures de 1, ou de toutes si la
  cible est à portée d'un objectif.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..data.models import Weapon
from ..data.parse import DiceExpr
from .attack import (
    REROLL_FAILS,
    REROLL_NONE,
    REROLL_ONES,
    AttackProfile,
    Defender,
    expected_attacks,
    hazardous_test,
    resolve_attacks,
)
from .geometry import disk_gap, within
from .rules import DEFAULT_RULES, save_needed
from .state import GameState, Model, Unit
from .effects import best_threshold, effect_total, granted_abilities, has_effect, reroll_policy
from .stratagems import effect_of

__all__ = [
    "CombatReport",
    "defender_of",
    "can_shoot",
    "target_locked_in_combat",
    "is_hidden",
    "models_outside_terrain",
    "is_lone_operative",
    "targeting_range_cap",
    "is_vehicle_or_monster",
    "damage_reduction_of",
    "target_engaged_with_friends",
    "shooting_ineligibility",
    "shooting_targets",
    "build_shooting_profiles",
    "resolve_shooting",
    "expected_shooting",
    "fight_targets",
    "build_melee_profiles",
    "resolve_fight",
    "expected_fight",
]


@dataclass
class CombatReport:
    attacker: str
    target: str
    kind: str  #: "shooting" / "fight"
    profiles: int = 0
    attacks: int = 0
    hits: int = 0
    wounds: int = 0
    unsaved: int = 0
    damage: int = 0
    models_slain: int = 0
    target_destroyed: bool = False
    hazardous_mortal_wounds: int = 0
    own_models_lost: int = 0
    details: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        s = (f"{self.attacker} → {self.target} ({self.kind}) : {self.attacks} att., {self.hits} touches, "
             f"{self.wounds} blessures, {self.unsaved} passent, {self.damage} dégâts, {self.models_slain} fig. tuée(s)")
        if self.target_destroyed:
            s += " — UNITÉ DÉTRUITE"
        if self.hazardous_mortal_wounds:
            s += f" ; Hazardous : {self.hazardous_mortal_wounds} BM, {self.own_models_lost} fig. perdue(s)"
        return s


_DAMAGE_REDUCTION_RE = re.compile(r"each time an attack is allocated to (?:this model|a model in this unit),\s*subtract (\d+|one) from the damage characteristic of that attack", re.IGNORECASE)


def damage_reduction_of(unit: Unit) -> int:
    """« Each time an attack is allocated to this model, subtract 1 from the Damage characteristic »
    (Duty Eternal des Dreadnoughts, etc.), lu dans le texte des capacités de la fiche."""
    total = 0
    for a in unit.datasheet.abilities:
        m = _DAMAGE_REDUCTION_RE.search(a.text or "")
        if m:
            total += 1 if m.group(1).lower() == "one" else int(m.group(1))
    return total


def is_close_quarters(weapon: Weapon) -> bool:
    """[CLOSE-QUARTERS] (24.07) : [PISTOL] et [CLOSE-QUARTERS] sont identiques pour toutes les règles (24.27)."""
    return weapon.has("pistol") or weapon.has("close-quarters")


def is_vehicle_or_monster(unit: Unit) -> bool:
    return unit.has_keyword("Vehicle") or unit.has_keyword("Monster")


def unit_toughness(unit: Unit) -> int:
    """19.02 : E la plus haute des gardes du corps de l'unité (celle des personnages s'il n'en reste plus)."""
    guards = [m for m in unit.alive_models if not m.is_leader] or unit.alive_models
    return max((m.profile.toughness for m in guards), default=unit.profile.toughness)


def defender_of(unit: Unit, state: Optional[GameState] = None) -> Defender:
    """Caractéristiques défensives de l'unité ; avec ``state``, les effets actifs (Feel No Pain,
    invulnérable, réduction de dégâts, sauvegarde améliorée) sont pris en compte."""
    p = unit.profile
    fnp = None
    for a in unit.datasheet.core_abilities:
        if a.name == "Feel No Pain" and a.parameter.rstrip("+").isdigit():
            fnp = int(a.parameter.rstrip("+"))
    save, invuln, reduction = p.save, p.invuln, damage_reduction_of(unit)
    if state is not None and state.effects:
        fnp = best_threshold(state, unit.id, "fnp", fnp)
        invuln = best_threshold(state, unit.id, "invuln", invuln)
        reduction += effect_total(state, unit.id, "damage_reduction")
        save = max(2, save - effect_total(state, unit.id, "save_mod"))
    toughness = unit_toughness(unit)
    if state is not None and state.effects:
        toughness += effect_total(state, unit.id, "toughness_mod")
    return Defender(
        toughness=toughness,
        save=save,
        invuln=invuln,
        feel_no_pain=fnp,
        is_vehicle_or_monster=is_vehicle_or_monster(unit),
        damage_reduction=reduction,
    )


# ------------------------------------------------------------------ tir


def _thrill_seekers(unit: Unit) -> bool:
    return unit.has_ability("Thrill Seekers")


def _fall_back_shoot(state: GameState, unit: Unit) -> bool:
    return _thrill_seekers(unit) or (bool(state.effects) and has_effect(state, unit.id, "fall_back_and_shoot"))


def _assault_all(state: GameState, unit: Unit) -> bool:
    """Toutes les armes de tir comptent comme [ASSAULT] (« can shoot even if it made an Advance »)."""
    return bool(state.effects) and (has_effect(state, unit.id, "advance_and_shoot") or "assault" in granted_abilities(state, unit.id, "ranged"))


def can_shoot(state: GameState, unit: Unit) -> bool:
    if unit.is_destroyed or unit.has_shot:
        return False
    if unit.fell_back and not _fall_back_shoot(state, unit):
        return False
    return True


def _thrill_seekers_forbids(state: GameState, unit: Unit, target: Unit) -> bool:
    """Après Advance ou Fall Back, Thrill Seekers interdit de viser une unité dont on était à portée
    d'engagement au début du tour, ou déjà visée par une autre unité cette phase."""
    if not (unit.advanced or unit.fell_back) or not _thrill_seekers(unit):
        return False
    return target.id in unit.engaged_at_turn_start or target.id in state.charge_targets_this_phase


def _big_guns(state: GameState, unit: Unit) -> bool:
    """Big Guns Never Tire (validé V11) : les MONSTER / VEHICLE tirent malgré la portée d'engagement."""
    return state.rules.big_guns_never_tire and is_vehicle_or_monster(unit)


def _weapon_usable(unit: Unit, weapon: Weapon, engaged: bool, big_guns: bool = False, assault_all: bool = False) -> bool:
    if weapon.is_melee:
        return False
    if engaged and not is_close_quarters(weapon) and not big_guns:
        return False
    if unit.advanced and not weapon.has("assault") and not _thrill_seekers(unit) and not assault_all:
        return False
    return True


def _unit_has_cover_from(state: GameState, shooter: Model, target: Unit) -> bool:
    """Couvert (V11 13.08), évalué pour ce tireur : chaque figurine visée est soit INFANTRY / BEASTS /
    SWARM dans une zone de terrain (le socle touche l'empreinte), soit pas pleinement visible du tireur
    à cause des décors. Un MONSTER / VEHICLE n'a donc le couvert que s'il n'est pas pleinement visible.

    Les cartes ne donnent pas les murs des ruines : une figurine entièrement dans l'empreinte d'une
    ruine est considérée comme partiellement masquée par ses murs (règle validée avec Valentin).
    Smokescreen (15.10) : l'unité SMOKE visée a le couvert jusqu'à la fin de la phase ; un effet
    « cover » (stratagème, capacité) aussi."""
    if state.strat_used and effect_of(state, "smokescreen", target.id):
        return True
    if state.effects and has_effect(state, target.id, "cover", "ranged"):
        return True
    in_area_ok = any(target.has_keyword(k) for k in state.rules.cover_keywords) and not is_vehicle_or_monster(target)
    for m in target.alive_models:
        if in_area_ok and state.layout.disk_touches_area(m.disk):
            continue
        if state.layout.ruin_containing_disk(m.disk) is not None:
            continue
        if not state.los.fully_visible(shooter.disk, m.disk):
            continue
        return False
    return True


def shot_recently(state: GameState, unit: Unit) -> bool:
    """L'unité a fait des attaques de tir pendant ce tour ou le tour précédent (13.09)."""
    return unit.last_shot_turn >= state.turn_counter - 1


def hidden_model_ids(state: GameState, target: Unit) -> set:
    """Figurines cachées (V11 13.09) : INFANTRY / BEASTS / SWARM dans une zone de terrain contenant un
    décor dense (un orteil suffit), si l'unité n'a pas tiré ce tour ni au tour précédent. Une figurine
    cachée n'est visible que des ennemis à 15" ou moins."""
    if not any(target.has_keyword(k) for k in state.rules.hidden_keywords) or is_vehicle_or_monster(target):
        return set()
    if shot_recently(state, target):
        return set()
    return {m.id for m in target.alive_models if state.layout.disk_touches_dense_area(m.disk)}


def is_hidden(state: GameState, target: Unit) -> bool:
    """Toute l'unité est cachée : elle n'est visible que des figurines à 15" ou moins."""
    hidden = hidden_model_ids(state, target)
    return bool(hidden) and len(hidden) == target.strength


def models_outside_terrain(state: GameState, target: Unit) -> List[str]:
    """Figurines qui empêchent une unité d'être entièrement cachée alors qu'une partie l'est :
    identifiants des socles qui ne touchent aucune zone dense. Liste vide si l'unité est cachée,
    n'y a pas droit, ou est totalement hors décor dense."""
    hidden = hidden_model_ids(state, target)
    if not hidden or len(hidden) == target.strength:
        return []
    return [m.id for m in target.alive_models if m.id not in hidden]


def is_lone_operative(state: GameState, target: Unit) -> bool:
    """Lone Operative : ciblable au tir seulement à 12" ou moins. Capacité Core, ou conditionnelle
    comme Lord of Excess (Daemon Prince of Slaanesh à 3" d'une unité amie Slaanesh Infantry)."""
    if any(a.name == "Lone Operative" for a in target.datasheet.core_abilities):
        return True
    if state.effects and has_effect(state, target.id, "lone_operative"):
        return True
    if target.has_ability("Lord of Excess"):
        for friend in state.units_of(target.side):
            if friend.id != target.id and friend.has_keyword("Slaanesh") and friend.has_keyword("Infantry") and target.min_gap_to(friend) <= 3.0:
                return True
    return False


def targeting_range_cap(state: GameState, target: Unit, hidden=None) -> Optional[float]:
    """Portée maximale à laquelle la cible peut être visée au tir (Lone Operative 12", y compris par
    [INDIRECT FIRE]) ; None si aucune limite. Hidden se traite figurine par figurine
    (:func:`hidden_model_ids`)."""
    if is_lone_operative(state, target):
        return state.rules.lone_operative_range_in
    return None


_UNSET = object()


def _visible_targets_in_range(state: GameState, shooter: Model, weapon: Weapon, target: Unit, hidden=None,
                              cap=_UNSET, half_range: bool = False, indirect: bool = False) -> bool:
    """Au moins une figurine de la cible à portée de l'arme (plafonnée par Lone Operative ; ``cap`` :
    plafond déjà calculé, None = aucun) et visible du tireur — une figurine cachée ne l'est qu'à 15"
    (``hidden`` : identifiants déjà calculés). ``indirect`` : [INDIRECT FIRE], la vue n'est pas requise.
    ``half_range`` : teste la demi-portée (Rapid Fire, Melta)."""
    rng = weapon.range_in or 0.0
    if half_range:
        rng = rng / 2
    if cap is _UNSET:
        cap = targeting_range_cap(state, target)
    if cap is not None:
        rng = min(rng, cap)
    if hidden is None or isinstance(hidden, bool):
        hidden = hidden_model_ids(state, target)
    detect = state.rules.hidden_range_in
    sd = shooter.disk
    los = state.los
    for m in target.models:
        if not m.alive:
            continue
        if sd.is_round and m.hx == 0.0 and m.hy == 0.0:
            gap = math.hypot(sd.x - m.x, sd.y - m.y) - sd.r - m.radius
        else:
            gap = disk_gap(sd, m.disk)
        if gap > rng + 1e-9:
            continue
        if indirect:
            return True
        if m.id in hidden and gap > detect + 1e-9:
            continue
        if los.visible(sd, m.disk):
            return True
    return False


def _can_shoot_indirect(state: GameState, unit: Unit) -> bool:
    """10.07 : tir indirect si l'unité est désengagée et n'a pas fait d'Advance ce tour."""
    return not unit.advanced and not state.is_engaged(unit)


def _spotted(state: GameState, unit: Unit, target: Unit) -> bool:
    """La cible est visible d'au moins une unité amie (tir indirect, 10.07) : une figurine cachée ne
    l'est qu'à portée de détection (15"), un Lone Operative qu'à 12"."""
    los = state.los
    hidden = hidden_model_ids(state, target)
    detect = state.rules.hidden_range_in
    cap = targeting_range_cap(state, target)
    for f in state.units_of(unit.side):
        for fm in f.alive_models:
            for tm in target.alive_models:
                if (tm.id in hidden and not within(fm.disk, tm.disk, detect)) or (cap is not None and not within(fm.disk, tm.disk, cap)):
                    continue
                if los.visible(fm.disk, tm.disk):
                    return True
    return False


def _choose_profile(profiles: List[Weapon], target: Unit, count: int = 1) -> Weapon:
    """Pour une arme à plusieurs profils, garde celui qui maximise les dégâts attendus."""
    if len(profiles) == 1:
        return profiles[0]
    dfd = defender_of(target)
    best, best_val = profiles[0], -1.0
    for w in profiles:
        p = AttackProfile(w.attacks, None if w.has("torrent") else w.skill, w.strength, w.ap, w.damage, count=count,
                          devastating_wounds=w.has("devastating wounds"), lethal_hits=w.has("lethal hits"))
        val = expected_attacks(p, dfd).damage
        if val > best_val:
            best, best_val = w, val
    return best


def target_locked_in_combat(state: GameState, unit: Unit, target: Unit) -> bool:
    """Une unité ennemie à portée d'engagement d'une unité amie ne peut pas être visée au tir,
    sauf par les pistolets de l'unité qui l'engage elle-même — et sauf si la cible est un
    MONSTER / VEHICLE (Big Guns Never Tire : visable, avec -1 à la touche hors Pistol)."""
    for friend in state.units_of(unit.side):
        if target.in_engagement_range_of(friend, state.rules):
            if friend.id == unit.id:
                return False
            return not _big_guns(state, target)
    return False


def target_engaged_with_friends(state: GameState, unit: Unit, target: Unit) -> bool:
    """La cible est-elle à portée d'engagement d'une unité du camp du tireur (dont lui-même) ?"""
    return any(target.in_engagement_range_of(f, state.rules) for f in state.units_of(unit.side))


def shooting_ineligibility(state: GameState, unit: Unit) -> Optional[str]:
    """Pourquoi l'unité ne peut pas tirer du tout (None si elle peut)."""
    if unit.is_destroyed:
        return "détruite"
    if unit.has_shot:
        return "a déjà tiré"
    if unit.fell_back and not _fall_back_shoot(state, unit):
        return "s'est repliée ce tour"
    if not any(m.ranged_weapons for m in unit.alive_models):
        return "aucune arme de tir"
    if not shooting_targets(state, unit):
        hidden = [e.id for e in state.enemies_of(unit.side) if is_hidden(state, e)]
        extra = ""
        if state.is_engaged(unit):
            extra = " (engagée : pistolets seulement, sur l'unité engagée)" if not _big_guns(state, unit) else " (engagée : cibles à portée d'engagement seulement)"
        elif hidden:
            extra = f" ({', '.join(hidden)} : Hidden, dans un décor dense, inciblable à plus de 15\")"
        return "aucune cible visible et à portée" + extra
    return None


def _prefetch_los(state: GameState, unit: Unit, target: Unit, engaged: bool, big_guns: bool, cap: Optional[float], all_targets: bool = False,
                  assault_all: bool = False) -> None:
    """Pré-calcule en un lot les lignes de vue utiles : tireurs ayant une arme utilisable qui porte
    jusqu'à au moins une figurine de la cible, contre les figurines à portée (ou toutes, pour le
    couvert : ``all_targets``)."""
    tdisks = target.disks()
    shooters, targets = [], {}
    for m in unit.alive_models:
        reach = max((w.range_in or 0.0 for w in m.ranged_weapons if _weapon_usable(unit, w, engaged, big_guns, assault_all)), default=0.0)
        if cap is not None:
            reach = min(reach, cap)
        md = m.disk
        near = [d for d in tdisks if disk_gap(md, d) <= reach + 1e-9]
        if near:
            shooters.append(md)
            for d in (tdisks if all_targets else near):
                targets[tuple(d)] = d
    if shooters:
        state.los.prefetch(shooters, list(targets.values()))


def shooting_targets(state: GameState, unit: Unit) -> List[Unit]:
    """Unités ennemies que ``unit`` peut viser au tir avec au moins une arme."""
    if not can_shoot(state, unit):
        return []
    engaged = state.is_engaged(unit)
    big_guns = _big_guns(state, unit)
    assault_all = _assault_all(state, unit)
    out = []
    for enemy in state.enemies_of(unit.side):
        if engaged and not unit.in_engagement_range_of(enemy, state.rules):
            continue
        if target_locked_in_combat(state, unit, enemy):
            continue
        if _thrill_seekers_forbids(state, unit, enemy):
            continue
        cap = targeting_range_cap(state, enemy)
        _prefetch_los(state, unit, enemy, engaged, big_guns, cap, assault_all=assault_all)
        ok = False
        hidden = hidden_model_ids(state, enemy)
        indirect_ok = _can_shoot_indirect(state, unit)
        for m in unit.alive_models:
            for w in m.ranged_weapons:
                if not _weapon_usable(unit, w, engaged, big_guns, assault_all):
                    continue
                if _visible_targets_in_range(state, m, w, enemy, hidden=hidden, cap=cap) or (
                        indirect_ok and w.has("indirect fire") and _visible_targets_in_range(state, m, w, enemy, cap=cap, indirect=True)):
                    ok = True
                    break
            if ok:
                break
        if ok:
            out.append(enemy)
    return out


def snap_targets(state: GameState, unit: Unit, max_range: float = 24.0) -> List[Unit]:
    """Cibles d'un tir d'opportunité (15.09) : unités ennemies à ``max_range`` de l'unité, visées selon
    les règles normales et visibles (le tir indirect ne compte pas)."""
    out = []
    engaged = state.is_engaged(unit)
    big_guns = _big_guns(state, unit)
    for enemy in shooting_targets(state, unit):
        if unit.min_gap_to(enemy) > max_range + 1e-9:
            continue
        cap = targeting_range_cap(state, enemy)
        hidden = hidden_model_ids(state, enemy)
        if any(_weapon_usable(unit, w, engaged, big_guns, _assault_all(state, unit)) and _visible_targets_in_range(state, m, w, enemy, hidden=hidden, cap=cap)
               for m in unit.alive_models for w in m.ranged_weapons):
            out.append(enemy)
    return out


def explosives_targets(state: GameState, unit: Unit, max_range: float = 8.0) -> List[Unit]:
    """Explosives (15.05) : unités ennemies désengagées à 8" d'une figurine de l'unité et visibles
    d'elle (une figurine cachée l'est à cette distance)."""
    out = []
    for enemy in state.enemies_of(unit.side):
        if state.is_engaged(enemy):
            continue
        if any(within(m.disk, e.disk, max_range) and state.los.visible(m.disk, e.disk) for m in unit.alive_models for e in enemy.alive_models):
            out.append(enemy)
    return out


def build_shooting_profiles(state: GameState, unit: Unit, target: Unit, snap: bool = False) -> Tuple[List[AttackProfile], int]:
    """Profils d'attaque du tir de ``unit`` sur ``target`` et nombre d'armes Hazardous utilisées.
    ``snap`` : tir d'opportunité (15.09) — cible visible uniquement (pas de tir indirect), chaque
    attaque ne touche que sur un 6 non modifié, pas de relance des touches."""
    rules = state.rules
    engaged = state.is_engaged(unit)
    blast_forbidden = any(target.in_engagement_range_of(f, rules) for f in state.units_of(unit.side))
    # [HEAVY] (24.16) : désengagée, pas posée sur la table ce tour (débarquement), aucune figurine n'a bougé de plus de 3"
    heavy_ok = not engaged and not unit.disembarked and unit.moved_in <= rules.heavy_move_threshold_in + 1e-9
    heavy_bonus = rules.heavy_hit_bonus if heavy_ok else 0
    oath = state.oath_target == target.id and unit.has_ability("Oath of Moment")
    hail = unit.has_ability("Hail of Bolts")
    dfd = defender_of(target, state)
    assault_all = _assault_all(state, unit)
    granted = granted_abilities(state, unit.id, "ranged", target.id) if state.effects else []
    cap = targeting_range_cap(state, target)
    big_guns = _big_guns(state, unit)
    # Big Guns Never Tire : -1 à la touche (hors Pistol) si le tireur M/V est engagé, ou si la cible
    # M/V est à portée d'engagement d'une unité du camp du tireur
    big_guns_penalty = 0
    if (big_guns and engaged) or (_big_guns(state, target) and target_engaged_with_friends(state, unit, target)):
        big_guns_penalty = rules.big_guns_hit_penalty

    groups: Dict[Tuple, int] = defaultdict(int)
    hazardous_count = 0
    _prefetch_los(state, unit, target, engaged, big_guns, cap, all_targets=True, assault_all=assault_all)
    hidden = hidden_model_ids(state, target)
    # tir indirect (10.07) : dès qu'une arme [INDIRECT FIRE] tire sans voir la cible, l'unité tire en
    # « indirect shooting » et toutes ses attaques [INDIRECT FIRE] en subissent les règles
    indirect_ok = _can_shoot_indirect(state, unit) and not snap
    seen: Dict[str, List[Weapon]] = {}
    blind: Dict[str, List[Weapon]] = {}
    for m in unit.alive_models:
        for w in m.ranged_weapons:
            if not _weapon_usable(unit, w, engaged, big_guns, assault_all):
                continue
            if _visible_targets_in_range(state, m, w, target, hidden=hidden, cap=cap):
                seen.setdefault(m.id, []).append(w)
            elif indirect_ok and w.has("indirect fire") and _visible_targets_in_range(state, m, w, target, cap=cap, indirect=True):
                blind.setdefault(m.id, []).append(w)
    indirect_mode = bool(blind)
    indirect_min = 0
    if indirect_mode:
        spotted = unit.remained_stationary and _spotted(state, unit, target)
        indirect_min = rules.indirect_fail_below_spotted if spotted else rules.indirect_fail_below
    for m in unit.alive_models:
        usable = seen.get(m.id, []) + blind.get(m.id, [])
        if not usable:
            continue
        # une figurine tire soit ses pistolets, soit ses autres armes : on prend le lot le plus rentable
        pistols = [w for w in usable if is_close_quarters(w)]
        others = [w for w in usable if not is_close_quarters(w)]
        if pistols and others:
            def value(ws):
                return sum(expected_attacks(AttackProfile(w.attacks, None if w.has("torrent") else w.skill, w.strength, w.ap, w.damage), dfd).damage for w in ws)
            chosen = pistols if value(pistols) > value(others) else others
        else:
            chosen = usable
        # un profil par arme (plasma standard / surcharge) ; une arme portée en plusieurs exemplaires
        # (2 excruciator cannons) apparaît plusieurs fois dans ``m.weapons`` et tire autant de fois
        by_group: Dict[str, List[Weapon]] = defaultdict(list)
        for w in chosen:
            by_group[w.group].append(w)
        cover = _unit_has_cover_from(state, m, target)
        for group, all_profiles in by_group.items():
            profiles = list(dict.fromkeys(all_profiles))  # profils distincts, dans l'ordre
            copies = max(1, len(all_profiles) // len(profiles))
            w = _choose_profile(profiles, target)
            if w.has("blast") and blast_forbidden:
                continue
            hit_mod = 0
            if w.has("heavy") or "heavy" in granted:
                hit_mod += heavy_bonus
            indirect_w = indirect_mode and w.has("indirect fire")
            skill_worse = rules.cover_skill_penalty if ((cover or indirect_w) and not (w.has("ignores cover") or "ignores cover" in granted)) else 0
            if big_guns_penalty and not is_close_quarters(w):
                hit_mod -= big_guns_penalty
            extra_attacks = 0
            if w.has("blast"):
                extra_attacks += target.strength // rules.blast_models_per_extra_attack
            rapid = w.keyword("rapid fire")
            if rapid is not None and isinstance(rapid.value, int) and _visible_targets_in_range(state, m, w, target, cap=cap, half_range=True):
                extra_attacks += rapid.value
            if hail and w.group.lower() == "bolt rifle":
                extra_attacks += 2
            if w.has("hazardous"):
                hazardous_count += copies
            key = (w.name, hit_mod, extra_attacks, skill_worse, indirect_min if indirect_w else 0)
            groups[key] += copies

    weapons_by_name = {w.name: w for m in unit.alive_models for w in m.weapons}
    mods = _attack_mods(state, unit, target, "ranged")
    profiles = []
    for (name, hit_mod, extra_attacks, skill_worse, min_hit), count in groups.items():
        w = weapons_by_name[name]
        if snap:
            min_hit = state.rules.critical_roll  # 15.09 : seul un 6 non modifié touche
        attacks = DiceExpr(w.attacks.n_dice, w.attacks.sides, w.attacks.flat + extra_attacks + mods["attacks"]) if (extra_attacks or mods["attacks"]) else w.attacks
        base_reroll = REROLL_FAILS if (oath and not min_hit) else REROLL_NONE  # pas de relance en tir indirect ni d'opportunité
        profiles.append(_profile(state, unit, target, w, mods, granted, count=count, attacks=attacks,
                                 skill=None if w.has("torrent") else min(7, w.skill + skill_worse), hit_mod=hit_mod, min_hit=min_hit,
                                 reroll_hits=REROLL_NONE if min_hit else base_reroll, reroll_wounds=REROLL_NONE, scope="ranged", snap=snap,
                                 label=f"{w.name} ×{count}" + (f" ({hit_mod:+d} touche)" if hit_mod else "") + (" (couvert : CT -1)" if skill_worse else "")
                                 + ((" (tir d'opportunité : 6 non modifié)" if snap else f" (indirect : {min_hit}+ non modifié)") if min_hit else "")))
    return profiles, hazardous_count


def _attack_mods(state: GameState, unit: Unit, target: Unit, scope: str) -> Dict[str, int]:
    """Modificateurs d'effets (stratagèmes, capacités, effets manuels) pour les attaques de ``unit``
    contre ``target`` : ceux de l'attaquant et ceux « contre » la cible."""
    z = {"hit": 0, "wound": 0, "ap": 0, "damage": 0, "strength": 0, "attacks": 0}
    if not state.effects:
        return z
    u, t = unit.id, target.id
    z["hit"] = effect_total(state, u, "hit_mod", scope, t) + effect_total(state, t, "hit_mod_against", scope, u)
    z["wound"] = effect_total(state, u, "wound_mod", scope, t) + effect_total(state, t, "wound_mod_against", scope, u)
    z["ap"] = effect_total(state, u, "ap_mod", scope, t) - effect_total(state, t, "ap_worsen", scope, u)
    z["damage"] = effect_total(state, u, "damage_mod", scope, t)
    z["strength"] = effect_total(state, u, "strength_mod", scope, t)
    z["attacks"] = effect_total(state, u, "attacks_mod", scope, t)
    return z


def _granted_value(granted: List[str], key: str):
    """Valeur d'un mot-clé gagné (« sustained hits 2 » → 2, « anti-infantry 4+ » → ('infantry', 4))."""
    for g in granted:
        if g.startswith(key):
            rest = g[len(key):].strip(" -")
            return rest or True
    return None


def _profile(state: GameState, unit: Unit, target: Unit, w: Weapon, mods: Dict[str, int], granted: List[str], *, count: int, attacks, skill,
             hit_mod: int, min_hit: int, reroll_hits: str, reroll_wounds: str, scope: str, label: str, snap: bool = False,
             wound_mod: int = 0, ap_bonus: int = 0, precision: bool = False) -> AttackProfile:
    """Profil d'attaque d'une arme avec les mots-clés de la fiche, les effets actifs et le contexte."""
    has = (lambda k: w.has(k) or any(g == k or g.startswith(k + " ") for g in granted))
    sustained = w.keyword("sustained hits")
    sustained_value = None
    if sustained is not None:
        v = sustained.value if sustained.value is not None else 1
        sustained_value = v if isinstance(v, DiceExpr) else DiceExpr.fixed(int(v))
    extra_sus = _granted_value(granted, "sustained hits")
    if extra_sus is not None and sustained_value is None:
        sustained_value = DiceExpr.fixed(int(extra_sus)) if str(extra_sus).isdigit() else DiceExpr.fixed(1)
    anti = w.keyword("anti")
    crit_on = anti.value if (anti is not None and anti.target and target.has_keyword(anti.target) and isinstance(anti.value, int)) else 6
    for g in granted:  # « anti-infantry 4+ » gagné
        m = re.match(r"anti-(.+?)\s+(\d)\+?$", g)
        if m and target.has_keyword(m.group(1)):
            crit_on = min(crit_on, int(m.group(2)))
    rr_hits, rr_wounds = reroll_hits, (REROLL_FAILS if has("twin-linked") else reroll_wounds)
    crit_hit = 6
    if state.effects:
        if not snap and not min_hit:
            rr_hits = reroll_policy(rr_hits, state, unit.id, "reroll_hits", scope, target.id)
        rr_wounds = reroll_policy(rr_wounds, state, unit.id, "reroll_wounds", scope, target.id)
        crit_hit = best_threshold(state, unit.id, "crit_hit_on", 6, scope, target.id) if not snap else 6
    damage = w.damage
    if mods["damage"]:
        damage = DiceExpr(w.damage.n_dice, w.damage.sides, w.damage.flat + mods["damage"])
    ap = w.ap - ap_bonus - mods["ap"]
    if state.effects:  # « if the Strength of that attack is greater than your unit's Toughness » : arme par arme
        wound_mod += effect_total(state, target.id, "wound_mod_against", scope, unit.id, strength=w.strength + mods["strength"]) \
            - effect_total(state, target.id, "wound_mod_against", scope, unit.id)
    if skill is not None and state.effects:
        skill = max(2, skill - effect_total(state, unit.id, "skill_mod", scope, target.id))
    return AttackProfile(
        attacks=attacks, skill=skill, strength=w.strength + mods["strength"], ap=min(0, ap), damage=damage, count=count,
        hit_modifier=hit_mod + mods["hit"], wound_modifier=wound_mod + mods["wound"], min_unmodified_hit=min_hit,
        reroll_hits=rr_hits, reroll_wounds=rr_wounds, lethal_hits=has("lethal hits"), sustained_hits=sustained_value,
        devastating_wounds=has("devastating wounds"), critical_wound_on=crit_on, precision=precision or has("precision"),
        critical_hit_on=crit_hit, label=label,
    )


def is_character_model(unit: Unit, model: Model) -> bool:
    """Figurine PERSONNAGE : un personnage qui mène l'unité, ou toute figurine d'une unité PERSONNAGE seule."""
    if model.is_leader:
        return True
    return not unit.leaders and unit.has_keyword("Character")


def _is_wounded(model: Model) -> bool:
    return model.wounds < model.profile.wounds


def _save_target(model: Model, ap: int, rules, save_bonus: int = 0, invuln: Optional[int] = None) -> Optional[int]:
    inv = model.profile.invuln if invuln is None else (invuln if model.profile.invuln is None else min(invuln, model.profile.invuln))
    return save_needed(max(2, model.profile.save - save_bonus), ap, inv, rules)


def allocation_groups(unit: Unit, ap: int = 0, rules=None) -> List[List[Model]]:
    """Groupes d'allocation dans l'ordre d'allocation (V11 05.03, étapes 1 et 2).

    Un groupe par figurine PERSONNAGE ; un groupe pour toutes les autres figurines de mêmes PV, Sv
    et InSv. Ordre imposé : le groupe non-PERSONNAGE qui contient une figurine blessée d'abord, les
    PERSONNAGES en dernier (les blessés avant les autres). Le reste de l'ordre est un choix du
    défenseur ; heuristique : la meilleure sauvegarde contre cette PA d'abord (comme dans l'exemple
    du livre), puis le groupe le plus nombreux."""
    rules = rules or DEFAULT_RULES
    alive = unit.alive_models
    others: Dict[tuple, List[Model]] = {}
    chars: List[List[Model]] = []
    for m in alive:
        if is_character_model(unit, m):
            chars.append([m])
        else:
            others.setdefault((m.profile.wounds, m.profile.save, m.profile.invuln), []).append(m)

    def need(g):
        n = _save_target(g[0], ap, rules)
        return 7 if n is None else n

    non = sorted(others.values(), key=lambda g: (not any(_is_wounded(m) for m in g), need(g), -len(g)))
    chars.sort(key=lambda g: not _is_wounded(g[0]))
    return non + chars


def _select_in_group(group: List[Model]) -> Optional[Model]:
    """05.04 étape 1 : une figurine blessée du groupe si possible ; sinon la dernière (le chef
    d'escouade, en tête de liste, part en dernier)."""
    alive = [m for m in group if m.alive]
    if not alive:
        return None
    wounded = [m for m in alive if _is_wounded(m)]
    return wounded[0] if wounded else alive[-1]


def _precision_group(state: GameState, target: Unit, groups: List[List[Model]], attackers: List[Model]) -> Optional[List[Model]]:
    """[PRECISION] (24.28) : un groupe PERSONNAGE visible d'au moins un attaquant (choix de l'attaquant :
    le personnage blessé d'abord, sinon le premier). Une figurine cachée ne compte que si
    l'attaquant est à portée de détection."""
    hidden = hidden_model_ids(state, target)
    detection = state.rules.hidden_range_in
    for g in groups:
        m = g[0]
        if not is_character_model(target, m) or not m.alive:
            continue
        for a in attackers:
            if not a.alive:
                continue
            if m.id in hidden and not within(a.disk, m.disk, detection):
                continue
            if state.los.visible(a.disk, m.disk):
                return g
    return None


def apply_mortal_wounds(state: GameState, unit: Unit, amount: int) -> Tuple[int, int]:
    """Blessures mortelles (06.02) : Feel No Pain blessure par blessure (celui de la fiche, ou un effet,
    y compris « contre les blessures mortelles »), puis allocation qui déborde de figurine en figurine.
    Retourne (figurines perdues, blessures évitées)."""
    if amount <= 0:
        return 0, 0
    fnp = defender_of(unit, state).feel_no_pain
    if state.effects:
        fnp = best_threshold(state, unit.id, "fnp_mortal", fnp)
    kept = amount
    if fnp:
        kept = sum(1 for _ in range(amount) if state.rng.randint(1, 6) < fnp)
    return (unit.allocate_mortal_wounds(kept) if kept else 0), amount - kept


def _lose_wounds(state: GameState, model: Model, amount: int, fnp: Optional[int]) -> Tuple[int, bool]:
    """Feel No Pain (24.12) blessure par blessure, puis perte des PV. Retourne (PV perdus, détruite)."""
    if fnp:
        amount = sum(1 for _ in range(amount) if state.rng.randint(1, 6) < fnp)
    if amount <= 0:
        return 0, False
    lost = min(amount, model.wounds)
    return lost, model.take_damage(amount)


def _resolve_profiles(state: GameState, unit: Unit, target: Unit, profiles: List[AttackProfile], kind: str,
                      attackers: Optional[List[Model]] = None) -> CombatReport:
    """Séquence d'attaque V11 (05) profil par profil (chaque profil = un lot de dés d'attaque
    identiques) : touches et blessures, puis sauvegardes par groupes d'allocation (05.03) — les
    jets de sauvegarde sont résolus du plus faible au plus fort, chacun contre la Sv / InSv du
    groupe courant — et enfin les blessures mortelles des [DEVASTATING WOUNDS] (une figurine au
    plus par blessure critique, après les dégâts normaux)."""
    report = CombatReport(unit.name, target.name, kind, profiles=len(profiles))
    rules = state.rules
    dfd = defender_of(target, state)
    save_bonus = effect_total(state, target.id, "save_mod") if state.effects else 0
    inv_bonus = best_threshold(state, target.id, "invuln", None) if state.effects else None
    fnp = dfd.feel_no_pain
    attackers = attackers if attackers is not None else unit.alive_models
    dead_before = sum(1 for m in target.models if not m.alive)
    total_damage = 0
    for p in profiles:
        if target.is_destroyed:
            break
        out = resolve_attacks(p, dfd, state.rng, rules, defer_saves=True)
        report.attacks += out.attacks
        report.hits += out.hits
        report.wounds += out.wounds
        groups = allocation_groups(target, p.ap, rules)
        forced = _precision_group(state, target, groups, attackers) if p.precision else None
        precision_used = forced is not None
        rolls = sorted(state.rng.randint(1, 6) for _ in range(out.saves_attempted))
        unsaved = 0
        for r in rolls:
            if forced is not None and not any(m.alive for m in forced):
                forced = None
            group = forced if forced is not None else next((g for g in groups if any(m.alive for m in g)), None)
            if group is None:
                break  # unité détruite : les attaques restantes sont perdues
            model = _select_in_group(group)
            need = _save_target(model, p.ap, rules, save_bonus, inv_bonus)
            if r != 1 and need is not None and r >= need:
                continue
            unsaved += 1
            dmg = p.damage.roll(state.rng)
            if dfd.damage_reduction:
                dmg = max(1, dmg - dfd.damage_reduction)
            lost, _ = _lose_wounds(state, model, dmg, fnp)
            total_damage += lost
        devastating = 0
        for _ in range(out.devastating):
            model = target.mortal_wound_target()
            if model is None:
                break
            devastating += 1
            lost, _ = _lose_wounds(state, model, p.damage.roll(state.rng), fnp)  # l'excédent est perdu
            total_damage += lost
        report.unsaved += unsaved + devastating
        report.details.append(f"{p.label}: {out.attacks} att. → {out.hits} touches → {out.wounds} bless. → {unsaved + devastating} passent"
                              + (" (Precision)" if precision_used else ""))
    report.damage = total_damage
    report.models_slain = sum(1 for m in target.models if not m.alive) - dead_before
    report.target_destroyed = target.is_destroyed
    return report


def resolve_shooting(state: GameState, unit: Unit, target: Unit, snap: bool = False) -> CombatReport:
    profiles, hazardous_count = build_shooting_profiles(state, unit, target, snap=snap)
    report = _resolve_profiles(state, unit, target, profiles, "shooting")
    unit.has_shot = True
    unit.last_shot_turn = state.turn_counter  # Hidden (13.09) : a fait des attaques à distance ce tour
    if hazardous_count:
        mw = hazardous_test(state.rng, hazardous_count, unit.has_keyword("Vehicle") or unit.has_keyword("Monster"), state.rules)
        if mw:
            report.hazardous_mortal_wounds = mw
            report.own_models_lost = apply_mortal_wounds(state, unit, mw)[0] if state.rev >= 3 else unit.allocate_mortal_wounds(mw)
    return report


def expected_shooting(state: GameState, unit: Unit, target: Unit) -> float:
    """Dégâts attendus du tir de ``unit`` sur ``target`` (pour l'évaluation)."""
    profiles, _ = build_shooting_profiles(state, unit, target)
    dfd = defender_of(target, state)
    return sum(expected_attacks(p, dfd, state.rules).damage for p in profiles)


# ------------------------------------------------------------------ mêlée


def fight_targets(state: GameState, unit: Unit) -> List[Unit]:
    return [e for e in state.enemies_of(unit.side) if unit.models_in_engagement_range(e, state.rules)]


def build_melee_profiles(state: GameState, unit: Unit, target: Unit, fighters: Optional[List[Model]] = None) -> List[AttackProfile]:
    """Attaques de mêlée de ``unit`` sur ``target`` : les figurines qui l'engagent (ou ``fighters``,
    celles qui ont été affectées à cette cible)."""
    rules = state.rules
    engaged = unit.models_in_engagement_range(target, rules)
    fighters = engaged if fighters is None else [m for m in fighters if m in engaged]
    if not fighters:
        return []
    oath = state.oath_target == target.id and unit.has_ability("Oath of Moment")
    reroll_wounds = REROLL_NONE
    if unit.has_ability("Excessive Assault"):
        on_objective = any(state.unit_in_objective_range(target, o) for o in state.layout.objectives)
        reroll_wounds = REROLL_FAILS if on_objective else REROLL_ONES
    ap_bonus = 0
    if unit.charged and unit.has_keyword("Slaanesh") and any(
        f.has_ability("Excessive Vigour (Aura)") and (f.id == unit.id or unit.min_gap_to(f) <= 6.0) for f in state.units_of(unit.side)
    ):
        ap_bonus = 1  # Excessive Vigour : PA des armes de mêlée améliorée de 1 après une charge
    epic = effect_of(state, "epic_challenge", unit.id) if state.strat_used else None  # Epic Challenge : figurine choisie
    granted = granted_abilities(state, unit.id, "melee", target.id) if state.effects else []
    mods = _attack_mods(state, unit, target, "melee")
    groups: Dict[Tuple[str, bool], int] = defaultdict(int)
    weapons_by_name: Dict[str, Weapon] = {}
    for m in fighters:
        melee = list(m.melee_weapons)
        if not melee:
            continue
        # une seule arme de mêlée par figurine (la meilleure) ; Extra Attacks s'ajoute
        main = [w for w in melee if not w.has("extra attacks")]
        extra = [w for w in melee if w.has("extra attacks")]
        chosen = [_choose_profile(main, target)] if main else []
        chosen += extra
        for w in chosen:
            groups[(w.name, w.has("precision") or m.id == epic)] += 1
            weapons_by_name[w.name] = w
    profiles = []
    for (name, precision), count in groups.items():
        w = weapons_by_name[name]
        lance = w.has("lance") or "lance" in granted
        attacks = DiceExpr(w.attacks.n_dice, w.attacks.sides, w.attacks.flat + mods["attacks"]) if mods["attacks"] else w.attacks
        profiles.append(_profile(state, unit, target, w, mods, granted, count=count, attacks=attacks, skill=w.skill, hit_mod=0, min_hit=0,
                                 reroll_hits=REROLL_FAILS if oath else REROLL_NONE, reroll_wounds=reroll_wounds, scope="melee",
                                 wound_mod=1 if (lance and unit.charged) else 0, ap_bonus=ap_bonus, precision=precision,
                                 label=f"{w.name} ×{count}" + (" (Epic Challenge)" if precision and not w.has("precision") else "")))
    return profiles


def resolve_fight(state: GameState, unit: Unit, target: Unit, fighters: Optional[List[Model]] = None) -> CombatReport:
    profiles = build_melee_profiles(state, unit, target, fighters)
    report = _resolve_profiles(state, unit, target, profiles, "fight", attackers=fighters)
    unit.has_fought = True
    return report


def expected_outcome(state: GameState, unit: Unit, target: Unit, kind: str = "shooting") -> Tuple[float, float]:
    """(dégâts attendus, figurines tuées attendues) d'un tir ou d'un combat de ``unit`` sur ``target`` —
    aide à la décision affichée dans l'interface (l'excédent de dégâts d'une blessure est perdu)."""
    from .attack import estimate_models_slain

    if kind in ("shooting", "snap"):
        profiles, _ = build_shooting_profiles(state, unit, target, snap=kind == "snap")
    else:
        profiles = build_melee_profiles(state, unit, target)
    dfd = defender_of(target, state)
    guards = [m for m in target.alive_models if not m.is_leader] or target.alive_models
    w = guards[0].profile.wounds if guards else 1
    dmg = models = 0.0
    for p in profiles:
        e = expected_attacks(p, dfd, state.rules)
        dmg += e.damage
        models += estimate_models_slain(e.unsaved + e.devastating, p.damage, w, dfd.feel_no_pain)
    return dmg, min(models, float(target.strength))


def expected_fight(state: GameState, unit: Unit, target: Unit) -> float:
    profiles = build_melee_profiles(state, unit, target)
    dfd = defender_of(target, state)
    return sum(expected_attacks(p, dfd, state.rules).damage for p in profiles)
