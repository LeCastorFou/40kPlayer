"""Effets de règles à durée limitée : la brique commune des stratagèmes, capacités et effets manuels.

Un :class:`Effect` modifie une unité jusqu'à la fin de la phase, du tour, du round ou de la bataille
(« Until the end of the phase, each time a model in your unit makes an attack, add 1 to the Hit
roll »). Le combat, le mouvement et le contrôle des objectifs lisent les effets actifs au moment où
ils en ont besoin : il suffit d'ajouter un effet pour qu'une règle soit jouée, qu'elle vienne d'un
stratagème de base, d'un stratagème de détachement traduit depuis Wahapedia (voir
:mod:`fortyk.rules_compiler`) ou d'un effet manuel posé par un joueur.

Genres d'effets (``kind``) — côté attaquant (``scope`` : all, ranged ou melee ; ``vs`` : seulement
contre cette unité ennemie) :

* ``hit_mod``, ``wound_mod`` : +/-N au jet de touche / de blessure ;
* ``ap_mod`` : améliore la PA de N ; ``damage_mod``, ``strength_mod``, ``attacks_mod`` : +N ;
* ``reroll_hits``, ``reroll_wounds`` : "ones" ou "fails" ;
* ``weapon_ability`` : mot-clé d'arme gagné (« lethal hits », « sustained hits 1 », « devastating
  wounds », « precision », « ignores cover », « lance », « assault », « heavy », « twin-linked »,
  « anti-infantry 4+ »…) ;
* ``crit_hit_on`` : touche critique sur N+ ; ``skill_mod`` : CT / CC améliorée de N.

Côté défenseur (attaques qui ciblent l'unité) :

* ``hit_mod_against``, ``wound_mod_against`` : +/-N aux jets de l'attaquant (Stealth = -1 au tir) ;
* ``ap_worsen`` : dégrade la PA de N ; ``damage_reduction`` : -N aux dégâts (minimum 1) ;
* ``fnp`` : Feel No Pain N+ ; ``invuln`` : sauvegarde invulnérable N+ ; ``cover`` : a le couvert ;
* ``save_mod`` : +N à la sauvegarde.

Mouvement et divers : ``move_mod`` (+N"), ``advance_mod`` / ``charge_mod`` (+N au jet),
``advance_and_charge``, ``advance_and_shoot``, ``fall_back_and_shoot``, ``fall_back_and_charge``,
``fights_first``, ``oc_mod``, ``battleshock_pass`` (réussit ses tests), ``lone_operative``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

__all__ = ["Effect", "KINDS", "UNTIL", "add_effect", "active_effects", "effect_total", "has_effect", "best_threshold",
           "reroll_policy", "granted_abilities", "prune_effects", "describe_effect"]

#: genre d'effet → libellé (interface, journal)
KINDS: Dict[str, str] = {
    "hit_mod": "au jet de touche",
    "wound_mod": "au jet de blessure",
    "ap_mod": "PA améliorée",
    "damage_mod": "aux dégâts",
    "strength_mod": "à la Force",
    "attacks_mod": "aux Attaques",
    "reroll_hits": "relance des touches",
    "reroll_wounds": "relance des blessures",
    "weapon_ability": "mot-clé d'arme",
    "crit_hit_on": "touche critique sur",
    "skill_mod": "CT / CC améliorée",
    "hit_mod_against": "au jet de touche adverse",
    "wound_mod_against": "au jet de blessure adverse",
    "ap_worsen": "PA adverse dégradée",
    "damage_reduction": "dégâts subis réduits",
    "fnp": "Feel No Pain",
    "fnp_mortal": "Feel No Pain contre les blessures mortelles",
    "toughness_mod": "à l'Endurance",
    "invuln": "sauvegarde invulnérable",
    "cover": "couvert",
    "save_mod": "à la sauvegarde",
    "move_mod": "au mouvement",
    "advance_mod": "au jet d'Advance",
    "charge_mod": "au jet de charge",
    "advance_and_charge": "peut charger après une Advance",
    "advance_and_shoot": "peut tirer après une Advance",
    "fall_back_and_shoot": "peut tirer après un repli",
    "fall_back_and_charge": "peut charger après un repli",
    "fights_first": "Fights First",
    "oc_mod": "à l'OC",
    "battleshock_pass": "réussit ses tests de battle-shock",
    "lone_operative": "Lone Operative",
    "deep_strike": "arrive comme Deep Strike",
    "advance_fixed": "Advance sans jet (distance fixe)",
}

#: conditions reconnues (``Effect.cond``) : kw:A|B (l'autre unité a un de ces mots-clés), battleshocked,
#: below_half, objective (l'autre unité est à portée d'un objectif), self:stationary, self:charged,
#: s_gt_t (la Force de l'attaque dépasse l'Endurance de l'unité — évaluée arme par arme)
CONDITIONS = ("kw:", "battleshocked", "below_half", "objective", "self:stationary", "self:charged", "s_gt_t")

#: durées possibles
UNTIL = ("phase", "turn", "round", "battle", "owner_next_turn")


@dataclass(frozen=True)
class Effect:
    kind: str
    unit_id: str
    value: object = None
    source: str = ""  #: règle, stratagème ou « effet manuel »
    side: str = ""  #: camp qui a posé l'effet
    until: str = "phase"
    turn: int = 0  #: tour de joueur (GameState.turn_counter) de création
    phase: str = ""
    round: int = 0
    scope: str = "all"  #: all | ranged | melee
    vs: Optional[str] = None  #: unité ennemie concernée (« contre cette unité »)
    cond: Optional[str] = None  #: condition sur l'autre unité ou sur l'unité elle-même (voir CONDITIONS)


def add_effect(state, kind: str, unit_id: str, value=None, source: str = "", side: str = "", until: str = "phase",
               scope: str = "all", vs: Optional[str] = None, cond: Optional[str] = None) -> Effect:
    if kind not in KINDS:
        raise ValueError(f"effet inconnu : {kind}")
    if until not in UNTIL:
        raise ValueError(f"durée inconnue : {until}")
    e = Effect(kind, unit_id, value, source, side, until, state.turn_counter, state.phase, state.battle_round, scope, vs, cond)
    state.effects.append(e)
    return e


def _alive(state, e: Effect) -> bool:
    if e.until == "battle":
        return True
    if e.until == "phase":
        return e.turn == state.turn_counter and e.phase == state.phase
    if e.until == "turn":
        return e.turn == state.turn_counter
    if e.until == "round":
        return e.round == state.battle_round
    # owner_next_turn : jusqu'au début du prochain tour du camp qui a posé l'effet
    if state.turn_counter == e.turn:
        return True
    return state.turn_counter == e.turn + 1 and state.side_to_move != e.side


def prune_effects(state) -> None:
    """Retire les effets expirés (appelé aux changements de phase ; la lecture filtre de toute façon)."""
    if state.effects:
        state.effects = [e for e in state.effects if _alive(state, e)]


def _cond_ok(state, e: Effect, vs: Optional[str], strength: Optional[int]) -> bool:
    c = e.cond
    if not c:
        return True
    if c == "s_gt_t":
        if strength is None:
            return False
        from .combat import unit_toughness  # import local : combat importe ce module

        return strength > unit_toughness(state.units[e.unit_id])
    if c.startswith("self:"):
        me = state.units.get(e.unit_id)
        return me is not None and bool(getattr(me, {"self:stationary": "remained_stationary", "self:charged": "charged"}.get(c, "_"), False))
    other = state.units.get(vs) if vs else None
    if other is None:
        return False
    if c.startswith("kw:"):
        return any(other.has_keyword(k) for k in c[3:].split("|"))
    if c == "battleshocked":
        return other.battle_shocked
    if c == "below_half":
        return other.below_half_strength
    if c == "objective":
        return any(state.unit_in_objective_range(other, o) for o in state.layout.objectives)
    return False


def active_effects(state, unit_id: str, kinds: Optional[Iterable[str]] = None, scope: Optional[str] = None,
                   vs: Optional[str] = None, strength: Optional[int] = None) -> List[Effect]:
    """Effets actifs sur ``unit_id`` ; ``scope`` : ranged / melee (garde les effets « all ») ;
    ``vs`` : unité adverse concernée (garde les effets sans restriction ; sert aux conditions) ;
    ``strength`` : Force de l'attaque (condition s_gt_t)."""
    if not state.effects:
        return []
    ks = set(kinds) if kinds is not None else None
    out = []
    for e in state.effects:
        if e.unit_id != unit_id or (ks is not None and e.kind not in ks) or not _alive(state, e):
            continue
        if scope is not None and e.scope not in ("all", scope):
            continue
        if e.vs is not None and vs is not None and e.vs != vs:
            continue
        if e.cond and not _cond_ok(state, e, vs, strength):
            continue
        out.append(e)
    return out


def effect_total(state, unit_id: str, kind: str, scope: Optional[str] = None, vs: Optional[str] = None, strength: Optional[int] = None) -> int:
    return sum(int(e.value or 0) for e in active_effects(state, unit_id, (kind,), scope, vs, strength))


def has_effect(state, unit_id: str, kind: str, scope: Optional[str] = None, vs: Optional[str] = None) -> bool:
    return bool(active_effects(state, unit_id, (kind,), scope, vs))


def best_threshold(state, unit_id: str, kind: str, current: Optional[int], scope: Optional[str] = None, vs: Optional[str] = None) -> Optional[int]:
    """Meilleur seuil (le plus bas) entre ``current`` et les effets « N+ » (Feel No Pain, invulnérable,
    touche critique)."""
    vals = [int(e.value) for e in active_effects(state, unit_id, (kind,), scope, vs) if e.value is not None]
    if current is not None:
        vals.append(int(current))
    return min(vals) if vals else None


_ORDER = {"none": 0, "ones": 1, "fails": 2}


def reroll_policy(current: str, state, unit_id: str, kind: str, scope: Optional[str] = None, vs: Optional[str] = None) -> str:
    """La relance la plus généreuse entre ``current`` et les effets (« fails » > « ones » > « none »)."""
    best = current
    for e in active_effects(state, unit_id, (kind,), scope, vs):
        if _ORDER.get(str(e.value), 0) > _ORDER.get(best, 0):
            best = str(e.value)
    return best


def granted_abilities(state, unit_id: str, scope: Optional[str] = None, vs: Optional[str] = None) -> List[str]:
    """Mots-clés d'arme gagnés (en minuscules : « lethal hits », « sustained hits 1 »…)."""
    return [str(e.value).lower() for e in active_effects(state, unit_id, ("weapon_ability",), scope, vs)]


def describe_effect(e: Effect) -> str:
    label = KINDS.get(e.kind, e.kind)
    scope = {"ranged": " (tir)", "melee": " (mêlée)"}.get(e.scope, "")
    until = {"phase": "fin de la phase", "turn": "fin du tour", "round": "fin du round", "battle": "fin de la bataille",
             "owner_next_turn": "ton prochain tour"}[e.until]
    if e.kind in ("fnp", "fnp_mortal", "invuln", "crit_hit_on"):
        text = f"{label} {e.value}+"
    elif e.kind in ("reroll_hits", "reroll_wounds"):
        text = f"{label} ({'les 1' if e.value == 'ones' else 'les échecs'})"
    elif e.kind == "weapon_ability":
        text = f"[{str(e.value).upper()}]"
    elif isinstance(e.value, int):
        text = f"{e.value:+d} {label}" if e.kind not in ("ap_mod", "ap_worsen", "damage_reduction", "skill_mod") else f"{label} de {e.value}"
    else:
        text = label
    return f"{text}{scope}{' contre ' + e.vs if e.vs else ''} jusqu'à la {until}" if e.until != "owner_next_turn" else f"{text}{scope} jusqu'à {until}"
