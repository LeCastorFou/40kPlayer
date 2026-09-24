"""Actions libres : effets manuels et stratagèmes joués depuis le panneau.

Elles sont jouables par l'un ou l'autre joueur à tout moment (y compris pendant le tour adverse),
hors des décisions du moteur : c'est la « latitude » d'une vraie table, pour tout ce qu'une règle
permet et que le moteur ne sait pas encore jouer seul (réserves, soins, figurines qui reviennent,
blessures mortelles, mouvements spéciaux, bonus…). Chaque action est journalisée et reste dans
l'historique (l'adversaire la voit, n'importe quel joueur peut l'annuler).

* :class:`~fortyk.engine.actions.ManualAction` — effet appliqué tel quel (validation légère :
  sur la table, pas de chevauchement, cohérence) ;
* :class:`~fortyk.engine.actions.UseStratagemAction` — stratagème de la liste du joueur : CP et limites
  de 15.01 vérifiés, moment vérifié (tour et phase), effet appliqué si le compilateur de règles l'a
  traduit (:mod:`fortyk.rules_compiler`), sinon rappelé au joueur pour qu'il l'applique à la main.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..data.parse import parse_dice
from .actions import ManualAction, UseStratagemAction
from .effects import KINDS, UNTIL, add_effect, describe_effect
from .geometry import disk_gap
from .movement import check_model_positions, pose_disk
from .state import GameState, Unit, other_side
from .stratagems import key_of, record_use, unavailable

if TYPE_CHECKING:  # pragma: no cover
    from .engine import Engine

__all__ = ["free_error", "apply_free", "stratagem_of", "stratagem_timing_error", "target_ok", "heal_unit", "stratagem_name", "MANUAL_KINDS"]

MANUAL_KINDS = {
    "note": "règle invoquée (sans effet mécanique)",
    "cp": "points de commandement +/-",
    "heal": "soigner des PV",
    "revive": "ramener des figurines détruites",
    "mortal": "blessures mortelles",
    "destroy": "retirer des figurines",
    "reserve": "retour en réserve stratégique",
    "set_up": "poser l'unité (arrivée de réserve, redéploiement)",
    "move": "déplacer des figurines",
    "battleshock": "battle-shock oui / non",
    "effect": "effet à durée (bonus, capacité)",
}
_PHASE_NAME = {"command": "command", "movement": "movement", "shooting": "shooting", "charge": "charge", "fight": "fight"}


def _positions(action) -> Dict[str, tuple]:
    return {p[0]: tuple(float(v) for v in p[1:4]) for p in action.positions}


def _unit(s: GameState, uid: Optional[str]) -> Optional[Unit]:
    return s.units.get(uid) if uid else None


def _name(u: Optional[Unit]) -> str:
    return f"{u.name}" if u is not None else "?"


# ------------------------------------------------------------------ stratagèmes du panneau


def stratagem_of(s: GameState, side: str, sid: str):
    return next((st for st in s.stratagem_book.get(side, []) if st.id == sid), None)


def stratagem_name(s: GameState, side: str, sid: str) -> str:
    st = stratagem_of(s, side, sid)
    return st.name.title() if st is not None else sid


def stratagem_timing_error(s: GameState, side: str, st) -> Optional[str]:
    """Le stratagème se joue-t-il maintenant (tour et phase) ? Le moment précis (« juste après… »)
    reste à l'appréciation des joueurs, comme à la table."""
    if s.phase not in _PHASE_NAME:
        return "les stratagèmes se jouent pendant la bataille"
    turn = (st.turn or "").lower()
    mine = side == s.side_to_move
    if turn.startswith("your") and not mine:
        return "ce stratagème se joue pendant ton tour"
    if turn.startswith("opponent") and mine:
        return "ce stratagème se joue pendant le tour adverse"
    ph = (st.phase or "").lower()
    if ph and "any phase" not in ph and s.phase not in ph:
        fr = {"command": "commandement", "movement": "mouvement", "shooting": "tir", "charge": "charge", "fight": "combat"}
        when = " ou ".join(fr.get(p, p) for p in re.findall(r"command|movement|shooting|charge|fight", ph)) or ph
        return f"ce stratagème se joue en phase de {when}"
    return None


def _stratagem_error(engine: "Engine", s: GameState, side: str, a: UseStratagemAction) -> Optional[str]:
    from ..rules_compiler import compile_stratagem

    st = stratagem_of(s, side, a.stratagem_id)
    if st is None:
        return "stratagème inconnu pour ce camp"
    err = stratagem_timing_error(s, side, st)
    if err:
        return err
    unit = _unit(s, a.unit_id)
    if a.unit_id and unit is None:
        return f"unité inconnue : {a.unit_id}"
    comp = compile_stratagem(st)
    if unit is None and (comp.effects or any(i.on == "unit" for i in comp.instants)):
        return "choisis l'unité ciblée par le stratagème"
    if unit is not None and unit.side != side and re.search(r"from your army|friendly", st.target or "", re.I):
        return "ce stratagème cible une unité de ton armée"
    if unit is not None and unit.is_destroyed:
        return "cette unité est détruite"
    if unit is not None and unit.side == side and not target_ok(unit, st):
        return f"cette unité ne correspond pas à la cible : {st.target}"
    if comp.needs_enemy and (a.target_id is None or _unit(s, a.target_id) is None):
        return "choisis l'unité ennemie visée"
    if comp.choices and a.choice not in comp.choices:
        return "choisis l'option : " + " ou ".join(comp.choices)
    return unavailable(s, side, key_of(st.name), unit if (unit is not None and unit.side == side) else None, cost=st.cp or 0)


def _roll_expr(s: GameState, text) -> int:
    t = str(text).lower().strip()
    if t == "one":
        return 1
    if t.isdigit():
        return int(t)
    expr = parse_dice(t.upper())
    return expr.roll(s.rng) if expr is not None else 0


def _apply_stratagem(engine: "Engine", s: GameState, side: str, a: UseStratagemAction) -> str:
    from ..rules_compiler import compile_stratagem

    st = stratagem_of(s, side, a.stratagem_id)
    comp = compile_stratagem(st)
    unit, target = _unit(s, a.unit_id), _unit(s, a.target_id)
    cost = record_use(s, side, key_of(st.name), unit.id if (unit is not None and unit.side == side) else None, detail=st.name, cost=st.cp or 0)
    done: List[str] = []
    for e in comp.effects:
        value = a.choice if e.value == "$choice" else e.value
        if e.on == "target":
            if target is None:
                continue
            uid, vs = target.id, None
        else:
            if unit is None:
                continue
            uid, vs = unit.id, (target.id if (e.on == "vs_target" and target is not None) else None)
        eff = add_effect(s, e.kind, uid, value, source=st.name, side=side, until=e.until, scope=e.scope, vs=vs, cond=e.cond)
        done.append(describe_effect(eff))
    for i in comp.instants:
        victim = target if i.on == "target" else unit
        if victim is None:
            continue
        done.append(_instant(engine, s, side, i, victim, st.name))
    left = " ".join(comp.unparsed)
    who = f" sur {_name(unit)}" if unit is not None else ""
    vs = f" (contre {_name(target)})" if target is not None else ""
    text = f"Stratagème {st.name} ({cost} CP, reste {s.cp[side]}) — {side}{who}{vs}"
    if done:
        text += " : " + " ; ".join(done)
    if comp.status != "auto":
        text += f" — à appliquer à la main : {left or st.effect}"
    return text


def _instant(engine: "Engine", s: GameState, side: str, i, victim: Unit, source: str) -> str:
    if i.kind == "reserve":
        engine.to_reserves(s, victim, source, quiet=True)
        return f"{victim.name} retourne en réserve"
    if i.kind == "mortal":
        spec = i.value
        if spec[0] == "dice":
            _, n, need, per = spec
            rolls = [s.d6() for _ in range(n)]
            mw = sum(per for r in rolls if r >= need)
            detail = f"{n}D6 {rolls}"
        elif spec[0] == "check":
            _, need, amount = spec
            r = s.d6()
            mw = _roll_expr(s, amount) if r >= need else 0
            detail = f"D6 = {r} ({need}+)"
        else:
            mw = _roll_expr(s, spec[1])
            detail = str(spec[1]).upper()
        lost = engine.mortal_wounds(s, victim, mw)
        if victim.is_destroyed:
            if victim.side != s.side_to_move:
                s.destroyed_this_turn += 1
            engine.on_unit_destroyed(s, victim)
        return f"{detail} → {mw} BM à {victim.name} ({lost} fig.)" + (" — UNITÉ DÉTRUITE" if victim.is_destroyed else "")
    if i.kind == "heal":
        n = _roll_expr(s, i.value)
        healed = heal_unit(victim, n)
        return f"{victim.name} récupère {healed} PV"
    if i.kind == "battleshock_test":
        r = s.roll(2)
        failed = r < victim.leadership
        if failed:
            victim.battle_shocked = True
        return f"test de battle-shock de {victim.name} : {r} ({victim.leadership}+) — " + ("raté, battle-shocked" if failed else "réussi")
    if i.kind == "move_roll":
        n = _roll_expr(s, i.value)
        return f"{victim.name} peut se déplacer de {n}\" (déplacement à faire à la main)"
    if i.kind == "secure":
        got = []
        for o in s.layout.objectives:
            if s.objective_controller(o) == side and s.unit_in_objective_range(victim, o):
                s.sticky_control[o.id] = side
                got.append(o.id)
        return f"objectif(s) sécurisé(s) : {', '.join(got) or 'aucun'}"
    return i.kind


def target_ok(unit: Unit, st) -> bool:
    """La cible du stratagème (« One ADEPTUS ASTARTES INFANTRY or MOUNTED unit from your army ») convient-elle
    à ``unit`` ? Lecture tolérante : si le texte n'est pas compris, l'unité est acceptée."""
    m = re.search(r"\b(?:one|that|up to \w+|any number of)\s+(.+?)\s+(?:unit|units|model|models)\b", st.target or "", re.I)
    if not m:
        return True
    phrase = re.sub(r"\((?:[^)]*)\)", " ", m.group(1))
    phrase = re.sub(r"\b(friendly|unengaged|engaged|other|visible|non-\w+)\b", " ", phrase, flags=re.I)
    if re.search(r"\benemy\b", phrase, re.I):
        return True
    groups = [g.strip() for g in re.split(r"\bor\b|,|/", phrase, flags=re.I) if g.strip()]
    if not groups:
        return True
    for g in groups:
        tokens = g.upper().split()
        if not tokens:
            continue
        if unit.has_keyword(" ".join(tokens)) or all(unit.has_keyword(t) for t in tokens):
            return True
        for k in range(1, len(tokens)):  # préfixe de faction (« EMPEROR’S CHILDREN ») + mots-clés d'unité
            if unit.has_keyword(" ".join(tokens[:k])) and all(unit.has_keyword(t) for t in tokens[k:]):
                return True
    # « ADEPTUS ASTARTES INFANTRY or MOUNTED » : le préfixe de faction du premier groupe vaut pour les suivants
    first = groups[0].upper().split()
    for k in range(1, len(first)):
        if unit.has_keyword(" ".join(first[:k])):
            if any(all(unit.has_keyword(t) for t in g.upper().split()) for g in groups[1:]):
                return True
    return False


# ------------------------------------------------------------------ primitives


def heal_unit(unit: Unit, n: int, model_ids: Optional[List[str]] = None) -> int:
    """Soigne ``n`` PV (sur les figurines ``model_ids``, sinon la plus blessée) ; PV rendus."""
    total = 0
    targets = [m for m in unit.alive_models if (model_ids is None or m.id in model_ids)]
    if model_ids is None:
        targets.sort(key=lambda m: m.wounds - m.profile.wounds)
        targets = targets[:1]
    for m in targets:
        gain = min(n, m.profile.wounds - m.wounds)
        m.wounds += gain
        total += gain
    return total


def _placement_error(s: GameState, unit: Unit, positions: Dict[str, tuple], new_models: List = ()) -> Optional[str]:
    """Mise en place légère : sur la table, sans chevauchement, cohérence, hors portée d'engagement
    pour les figurines posées (03.02 « That unit is unengaged »)."""
    err = check_model_positions(s, unit, positions, "set_up", 0.0)
    if err:
        return err
    er = s.rules.engagement_range_in
    enemies = [m.disk for e in s.units_of(other_side(unit.side)) for m in e.alive_models]
    for m in new_models:
        d = pose_disk(m, positions[m.id])
        if any(disk_gap(d, e) <= er + 1e-9 for e in enemies):
            return f"{m.id} serait à portée d'engagement d'un ennemi"
    return None


# ------------------------------------------------------------------ effets manuels


def _manual_error(engine: "Engine", s: GameState, side: str, a: ManualAction) -> Optional[str]:
    if a.kind not in MANUAL_KINDS:
        return f"effet manuel inconnu : {a.kind}"
    if a.kind in ("note",):
        return None if (a.rule or a.note) else "indique la règle invoquée"
    if a.kind == "cp":
        if not a.value:
            return "indique le nombre de CP (+ ou -)"
        return None if s.cp.get(side, 0) + a.value >= 0 else "pas assez de CP"
    unit = _unit(s, a.unit_id)
    if unit is None:
        return "choisis une unité"
    if a.kind == "heal":
        if not a.value or a.value < 1:
            return "indique le nombre de PV"
        if unit.is_destroyed:
            return "unité détruite : utilise « ramener des figurines »"
        return None
    if a.kind == "revive":
        dead = [m for m in unit.models if not m.alive and m.id in a.model_ids]
        if not dead:
            return "choisis des figurines détruites de l'unité"
        if unit.in_reserve or unit.embarked_in:
            return None  # l'unité est hors table : les figurines reviennent avec elle
        pos = _positions(a)
        if any(m.id not in pos for m in dead):
            return "place chaque figurine ramenée sur la table"
        for m in dead:
            m.alive = True
        try:
            return _placement_error(s, unit, pos, dead)
        finally:
            for m in dead:
                m.alive = False
    if a.kind == "mortal":
        return None if (a.value and a.value > 0 and not unit.is_destroyed) else "indique le nombre de blessures mortelles"
    if a.kind == "destroy":
        return None if any(m.alive and m.id in a.model_ids for m in unit.models) else "choisis des figurines vivantes"
    if a.kind == "reserve":
        return None if (not unit.in_reserve and not unit.is_destroyed and unit.embarked_in is None) else "l'unité n'est pas sur la table"
    if a.kind == "set_up":
        if not (unit.in_reserve or unit.embarked_in):
            return "l'unité est déjà sur la table (utilise « déplacer »)"
        pos = _positions(a)
        if any(m.id not in pos for m in unit.alive_models):
            return "place toutes les figurines"
        return _placement_error(s, unit, pos, unit.alive_models)
    if a.kind == "move":
        if unit.in_reserve or unit.embarked_in or unit.is_destroyed:
            return "l'unité n'est pas sur la table"
        pos = _positions(a)
        if not pos:
            return "déplace au moins une figurine"
        return check_model_positions(s, unit, pos, "set_up", 0.0)
    if a.kind == "battleshock":
        return None if a.value in (0, 1) else "battle-shock : 1 (oui) ou 0 (non)"
    if a.kind == "effect":
        if a.effect not in KINDS:
            return f"effet inconnu : {a.effect}"
        if a.until not in UNTIL:
            return f"durée inconnue : {a.until}"
        return None
    return None


def _apply_manual(engine: "Engine", s: GameState, side: str, a: ManualAction) -> str:
    unit = _unit(s, a.unit_id)
    rule = f" ({a.rule})" if a.rule else ""
    note = f" — {a.note}" if a.note else ""
    head = f"Effet manuel — {side}{rule}"
    if a.kind == "note":
        return f"{head}{note or ''}"
    if a.kind == "cp":
        s.cp[side] = s.cp.get(side, 0) + int(a.value)
        return f"{head} : {a.value:+d} CP (reste {s.cp[side]}){note}"
    if a.kind == "heal":
        healed = heal_unit(unit, int(a.value), list(a.model_ids) or None)
        return f"{head} : {unit.name} récupère {healed} PV{note}"
    if a.kind == "revive":
        pos = _positions(a)
        back = []
        for m in unit.models:
            if m.alive or m.id not in a.model_ids:
                continue
            m.alive = True
            m.wounds = min(m.profile.wounds, int(a.value)) if a.value else m.profile.wounds
            if m.id in pos:
                p = pos[m.id]
                m.move_to(p[0], p[1], p[2] if len(p) > 2 and not m.is_round else None)
            back.append(m.id)
        return f"{head} : {len(back)} figurine(s) de {unit.name} reviennent ({', '.join(back)}){note}"
    if a.kind == "mortal":
        lost = engine.mortal_wounds(s, unit, int(a.value))
        text = f"{head} : {unit.name} subit {a.value} BM ({lost} fig.)" + (" — UNITÉ DÉTRUITE" if unit.is_destroyed else "") + note
        if unit.is_destroyed:
            if unit.side != s.side_to_move:
                s.destroyed_this_turn += 1
            engine.on_unit_destroyed(s, unit)
        return text
    if a.kind == "destroy":
        gone = [m.id for m in unit.models if m.alive and m.id in a.model_ids]
        for m in unit.models:
            if m.id in gone:
                m.alive, m.wounds = False, 0
        text = f"{head} : {', '.join(gone)} retirée(s) de {unit.name}" + (" — UNITÉ DÉTRUITE" if unit.is_destroyed else "") + note
        if unit.is_destroyed:
            if unit.side != s.side_to_move:
                s.destroyed_this_turn += 1
            engine.on_unit_destroyed(s, unit)
        return text
    if a.kind == "reserve":
        engine.to_reserves(s, unit, a.rule or "effet manuel", quiet=True)
        return f"{head} : {unit.name} en réserve stratégique{note}"
    if a.kind == "set_up":
        engine.set_up_unit(s, unit, _positions(a))
        unit.embarked_in = None
        unit.arrived = True
        c = unit.centroid
        return f"{head} : {unit.name} posée en ({c[0]:.1f}, {c[1]:.1f}){note}"
    if a.kind == "move":
        pos = _positions(a)
        longest = 0.0
        for m in unit.alive_models:
            if m.id in pos:
                p = pos[m.id]
                longest = max(longest, ((p[0] - m.x) ** 2 + (p[1] - m.y) ** 2) ** 0.5)
                m.move_to(p[0], p[1], p[2] if len(p) > 2 and not m.is_round else None)
        return f"{head} : {unit.name} se déplace (jusqu'à {longest:.1f}\"){note}"
    if a.kind == "battleshock":
        unit.battle_shocked = bool(a.value)
        return f"{head} : {unit.name} " + ("est battle-shocked" if a.value else "n'est plus battle-shocked") + note
    if a.kind == "effect":
        value = a.effect_value
        if isinstance(value, str) and value.lstrip("-").isdigit():
            value = int(value)
        eff = add_effect(s, a.effect, unit.id, value, source=a.rule or "effet manuel", side=side, until=a.until, scope=a.scope, vs=a.vs)
        return f"{head} : {unit.name} — {describe_effect(eff)}{note}"
    return head


# ------------------------------------------------------------------ API


def free_error(engine: "Engine", s: GameState, side: str, action) -> Optional[str]:
    if isinstance(action, UseStratagemAction):
        return _stratagem_error(engine, s, side, action)
    if isinstance(action, ManualAction):
        return _manual_error(engine, s, side, action)
    return "action libre inconnue"


def apply_free(engine: "Engine", s: GameState, side: str, action) -> str:
    if isinstance(action, UseStratagemAction):
        return _apply_stratagem(engine, s, side, action)
    return _apply_manual(engine, s, side, action)
