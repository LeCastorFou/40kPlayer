"""Sérialisation JSON de l'état de partie et des décisions pour l'interface navigateur."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..engine.actions import (
    AutoChargeMoveAction,
    ChargeAction,
    Decision,
    DeclareAdvanceAction,
    DeclareChargeAction,
    DeployAction,
    DeployModelsAction,
    DisembarkAction,
    DisembarkModelsAction,
    EmbarkAction,
    EndPhaseAction,
    FightAction,
    ModelMoveAction,
    MoveAction,
    OathAction,
    SelectUnitAction,
    ShootAction,
)
from ..engine.combat import is_hidden, models_outside_terrain
from ..engine.layout import Layout
from ..engine.movement import MoveKind
from ..engine.state import GameState, Unit

__all__ = ["layout_to_json", "state_to_json", "decision_to_json", "action_label", "factions_of"]


def layout_to_json(layout: Layout, rules) -> Dict[str, Any]:
    return {
        "name": layout.name,
        "board": list(layout.board),
        "deployment_zones": {side: [list(v) for v in poly.vertices] for side, poly in layout.deployment_zones.items()},
        "territory_line": [list(p) for p in layout.territory_line] if layout.territory_line else None,
        "objectives": [
            {"id": o.id, "kind": o.kind, "owner": o.owner, "x": o.x, "y": o.y,
             "terrain_id": o.terrain_id, "rect": list(o.rect) if o.rect is not None else None}
            for o in layout.objectives
        ],
        "terrain": [
            {"id": t.id, "kind": t.kind, "density": t.density, "rect": list(t.rect), "label": t.label}
            for t in layout.terrain
        ],
        "rules": {
            "engagement_range": rules.engagement_range_in,
            "objective_control_radius": rules.objective_control_radius_in,
            "objective_marker_radius": rules.objective_marker_radius_in,
            "coherency": rules.coherency_range_in,
            "charge_range": rules.charge_declare_range_in,
        },
    }


def _unit_to_json(state: GameState, u: Unit) -> Dict[str, Any]:
    deployed = (u.embarked_in is None and all(state.on_board(m.disk) for m in u.alive_models)) if u.alive_models else False
    weapons = sorted({w.name for m in u.alive_models for w in m.weapons})
    return {
        "id": u.id,
        "side": u.side,
        "name": u.datasheet.name + "".join(f" + {l.name}" for l in u.leaders),
        "strength": u.strength,
        "starting_strength": u.starting_strength,
        "destroyed": u.is_destroyed,
        "deployed": deployed,
        "embarked_in": u.embarked_in,
        "passengers": [p.id for p in state.passengers(u.id)],
        "move": u.move_in,
        "profile": {
            "T": u.profile.toughness,
            "Sv": u.profile.save,
            "W": u.profile.wounds,
            "Ld": u.profile.leadership,
            "OC": u.profile.oc,
        },
        "flags": {
            "advanced": u.advanced,
            "fell_back": u.fell_back,
            "charged": u.charged,
            "battle_shocked": u.battle_shocked,
            "engaged": (not u.is_destroyed) and deployed and state.battle_round >= 1 and state.is_engaged(u),
            "has_shot": u.has_shot,
            "has_fought": u.has_fought,
            "hidden": (not u.is_destroyed) and deployed and is_hidden(state, u),
        },
        "weapons": weapons,
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "x": round(m.x, 3),
                "y": round(m.y, 3),
                "r": round(m.radius, 4),
                "hx": round(m.hx, 4),
                "hy": round(m.hy, 4),
                "a": round(m.angle, 5),
                "wounds": m.wounds,
                "max_wounds": m.profile.wounds,
                "alive": m.alive,
            }
            for m in u.models
        ],
    }


def decision_to_json(state: GameState, decision: Optional[Decision]) -> Optional[Dict[str, Any]]:
    if decision is None:
        return None
    options: List[Dict[str, Any]] = []
    for i, opt in enumerate(decision.options):
        item: Dict[str, Any] = {"index": i, "label": str(opt)}
        if isinstance(opt, ChargeAction) and opt.extra_targets:
            item["target_id"] = opt.target_id
            item["targets"] = list(opt.targets)
            item["label"] = " + ".join(f"{state.unit(t).datasheet.name} [{t}]" for t in opt.targets)
            item["distance"] = round(max(state.unit(opt.unit_id).min_gap_to(state.unit(t)) for t in opt.targets), 1)
        elif isinstance(opt, (ShootAction, ChargeAction)):
            item["target_id"] = opt.target_id
            item["label"] = "ne rien faire" if opt.target_id is None else state.unit(opt.target_id).datasheet.name
            if opt.target_id is not None:
                tgt = state.unit(opt.target_id)
                item["distance"] = round(state.unit(opt.unit_id).min_gap_to(tgt), 1)
                item["hidden"] = is_hidden(state, tgt)
                item["outside_terrain"] = models_outside_terrain(state, tgt)
        elif isinstance(opt, DeclareChargeAction):
            item["declare_charge"] = True
            item["label"] = "Déclarer la charge (2D6)"
        elif isinstance(opt, AutoChargeMoveAction):
            item["auto_charge"] = True
            item["target_id"] = opt.target_id
            item["label"] = "Placement automatique"

        elif isinstance(opt, FightAction):
            item["unit_id"] = opt.unit_id
            item["target_id"] = opt.target_id
            item["label"] = f"{state.unit(opt.unit_id).datasheet.name} [{opt.unit_id}] frappe {state.unit(opt.target_id).datasheet.name} [{opt.target_id}]"
        elif isinstance(opt, OathAction):
            item["target_id"] = opt.target_id
            item["label"] = state.unit(opt.target_id).datasheet.name + f" [{opt.target_id}]" if opt.target_id else "aucun"
        elif isinstance(opt, MoveAction):
            mv = opt.move
            item["move_kind"] = mv.kind
            item["dx"], item["dy"] = round(mv.dx, 2), round(mv.dy, 2)
            item["label"] = mv.label or mv.kind
        elif isinstance(opt, DeployAction):
            item["x"], item["y"] = opt.x, opt.y
        elif isinstance(opt, SelectUnitAction):
            u = state.unit(opt.unit_id)
            item["unit_id"] = opt.unit_id
            item["label"] = f"{u.datasheet.name} [{u.id}] — " + (f"{u.models[0].wounds}/{u.models[0].profile.wounds} PV" if u.starting_strength == 1 else f"{u.strength}/{u.starting_strength} fig.")
            if u.embarked_in is not None:
                item["embarked_in"] = u.embarked_in
                item["label"] += f" — à bord de {u.embarked_in} : débarquer ?"
        elif isinstance(opt, EmbarkAction):
            item["unit_id"] = opt.unit_id
            item["transport_id"] = opt.transport_id
            item["embark"] = True
            item["label"] = f"Embarquer dans {state.unit(opt.transport_id).datasheet.name} [{opt.transport_id}]" if opt.transport_id else "Ne pas embarquer"
        elif isinstance(opt, DisembarkModelsAction):
            item["unit_id"] = opt.unit_id
            item["disembark"] = True
            item["positions"] = {p[0]: [round(p[1], 3), round(p[2], 3)] for p in opt.positions}
            item["label"] = "Débarquer en couronne autour du transport"
        elif isinstance(opt, DisembarkAction):
            item["unit_id"] = opt.unit_id
            item["disembark"] = True
            if opt.x is None:
                item["stay"] = True
                item["label"] = "Rester à bord"
            else:
                item["x"], item["y"] = opt.x, opt.y
                item["label"] = f"Débarquer en ({opt.x:.1f}, {opt.y:.1f})"
        elif isinstance(opt, EndPhaseAction):
            item["end_phase"] = True
            item["label"] = "Terminer la phase"
        options.append(item)
    ineligible = {uid: {"name": state.unit(uid).datasheet.name, "reason": why} for uid, why in decision.ineligible.items()}
    return {
        "kind": decision.kind,
        "side": decision.side,
        "unit_id": decision.unit_id,
        "note": decision.note,
        "phase": decision.phase,
        "max_distance": decision.max_distance,
        "target_id": decision.target_id,
        "targets": list(decision.targets),
        "options": options,
        "ineligible": ineligible,
    }


_MOVE_KIND_FR = {"normal": "déplace", "advance_move": "Advance", "advance": "Advance", "fall_back": "se replie", "desperate": "Desperate Escape",
                 "scout": "mouvement de scout", "charge": "charge (placement à la main)"}


def action_label(state: GameState, decision: Decision, action) -> str:
    """Libellé court d'une action pour l'historique de la partie (calculé avant de l'appliquer)."""

    def name(uid):
        if uid is None:
            return "?"
        try:
            u = state.unit(uid)
        except KeyError:
            return uid
        return f"{u.datasheet.name} [{uid}]"

    uid = getattr(action, "unit_id", None) or decision.unit_id
    if isinstance(action, SelectUnitAction):
        return f"active {name(action.unit_id)}"
    if isinstance(action, EndPhaseAction):
        return "termine la phase"
    if isinstance(action, (DeployAction, DeployModelsAction)):
        return f"déploie {name(uid)}"
    if isinstance(action, EmbarkAction):
        return f"{name(uid)} embarque dans {name(action.transport_id)}" if action.transport_id else f"{name(uid)} n'embarque pas"
    if isinstance(action, (DisembarkAction, DisembarkModelsAction)):
        if isinstance(action, DisembarkAction) and action.x is None:
            return f"{name(uid)} reste à bord"
        return f"{name(uid)} débarque"
    if isinstance(action, MoveAction):
        return f"{name(uid)} : {action.move.label or action.move.kind}"
    if isinstance(action, ModelMoveAction):
        return f"{name(uid)} : {_MOVE_KIND_FR.get(action.kind, action.kind)}"
    if isinstance(action, DeclareAdvanceAction):
        return f"{name(uid)} déclare une Advance (D6)"
    if isinstance(action, ShootAction):
        return f"{name(uid)} tire sur {name(action.target_id)}" if action.target_id else f"{name(uid)} ne tire pas"
    if isinstance(action, DeclareChargeAction):
        return f"{name(uid)} déclare une charge (2D6)"
    if isinstance(action, ChargeAction):
        if action.target_id is None:
            return f"{name(uid)} ne charge pas"
        return f"{name(uid)} charge " + " + ".join(name(t) for t in action.targets)
    if isinstance(action, AutoChargeMoveAction):
        return f"{name(uid)} : charge, placement automatique"
    if isinstance(action, FightAction):
        return f"{name(action.unit_id)} frappe {name(action.target_id)}"
    if isinstance(action, OathAction):
        return f"Oath of Moment sur {name(action.target_id)}" if action.target_id else "pas d'Oath of Moment"
    return str(action)


def factions_of(state: GameState) -> Dict[str, str]:
    return _factions(state)


def _factions(state: GameState) -> Dict[str, str]:
    """Nom de faction affiché par camp : celui de la liste importée, sinon celui des fiches."""
    out = {}
    for side in ("attacker", "defender"):
        al = (state.army_lists or {}).get(side)
        if al is not None:
            out[side] = al.faction_name
            continue
        names = sorted({u.datasheet.faction_name for u in state.units.values() if u.side == side})
        out[side] = " / ".join(names)
    return out


def state_to_json(state: GameState, decision: Optional[Decision], human_side: Optional[str], since: int = 0, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    objectives = []
    for o in state.layout.objectives:
        levels = state.levels_of_control(o)
        objectives.append({"id": o.id, "controller": state.objective_controller(o), "levels": levels, "sticky": state.sticky_control.get(o.id)})
    data = {
        "battle_round": state.battle_round,
        "phase": state.phase,
        "side_to_move": state.side_to_move,
        "first_player": state.first_player,
        "human_side": human_side,
        "scores": state.scoreboard.totals(),
        "cp": dict(state.cp),
        "round_scores": {side: state.scoreboard.round_total(side, state.battle_round) for side in ("attacker", "defender")},
        "oath_target": state.oath_target,
        "units": [_unit_to_json(state, u) for u in state.units.values()],
        "factions": _factions(state),
        "objectives": objectives,
        "log": state.log[since:],
        "log_length": len(state.log),
        "pending": decision_to_json(state, decision),
    }
    if extra:
        data.update(extra)
    return data
