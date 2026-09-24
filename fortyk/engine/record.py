"""Enregistrement des parties : actions ⇄ JSON, sauvegarde et relecture.

Une partie est entièrement déterminée par son état initial (graine des dés, rosters, carte) et la
liste de ses actions (``GameState.history``) : on la sauvegarde en JSON et on la rejoue à l'identique
avec :meth:`Engine.replay`. C'est la base des jeux de données d'entraînement et de l'analyse des
parties du bot.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import actions as A
from .movement import FormationMove

__all__ = ["action_to_json", "action_from_json", "history_to_json", "history_from_json", "save_game", "load_game"]

_ACTION_TYPES = {cls.__name__: cls for cls in (
    A.DeployAction, A.DeployModelsAction, A.SelectUnitAction, A.EndPhaseAction, A.ContinueAction, A.OathAction,
    A.MoveAction, A.ModelMoveAction, A.DeclareAdvanceAction, A.ShootAction, A.ChargeAction, A.DeclareChargeAction,
    A.AutoChargeMoveAction, A.FightAction, A.EmbarkAction, A.DisembarkAction, A.DisembarkModelsAction, A.StratagemAction, A.ReserveAction, A.ManualAction,
    A.UseStratagemAction,
)}


def action_to_json(action) -> Dict[str, Any]:
    data: Dict[str, Any] = {"type": type(action).__name__}
    for f in dataclasses.fields(action):
        v = getattr(action, f.name)
        if isinstance(v, FormationMove):
            v = {"kind": v.kind, "dx": v.dx, "dy": v.dy, "label": v.label}
        elif isinstance(v, tuple):
            v = [list(x) if isinstance(x, tuple) else x for x in v]
        data[f.name] = v
    return data


def action_from_json(data: Dict[str, Any]):
    cls = _ACTION_TYPES[data["type"]]
    kwargs = {}
    for f in dataclasses.fields(cls):
        v = data.get(f.name)
        if f.name == "move" and isinstance(v, dict):
            v = FormationMove(v["kind"], v["dx"], v["dy"], v.get("label", ""))
        elif f.name == "positions" and v is not None:
            v = tuple(tuple(x) for x in v)
        elif isinstance(v, list):
            v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        kwargs[f.name] = v
    return cls(**kwargs)


def history_to_json(history) -> List[Dict[str, Any]]:
    return [{"side": side, "action": action_to_json(a)} for side, a in history]


def history_from_json(items) -> List[Tuple[str, Any]]:
    return [(it["side"], action_from_json(it["action"])) for it in items]


def save_game(path, state, meta: Dict[str, Any]) -> None:
    """Sauvegarde ``meta`` (graine, rosters, carte, options de départ…) et l'historique de ``state``."""
    doc = {"format": "fortyk-game/1", "meta": meta, "history": history_to_json(state.history),
           "scores": state.scoreboard.totals()}
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def load_game(path) -> Dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    doc["history"] = history_from_json(doc["history"])
    return doc
