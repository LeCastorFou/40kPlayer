"""Parties sauvegardées → données d'entraînement.

Une partie jouée sur le service web est enregistrée au format ``fortyk-game/2`` (voir
:mod:`fortyk.web.rooms`) : configuration (graine, listes en texte), décisions dans l'ordre, re-tirages
des dés après les annulations, branches annulées. Ce module la rejoue et en tire un export
``fortyk-training/1`` : pour chaque décision, l'état compact du plateau **avant** la décision, la
décision posée (genre, camp, nombre d'options) et l'action choisie, puis les événements structurés du
moteur et le résultat. C'est la matière première de l'entraînement (imitation des parties humaines,
évaluation de positions).

    python3 scripts/export_games.py /data/games parties.jsonl     # une partie par ligne
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .data import Catalog
from .engine import Engine
from .engine.actions import Decision
from .engine.record import action_from_json
from .engine.state import SIDES, GameState

__all__ = ["TRAINING_FORMAT", "build_initial_state", "iter_replay", "replay_record", "observation", "training_export"]

TRAINING_FORMAT = "fortyk-training/1"


def build_initial_state(cat: Catalog, config: Dict[str, Any]) -> GameState:
    """État initial d'une partie depuis sa ``config`` (listes en texte ; pas de liste = toy model)."""
    from .data.list_library import resolve_text
    from .toy import new_state_from_lists

    lists = config.get("lists") or {}
    resolved = []
    for side in SIDES:
        text = (lists.get(side) or {}).get("text")
        resolved.append(resolve_text(text, cat) if text else None)
    state = new_state_from_lists(resolved[0], resolved[1], cat, layout=config.get("layout", "layout_a"), seed=int(config["seed"]))
    # règles du moteur à la création de la partie : une partie enregistrée avant l'arrivée des
    # stratagèmes (sans ces clés) se rejoue avec les règles d'alors
    state.rev = int(config.get("rev", 1))
    state.stratagems = bool(config.get("stratagems", False))
    return state


def iter_replay(engine: Engine, initial: GameState, record: Dict[str, Any]) -> Iterator[Tuple[int, GameState, Decision, Dict[str, Any]]]:
    """Rejoue ``record["history"]`` sur une copie de ``initial`` en réappliquant les re-tirages des dés.
    Produit (i, état avant la décision i, décision, entrée d'historique) puis applique l'action ; à la
    fin, l'état final est dans ``StopIteration.value`` (utiliser :func:`replay_record`)."""
    reseeds = {int(r["at"]): int(r["seed"]) for r in record.get("reseeds", [])}
    s = initial.clone()
    engine.start(s, deploy=record["config"].get("deploy", True))
    history = record["history"]
    for i, h in enumerate(history):
        if i in reseeds:
            s.rng.seed(reseeds[i])
        if h.get("decision") == "free":  # action libre (effet manuel, stratagème du panneau), de l'un ou l'autre camp
            act = action_from_json(h["action"])
            yield i, s, Decision("free", h["side"], [act]), h
            engine.apply_free(s, h["side"], act, validate=False)
            continue
        d = engine.decision(s)
        if d is None or d.side != h["side"]:
            raise ValueError(f"historique désynchronisé à l'action {i} ({h.get('label')})")
        yield i, s, d, h
        engine.step(s, action_from_json(h["action"]), validate=False)
    if len(history) in reseeds:
        s.rng.seed(reseeds[len(history)])
    return s


def replay_record(engine: Engine, initial: GameState, record: Dict[str, Any], on_decision: Optional[Callable] = None) -> GameState:
    """Rejoue toute la partie ; ``on_decision(i, state, decision, entry)`` avant chaque action."""
    it = iter_replay(engine, initial, record)
    while True:
        try:
            item = next(it)
        except StopIteration as stop:
            return stop.value
        if on_decision is not None:
            on_decision(*item)


def observation(state: GameState) -> Dict[str, Any]:
    """État compact du plateau (ce qu'un joueur voit), pour l'entraînement."""
    units = []
    for u in state.units.values():
        units.append({
            "id": u.id,
            "side": u.side,
            "datasheet": u.datasheet.name,
            "leaders": [l.name for l in u.leaders],
            "embarked_in": u.embarked_in,
            "battle_shocked": u.battle_shocked,
            "flags": [k for k in ("advanced", "fell_back", "charged", "has_shot", "has_fought") if getattr(u, k)],
            # figurines vivantes : [id, x, y, angle, PV]
            "models": [[m.id, round(m.x, 3), round(m.y, 3), round(m.angle, 4), m.wounds] for m in u.models if m.alive],
            "starting_strength": u.starting_strength,
        })
    return {
        "round": state.battle_round,
        "phase": state.phase,
        "side_to_move": state.side_to_move,
        "first_player": state.first_player,
        "scores": state.scoreboard.totals(),
        "cp": dict(state.cp),
        # stratagèmes utilisés jusqu'ici : [camp, clé, unité, tour, phase]
        "stratagems_used": [list(e[:5]) for e in state.strat_used],
        "objectives": {o.id: state.objective_controller(o) for o in state.layout.objectives},
        "units": units,
    }


def training_export(cat: Catalog, record: Dict[str, Any], states: bool = True) -> Dict[str, Any]:
    """Export d'entraînement d'une partie enregistrée (``states=False`` : sans les états, plus léger)."""
    engine = Engine()
    initial = build_initial_state(cat, record["config"])
    samples: List[Dict[str, Any]] = []

    def on_decision(i, s, d, h):
        item = {
            "i": i,
            "side": h["side"],
            "by": h.get("by", "human"),
            "decision": d.kind,
            "unit_id": d.unit_id,
            "n_options": len(d.options),
            "label": h.get("label"),
            "action": h["action"],
            "t": h.get("t"),
        }
        if states:
            item["state"] = observation(s)
        samples.append(item)

    final = replay_record(engine, initial, record, on_decision)
    players = {side: {"kind": p.get("kind"), "name": p.get("name")} for side, p in record.get("players", {}).items()}
    return {
        "format": TRAINING_FORMAT,
        "game": {"id": record["id"], "title": record.get("title"), "created": record.get("created"), "updated": record.get("updated"),
                 "status": record.get("status"), "players": players,
                 "lists": {side: (l or {}).get("name") for side, l in (record["config"].get("lists") or {}).items()},
                 "seed": record["config"]["seed"], "layout": record["config"].get("layout")},
        "samples": samples,
        "final_state": observation(final),
        "result": {"scores": final.scoreboard.totals(), "winner": final.scoreboard.leader() if engine.is_over(final) else None,
                   "finished": engine.is_over(final)},
        "events": list(final.events),
        "undos": [{k: u.get(k) for k in ("n", "by", "t", "to", "count", "undone")} for u in record.get("undos", [])],
    }
