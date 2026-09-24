"""Réserves stratégiques V11 (20), Deep Strike (24.09) et Infiltrators (24.20).

* Au déploiement, une unité peut être placée en réserve stratégique (50 % au plus du format de
  partie, en points ; les AIRCRAFT y sont toujours) — simplification : le choix se fait au moment de
  déployer l'unité plutôt que dans l'étape « Declare Battle Formations ».
* À partir du round 2, dans sa phase de mouvement, le joueur sélectionne chaque unité en réserve :
  elle fait un **mouvement d'ingress** (20.04) — entièrement à 6" d'un bord de table, à plus de 8"
  de toute unité ennemie, et avant le round 3 hors de la zone de déploiement adverse — ou reste en
  réserve. Deep Strike : n'importe où à plus de 8" de l'ennemi, zone adverse comprise.
* Fin du round 3 : les unités qui ne sont jamais arrivées sont détruites (sauf les unités remises en
  réserve pendant la bataille, 20.02, et celles embarquées dans un transport arrivé).
* Infiltrators : au déploiement, n'importe où à plus de 8" de la zone de déploiement adverse et de
  toute unité ennemie.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import Disk, disk_gap, disk_polygon_distance, extreme_points, point_in_polygon
from .movement import check_model_positions, pose_disk
from .state import GameState, Unit, other_side

__all__ = ["battle_size_limit", "reserve_points", "reserve_error", "has_deep_strike", "has_infiltrators", "ingress_error",
           "infiltrate_error", "ingress_candidates", "formation_at", "INGRESS_EDGE_IN", "INGRESS_ENEMY_IN"]

INGRESS_EDGE_IN = 6.0  #: 20.04 : distance de mise en place depuis un bord de table
INGRESS_ENEMY_IN = 8.0  #: 20.04 / 24.09 / 24.20 : plus de 8" horizontalement de toute unité ennemie
BATTLE_SIZES = (500, 1000, 2000, 3000)


def _every_model_has(unit: Unit, ability: str) -> bool:
    return all(ds.ability(ability) is not None for ds in (unit.datasheet, *unit.leaders))


def has_deep_strike(unit: Unit) -> bool:
    return _every_model_has(unit, "Deep Strike")


def has_infiltrators(unit: Unit) -> bool:
    return _every_model_has(unit, "Infiltrators")


def battle_size_limit(state: GameState, side: str) -> Optional[int]:
    """Format de partie (points) du camp, d'après sa liste importée ; None = pas de limite (toy model)."""
    al = (state.army_lists or {}).get(side)
    if al is None:
        return None
    pts = al.doc.total_points or al.points
    return next((b for b in BATTLE_SIZES if pts <= b), pts)


def reserve_points(state: GameState, side: str) -> int:
    """Points déjà en réserve stratégique (unités embarquées dans un transport en réserve comprises)."""
    reserve = {u.id for u in state.units.values() if u.side == side and u.in_reserve and not u.is_destroyed}
    total = 0
    for u in state.units.values():
        if u.side == side and not u.is_destroyed and (u.id in reserve or u.embarked_in in reserve):
            total += u.points
    return total


def reserve_error(state: GameState, unit: Unit) -> Optional[str]:
    """Pourquoi l'unité ne peut pas être placée en réserve au déploiement (None = possible)."""
    if unit.has_keyword("Fortification"):
        return "une FORTIFICATION ne peut pas être en réserve"
    limit = battle_size_limit(state, unit.side)
    if limit is not None:
        extra = unit.points + sum(p.points for p in state.passengers(unit.id))
        if reserve_points(state, unit.side) + extra > limit / 2:
            return f"réserves limitées à {limit // 2} pts ({reserve_points(state, unit.side)} déjà en réserve)"
    return None


def _disks(unit: Unit, positions: Dict[str, Sequence[float]]) -> Dict[str, Disk]:
    return {m.id: pose_disk(m, positions[m.id]) for m in unit.alive_models}


def _enemy_disks(state: GameState, side: str) -> List[Disk]:
    return [m.disk for u in state.units_of(other_side(side)) for m in u.alive_models]


def _wholly_near_edge(state: GameState, d: Disk, dist: float) -> bool:
    w, h = state.layout.board
    pts = extreme_points(d)
    return (all(p[0] <= dist + 1e-9 for p in pts) or all(p[0] >= w - dist - 1e-9 for p in pts)
            or all(p[1] <= dist + 1e-9 for p in pts) or all(p[1] >= h - dist - 1e-9 for p in pts))


def _touches_zone(d: Disk, zone) -> bool:
    return disk_polygon_distance(d, zone) <= 1e-9


def _common(state: GameState, unit: Unit, positions: Dict[str, Sequence[float]]) -> Optional[str]:
    missing = [m.id for m in unit.alive_models if m.id not in positions]
    if missing:
        return f"placer toutes les figurines ({', '.join(missing)} manquante(s))"
    return check_model_positions(state, unit, positions, "set_up", 0.0)


def ingress_error(state: GameState, unit: Unit, positions: Dict[str, Sequence[float]], deep_strike: Optional[bool] = None) -> Optional[str]:
    """Placement d'un mouvement d'ingress (20.04 ; Deep Strike 24.09) : message d'erreur ou None."""
    err = _common(state, unit, positions)
    if err:
        return err
    ds = has_deep_strike(unit) if deep_strike is None else deep_strike
    enemies = _enemy_disks(state, unit.side)
    opp_zone = state.layout.deployment_zones[other_side(unit.side)]
    for mid, d in _disks(unit, positions).items():
        if any(disk_gap(d, e) <= INGRESS_ENEMY_IN + 1e-9 for e in enemies):
            return f"{mid} est à 8\" ou moins d'une unité ennemie"
        if ds:
            continue
        if not _wholly_near_edge(state, d, INGRESS_EDGE_IN):
            return f"{mid} n'est pas entièrement à 6\" d'un bord de table"
        if state.battle_round < 3 and _touches_zone(d, opp_zone):
            return f"{mid} est dans la zone de déploiement adverse (interdit avant le round 3)"
    return None


def infiltrate_error(state: GameState, unit: Unit, positions: Dict[str, Sequence[float]]) -> Optional[str]:
    """Infiltrators (24.20) : à plus de 8" de la zone de déploiement adverse et de toute unité ennemie."""
    err = _common(state, unit, positions)
    if err:
        return err
    enemies = [e for e in _enemy_disks(state, unit.side) if state.on_board(e)]
    opp_zone = state.layout.deployment_zones[other_side(unit.side)]
    for mid, d in _disks(unit, positions).items():
        if disk_polygon_distance(d, opp_zone) <= INGRESS_ENEMY_IN + 1e-9:
            return f"{mid} est à 8\" ou moins de la zone de déploiement adverse"
        if any(disk_gap(d, e) <= INGRESS_ENEMY_IN + 1e-9 for e in enemies):
            return f"{mid} est à 8\" ou moins d'une unité ennemie"
    return None


def formation_at(unit: Unit, x: float, y: float) -> Dict[str, Tuple[float, ...]]:
    """Positions de l'unité dans sa formation actuelle, centrée en (x, y) (angles gardés)."""
    ms = unit.alive_models
    cx = sum(m.x for m in ms) / len(ms)
    cy = sum(m.y for m in ms) / len(ms)
    return {m.id: ((m.x - cx + x, m.y - cy + y) if m.is_round else (m.x - cx + x, m.y - cy + y, m.angle)) for m in ms}


def ingress_candidates(state: GameState, unit: Unit, step: float = 1.5, limit: int = 40, mode: str = "ingress") -> List[Tuple[float, float]]:
    """Centres de formation légaux pour un ingress (ou une infiltration, ``mode="infiltrate"``) :
    grille pré-filtrée en vectoriel, validée exactement, sous-échantillonnée régulièrement (sans dé)."""
    w, h = state.layout.board
    ms = unit.alive_models
    if not ms:
        return []
    cx = sum(m.x for m in ms) / len(ms)
    cy = sum(m.y for m in ms) / len(ms)
    reach = max(np.hypot(m.x - cx, m.y - cy) + m.extent for m in ms)
    half_w = max(abs(m.x - cx) + m.extent for m in ms)  # demi-largeur / demi-profondeur de la formation
    half_h = max(abs(m.y - cy) + m.extent for m in ms)
    xs = np.arange(half_w, w - half_w + 1e-9, step)
    ys = np.arange(half_h, h - half_h + 1e-9, step)
    if not len(xs) or not len(ys):
        return []
    gx, gy = np.meshgrid(xs, ys)
    centers = np.stack([gx.ravel(), gy.ravel()], axis=1)
    enemies = [e for e in _enemy_disks(state, unit.side) if state.on_board(e)]
    if enemies:
        exy = np.array([(e.x, e.y) for e in enemies])
        erad = np.array([e.r + np.hypot(e.hx, e.hy) for e in enemies])
        d = np.hypot(centers[:, None, 0] - exy[None, :, 0], centers[:, None, 1] - exy[None, :, 1]) - erad[None, :] - reach
        centers = centers[(d > INGRESS_ENEMY_IN).all(axis=1)]
    deep = has_deep_strike(unit)
    if mode == "ingress" and not deep:
        near = (centers[:, 0] <= INGRESS_EDGE_IN - half_w + 1e-9) | (centers[:, 0] >= w - INGRESS_EDGE_IN + half_w - 1e-9) \
            | (centers[:, 1] <= INGRESS_EDGE_IN - half_h + 1e-9) | (centers[:, 1] >= h - INGRESS_EDGE_IN + half_h - 1e-9)
        centers = centers[near]
    out = []
    check = infiltrate_error if mode == "infiltrate" else ingress_error
    for x, y in centers:
        if check(state, unit, formation_at(unit, float(x), float(y))) is None:
            out.append((round(float(x), 2), round(float(y), 2)))
        if len(out) >= limit * 4:
            break
    if len(out) > limit:
        stride = len(out) / limit
        out = [out[int(k * stride)] for k in range(limit)]
    return out
