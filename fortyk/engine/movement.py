"""Déplacements : légalité, génération de mouvements candidats, charge, pile-in, consolidation.

Représentation retenue pour le toy model : une unité se déplace **en formation** (toutes
ses figurines subissent la même translation), ce qui conserve la cohérence et réduit
l'espace d'actions à un vecteur par unité. Les déplacements individuels n'existent que
pour la charge, le pile-in et la consolidation, où chaque figurine se rapproche de
l'ennemi le plus proche.

Contraintes vérifiées pour un déplacement normal / une Advance :

* chaque figurine parcourt au plus la distance autorisée (ligne droite) ;
* toutes les figurines finissent sur la table ;
* aucun chevauchement de socle à l'arrivée, ni sur le trajet (échantillonné) ;
* aucune figurine ne passe ni ne finit à portée d'engagement d'un ennemi.

Un Fall Back peut traverser la portée d'engagement (et les figurines ennemies) mais
doit finir hors de portée d'engagement. Les décors sont tous franchissables par
l'infanterie dans ce toy model (pas de murs infranchissables, pas d'étages).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .fastgeo import shape_arrays, translations_legal
from .geometry import Disk, Point, core_points, disk_gap, disks_overlap, dist, extreme_points, point_in_polygon, within
from .rules import DEFAULT_RULES, RulesConfig
from .state import GameState, Model, Unit
from .effects import effect_total

__all__ = [
    "MoveKind",
    "FormationMove",
    "legal_translation",
    "legal_translations",
    "candidate_moves",
    "apply_translation",
    "check_model_positions",
    "apply_model_positions",
    "pose_disk",
    "move_distance",
    "is_monster_or_vehicle",
    "charge_move",
    "charge_gap",
    "charge_roll_succeeds",
    "charge_reachable_targets",
    "check_charge_positions",
    "auto_charge_move",
    "pile_in",
    "consolidate",
    "nearest_enemy_model",
    "unit_vector",
]

PATH_STEP_IN = 0.5


class MoveKind:
    STATIONARY = "stationary"
    NORMAL = "normal"
    ADVANCE = "advance"
    FALL_BACK = "fall_back"  #: Fall Back en « ordered retreat » (09.07) : ne traverse pas les ennemis
    DESPERATE = "desperate"  #: Fall Back en « desperate escape » : traverse les ennemis, jet de danger par figurine


def is_monster_or_vehicle(unit: Unit) -> bool:
    return unit.has_keyword("Vehicle") or unit.has_keyword("Monster")


@dataclass(frozen=True)
class FormationMove:
    """Translation (dx, dy) de toute l'unité, avec la nature du mouvement.
    Pour une Advance, ``dx, dy`` est la direction souhaitée à pleine distance M + D6 ;
    le moteur raccourcit au jet réel puis à la légalité."""

    kind: str
    dx: float = 0.0
    dy: float = 0.0
    label: str = ""

    @property
    def distance(self) -> float:
        return math.hypot(self.dx, self.dy)

    def scaled(self, factor: float) -> "FormationMove":
        return FormationMove(self.kind, self.dx * factor, self.dy * factor, self.label)


def unit_vector(a: Point, b: Point) -> Tuple[float, float]:
    d = dist(a, b)
    if d <= 1e-9:
        return (0.0, 0.0)
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d)


def nearest_enemy_model(state: GameState, point: Point, side: str) -> Optional[Model]:
    best, best_d = None, float("inf")
    for m in state.enemy_models(side):
        d = dist(point, m.position)
        if d < best_d:
            best, best_d = m, d
    return best


# ------------------------------------------------------------- légalité


def _blocked_by_models(disk: Disk, others: Sequence[Disk]) -> bool:
    return any(disks_overlap(disk, o) for o in others)


def _in_engagement(disk: Disk, enemies: Sequence[Disk], er: float) -> bool:
    return any(within(disk, e, er) for e in enemies)


def _move_arrays(state: GameState, unit: Unit):
    """Tableaux numpy (positions, rayons) de l'unité, des autres unités amies et des ennemis, plus les
    empreintes complètes (N, 6) quand l'une d'elles n'est pas ronde (coques, ovales), sinon None."""
    ms = unit.alive_models
    friends = [m for u in state.units_of(unit.side) if u.id != unit.id for m in u.alive_models]
    enemies = state.enemy_models(unit.side)

    def arr(models):
        if not models:
            return np.zeros((0, 2)), np.zeros(0)
        return np.array([(m.x, m.y) for m in models], dtype=float), np.array([m.radius for m in models], dtype=float)

    shapes = None
    if not all(m.hx == 0.0 and m.hy == 0.0 for group in (ms, friends, enemies) for m in group):
        shapes = tuple(shape_arrays([m.disk for m in group]) for group in (ms, friends, enemies))
    # 17.01 : en mouvement normal / Advance, un MONSTER / VEHICLE passe sur les figurines ennemies qui
    # ne sont pas elles-mêmes MONSTER / VEHICLE
    mv_blocks = None
    if is_monster_or_vehicle(unit):
        mv = {}
        mv_blocks = np.array([mv.setdefault(m.unit_id, is_monster_or_vehicle(state.unit(m.unit_id))) for m in enemies], dtype=bool)
    return arr(ms) + arr(friends) + arr(enemies) + (shapes, mv_blocks)


def _legal_with(state: GameState, arrays, vecs, kind: str, max_distance: float, engagement_range: Optional[float] = None) -> np.ndarray:
    mxy, mr, fxy, fr, exy, er, shapes, mv_blocks = arrays
    rng = state.rules.engagement_range_in if engagement_range is None else engagement_range
    blocks = mv_blocks if kind in (MoveKind.NORMAL, MoveKind.ADVANCE) else None
    return translations_legal(mxy, mr, vecs, kind, max_distance, state.layout.board, fxy, fr, exy, er, rng, shapes=shapes, enemy_blocks=blocks)


def legal_translations(state: GameState, unit: Unit, vecs, kind: str, max_distance: float, engagement_range: Optional[float] = None) -> np.ndarray:
    """Légalité de plusieurs translations (C, 2) de l'unité, en un seul calcul vectorisé
    (voir :func:`fortyk.engine.fastgeo.translations_legal`). ``engagement_range`` remplace la portée
    d'engagement (ex. 9" pour finir un mouvement de scout loin de l'ennemi)."""
    return _legal_with(state, _move_arrays(state, unit), vecs, kind, max_distance, engagement_range)


def legal_translation(
    state: GameState,
    unit: Unit,
    dx: float,
    dy: float,
    kind: str,
    max_distance: float,
) -> bool:
    """La translation (dx, dy) de l'unité est-elle légale pour ce type de mouvement ?

    V11 : longueur ≤ ``max_distance`` ; arrivée sur la table, sans chevauchement et hors portée
    d'engagement ; on traverse les amis ; ``normal`` / ``advance`` / ``fall_back`` ne traversent pas les
    socles ennemis (sauf un M/V sur l'infanterie en normal / Advance) ; ``desperate`` les traverse.
    Les trajectoires sont testées de façon exacte (segment contre socle)."""
    return bool(legal_translations(state, unit, [(dx, dy)], kind, max_distance)[0])


def _shrink_batch(state: GameState, unit: Unit, moves: Sequence[FormationMove], max_distance: float, tries: int = 8) -> List[Optional[FormationMove]]:
    """Pour chaque mouvement, la plus longue version légale parmi ``tries`` raccourcissements
    réguliers (1, 7/8, …, 1/8 de la longueur plafonnée à ``max_distance``) ; None si aucune."""
    if not moves:
        return []
    by_kind: Dict[str, List[int]] = {}
    for i, mv in enumerate(moves):
        by_kind.setdefault(mv.kind, []).append(i)
    out: List[Optional[FormationMove]] = [None] * len(moves)
    fr = 1 - np.arange(tries) / tries  # (T,)
    arrays = _move_arrays(state, unit)
    for kind, idx in by_kind.items():
        vecs = np.array([(moves[i].dx, moves[i].dy) for i in idx], dtype=float)
        length = np.hypot(vecs[:, 0], vecs[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            scale0 = np.where(length > 1e-9, np.minimum(1.0, max_distance / length), 1.0)
        scales = scale0[:, None] * fr[None, :]  # (C, T)
        # 1er passage à pleine longueur ; on ne raccourcit que les mouvements refusés
        legal = np.zeros((len(idx), tries), dtype=bool)
        legal[:, 0] = _legal_with(state, arrays, vecs * scales[:, :1], kind, max_distance)
        retry = np.flatnonzero(~legal[:, 0])
        if len(retry) and tries > 1:
            sub = (vecs[retry, None, :] * scales[retry, 1:, None]).reshape(-1, 2)
            legal[retry, 1:] = _legal_with(state, arrays, sub, kind, max_distance).reshape(len(retry), tries - 1)
        for row, i in enumerate(idx):
            if length[row] <= 1e-9:
                out[i] = moves[i]
                continue
            hits = np.flatnonzero(legal[row] & (scales[row] > 1e-6))
            if len(hits):
                out[i] = moves[i].scaled(float(scales[row, hits[0]]))
    return out


def shrink_to_legal(state: GameState, unit: Unit, move: FormationMove, max_distance: float, tries: int = 8) -> Optional[FormationMove]:
    """Raccourcit progressivement une translation jusqu'à la rendre légale ; None si rien ne passe."""
    return _shrink_batch(state, unit, [move], max_distance, tries)[0]


def apply_translation(unit: Unit, dx: float, dy: float) -> None:
    for m in unit.alive_models:
        m.move_to(m.x + dx, m.y + dy)
    unit.moved_in = max(unit.moved_in, math.hypot(dx, dy))
    if math.hypot(dx, dy) > 1e-9:
        unit.remained_stationary = False


# ------------------------------------------------------------- figurine par figurine


def pose_disk(model: Model, pos: Optional[Sequence[float]]) -> Disk:
    """Empreinte de la figurine posée en ``pos`` = (x, y) ou (x, y, angle) ; l'angle n'a de sens que
    pour une empreinte non ronde (un socle rond pivote librement)."""
    d = model.disk
    if pos is None:
        return d
    if len(pos) >= 3 and not (d.hx == 0.0 and d.hy == 0.0):
        return d._replace(x=float(pos[0]), y=float(pos[1]), a=float(pos[2]))
    return d._replace(x=float(pos[0]), y=float(pos[1]))


def _angle_delta(a0: float, a1: float) -> float:
    """Rotation la plus courte de a0 vers a1, dans [-π, π]."""
    d = a1 - a0
    return math.atan2(math.sin(d), math.cos(d))


def _rotation_for(start: Disk, end: Disk) -> float:
    """Rotation retenue entre deux poses : une empreinte rectangulaire ou ovale est symétrique par un
    demi-tour, on garde la rotation équivalente la plus courte."""
    d = _angle_delta(start.a, end.a)
    alt = _angle_delta(0.0, d - math.pi)
    return alt if abs(alt) < abs(d) else d


def move_distance(start: Disk, end: Disk) -> float:
    """Distance parcourue au sens des règles V11 (03.01) : on mesure depuis le même point du socle au
    départ et à l'arrivée, et pivoter autour du centre (ou de l'axe central d'une coque, 17.02) ne
    compte pas — c'est donc le déplacement du centre."""
    return math.hypot(end.x - start.x, end.y - start.y)


def _path_poses(start: Disk, end: Disk, step: float = None) -> List[Disk]:
    """Poses intermédiaires (hors départ) d'un trajet en ligne droite avec pivot progressif."""
    step = step or PATH_STEP_IN
    length = move_distance(start, end) + (abs(_rotation_for(start, end)) * max(start.hx, start.hy) if not (start.hx == 0.0 and start.hy == 0.0) else 0.0)
    steps = max(2, int(math.ceil(length / step)) + 1)
    th = 0.0 if (start.hx == 0.0 and start.hy == 0.0) else _rotation_for(start, end)
    out = []
    for k in range(1, steps):
        t = k / (steps - 1)
        out.append(start._replace(x=start.x + (end.x - start.x) * t, y=start.y + (end.y - start.y) * t, a=start.a + th * t))
    out[-1] = end
    return out


def check_model_positions(
    state: GameState,
    unit: Unit,
    positions: Dict[str, Sequence[float]],
    kind: str,
    max_distance: float,
    check_zone: Optional[str] = None,
) -> Optional[str]:
    """Valide un placement figurine par figurine ; retourne un message d'erreur, ou None si légal.

    ``positions`` : model_id → (x, y) ou (x, y, angle) ; une figurine absente reste où elle est.
    ``kind`` : ``normal`` / ``advance_move`` / ``fall_back`` (ne pas traverser de socle ennemi, finir
    hors portée d'engagement ; un M/V passe sur l'infanterie en normal / Advance), ``desperate`` (on
    traverse, fin hors portée), ``deploy`` (pas de limite de distance, toute l'empreinte dans la zone
    ``check_zone``), ``set_up`` (mise en place hors déploiement : ingress, figurines ramenées — les
    contraintes propres sont vérifiées par l'appelant). Toujours : sur la table, pas de chevauchement, cohérence d'unité à l'arrivée ; on
    traverse les amis ; pivoter ne compte pas dans la distance (:func:`move_distance`).
    """
    rules = state.rules
    er = rules.engagement_range_in
    alive = unit.alive_models
    by_id = {m.id: m for m in alive}
    for mid in positions:
        if mid not in by_id:
            return f"figurine inconnue ou détruite : {mid}"
    finals = {m.id: pose_disk(m, positions.get(m.id)) for m in alive}
    # les figurines pas encore déployées (hors table) ne comptent pas comme obstacles
    enemies = [e.disk for e in state.enemy_models(unit.side) if state.on_board(e.disk)]
    friendly_others = [m.disk for u in state.units_of(unit.side) if u.id != unit.id for m in u.alive_models if state.on_board(m.disk)]
    if kind == "deploy":
        zone = state.layout.deployment_zones[check_zone or unit.side]
        for mid, d in finals.items():
            if not all(point_in_polygon(p, zone) for p in extreme_points(d)):
                return f"{mid} n'est pas entièrement dans la zone de déploiement"
    for mid, d in finals.items():
        if not state.on_board(d):
            return f"{mid} sort de la table"
        if any(disks_overlap(d, o) for o in friendly_others) or any(disks_overlap(d, o) for o in enemies):
            return f"{mid} chevauche une autre figurine"
    ids = list(finals)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if disks_overlap(finals[a], finals[b]):
                return f"{a} et {b} se chevauchent"
    if kind not in ("deploy", "set_up"):
        normalish = kind in (MoveKind.NORMAL, "advance_move", MoveKind.ADVANCE)
        # 03.01 : on traverse les amis mais pas les socles ennemis ; 17.01 : un M/V passe sur les
        # figurines ennemies non M/V en mouvement normal / Advance ; Desperate Escape : on traverse tout
        if normalish and is_monster_or_vehicle(unit):
            path_blockers = [e.disk for e in state.enemy_models(unit.side) if state.on_board(e.disk) and is_monster_or_vehicle(state.unit(e.unit_id))]
        elif normalish or kind == MoveKind.FALL_BACK:
            path_blockers = enemies
        else:
            path_blockers = []
        for m in alive:
            start, end = m.disk, finals[m.id]
            length = move_distance(start, end)
            if length > max_distance + 1e-6:
                return f"{m.id} bouge de {length:.1f}\", plus que {max_distance:g}\""
            if path_blockers:
                for pos in _path_poses(start, end):
                    if _blocked_by_models(pos, path_blockers):
                        return f"{m.id} traverse une figurine ennemie"
            if (normalish or kind in (MoveKind.FALL_BACK, MoveKind.DESPERATE)) and _in_engagement(end, enemies, er):
                return f"{m.id} finit à portée d'engagement d'un ennemi"
    # cohérence à l'arrivée
    if len(alive) > 1:
        for mid, d in finals.items():
            others = [finals[o] for o in finals if o != mid]
            if not any(within(d, o, rules.coherency_range_in) for o in others):
                return f"{mid} n'est à 2\" d'aucune autre figurine (cohérence)"
            if any(disk_gap(d, o) > rules.coherency_max_spread_in for o in others):
                return f"{mid} est à plus de 9\" d'une figurine de l'unité (cohérence)"
    return None


def apply_model_positions(unit: Unit, positions: Dict[str, Sequence[float]]) -> float:
    """Applique un placement figurine par figurine (angle compris) ; retourne le plus grand déplacement."""
    longest = 0.0
    for m in unit.alive_models:
        if m.id in positions:
            end = pose_disk(m, positions[m.id])
            longest = max(longest, move_distance(m.disk, end))
            m.move_to(end.x, end.y, end.a if not m.is_round else None)
    unit.moved_in = max(unit.moved_in, longest)
    if longest > 1e-9:
        unit.remained_stationary = False
    return longest


# ------------------------------------------------------------- candidats


def candidate_moves(state: GameState, unit: Unit, distances: Sequence[float] = (1.0, 0.5)) -> List[FormationMove]:
    """Mouvements candidats discrets pour une unité : rester, 8 directions cardinales, vers chaque
    objectif, vers / à l'opposé de l'ennemi le plus proche, à pleine et demi-distance ; Advance
    dans les mêmes directions ; Fall Back si engagée. Tous sont déjà raccourcis à la légalité."""
    rules = state.rules
    out: List[FormationMove] = [FormationMove(MoveKind.STATIONARY, label="reste immobile")]
    if unit.is_destroyed:
        return out
    M = unit.move_in + (effect_total(state, unit.id, "move_mod") if state.effects else 0)
    c = unit.centroid
    engaged = state.is_engaged(unit)

    directions: List[Tuple[Tuple[float, float], str]] = []
    for k in range(8):
        a = math.pi / 4 * k
        directions.append(((math.cos(a), math.sin(a)), f"cap {k * 45}°"))
    for o in state.layout.objectives:
        v = unit_vector(c, o.center)
        if v != (0.0, 0.0):
            directions.append((v, f"vers {o.id}"))
    enemy = nearest_enemy_model(state, c, unit.side)
    if enemy is not None:
        v = unit_vector(c, enemy.position)
        if v != (0.0, 0.0):
            directions.append((v, "vers l'ennemi"))
            directions.append(((-v[0], -v[1]), "à l'opposé de l'ennemi"))

    if engaged:
        # Fall Back (09.07) : « ordered retreat » (sans traverser l'ennemi) si l'unité n'est pas
        # battle-shocked ; sinon « desperate escape » (on traverse, un jet de danger par figurine).
        # Une unité non battle-shocked peut aussi choisir le desperate escape : on le propose dans les
        # directions où le repli ordonné est impossible.
        dirs = [(v, label) for v, label in directions]
        desperate = [FormationMove(MoveKind.DESPERATE, v[0] * M, v[1] * M, f"desperate escape {label}") for v, label in dirs]
        if unit.battle_shocked:
            out.extend(mv for mv in _shrink_batch(state, unit, desperate, M) if mv is not None and mv.distance > 1e-6)
            return _dedupe(out)
        fb = [FormationMove(MoveKind.FALL_BACK, v[0] * M, v[1] * M, f"repli {label}") for v, label in dirs]
        ordered = _shrink_batch(state, unit, fb, M)
        out.extend(mv for mv in ordered if mv is not None and mv.distance > 1e-6)
        missing = [d for d, mv in zip(desperate, ordered) if mv is None or mv.distance <= 1e-6]
        out.extend(mv for mv in _shrink_batch(state, unit, missing, M) if mv is not None and mv.distance > 1e-6)
        return _dedupe(out)

    normal = [FormationMove(MoveKind.NORMAL, v[0] * M * f, v[1] * M * f, f"{label} ({M * f:g}\")") for v, label in directions for f in distances]
    # Advance : la distance réelle (M + D6) n'est connue qu'après le jet ; on stocke la direction à M + 6
    advances = [FormationMove(MoveKind.ADVANCE, v[0] * (M + 6), v[1] * (M + 6), f"advance {label}") for v, label in directions]
    shrunk_normal = _shrink_batch(state, unit, normal, M)
    shrunk_adv = _shrink_batch(state, unit, advances, M + 6)
    per_dir = len(distances)
    for k, adv in enumerate(advances):
        for ok in shrunk_normal[k * per_dir:(k + 1) * per_dir]:
            if ok is not None and ok.distance > 1e-6:
                out.append(ok)
        if shrunk_adv[k] is not None:
            out.append(adv)
    return _dedupe(out)


def _dedupe(moves: List[FormationMove]) -> List[FormationMove]:
    seen = set()
    out = []
    for mv in moves:
        key = (mv.kind, round(mv.dx, 2), round(mv.dy, 2))
        if key not in seen:
            seen.add(key)
            out.append(mv)
    return out


# ------------------------------------------------------------- charge et combat


def _approach_distance(d: Disk, ux: float, uy: float, target: Disk, stop_gap: float, upper: float) -> float:
    """Distance à parcourir le long de (ux, uy) pour arriver à ``stop_gap`` du socle visé. La distance
    entre deux empreintes convexes est convexe le long d'une droite : une bissection suffit."""
    if d.hx == 0.0 and d.hy == 0.0 and target.hx == 0.0 and target.hy == 0.0:
        return dist(d.center, target.center) - d.r - target.r - stop_gap
    if disk_gap(d, target) <= stop_gap:
        return 0.0
    if disk_gap(d.translated(ux * upper, uy * upper), target) > stop_gap:
        return upper
    lo, hi = 0.0, upper
    for _ in range(24):
        mid = (lo + hi) / 2
        if disk_gap(d.translated(ux * mid, uy * mid), target) > stop_gap:
            lo = mid
        else:
            hi = mid
    return lo


def _step_towards(model: Model, target: Disk, max_distance: float, stop_gap: float, blockers: Sequence[Disk], board_ok) -> float:
    """Avance la figurine en ligne droite vers le socle ``target`` d'au plus ``max_distance``, en
    s'arrêtant à ``stop_gap`` de lui et sans chevaucher ``blockers``. Retourne la distance parcourue."""
    d = model.disk
    ux, uy = unit_vector(model.position, target.center)
    if (ux, uy) == (0.0, 0.0):
        return 0.0
    reach = _approach_distance(d, ux, uy, target, stop_gap, dist(model.position, target.center))
    travel = max(0.0, min(max_distance, reach))
    if travel <= 1e-9:
        return 0.0
    # recule si la position finale chevauche un socle (échantillonnage arrière)
    steps = max(1, int(math.ceil(travel / 0.25)))
    for k in range(steps, 0, -1):
        t = travel * k / steps
        cand = d.translated(ux * t, uy * t)
        if board_ok(cand) and not any(disks_overlap(cand, b) for b in blockers):
            model.move_to(cand.x, cand.y)
            return t
    return 0.0


CONTACT_TOL_IN = 0.1  #: tolérance pour « socle à socle »
CHARGE_STEP_TOL_IN = 0.15  #: tolérance sur les paliers 1" / 2" de la règle de charge


def charge_gap(unit: Unit, target: Unit) -> float:
    """Distance socle à socle entre les figurines les plus proches : c'est le jet de charge minimal
    pour arriver au contact (6.3" → il faut 7)."""
    return unit.min_gap_to(target)


def charge_roll_succeeds(roll: int, gap: float) -> bool:
    """Un double 1 (2) est toujours un échec ; sinon il faut couvrir la distance socle à socle."""
    return roll > 2 and roll + 1e-9 >= gap


def charge_reachable_targets(state: GameState, unit: Unit, candidates: Sequence[Unit], roll: int) -> List[Unit]:
    """Parmi ``candidates`` (déjà filtrées : à 12", restrictions de capacités), celles qu'une
    figurine peut atteindre socle à socle avec ce jet."""
    return [t for t in candidates if charge_roll_succeeds(roll, charge_gap(unit, t))]


def _model_gap_to_unit(disk: Disk, target: Unit) -> float:
    return min(disk_gap(disk, t) for t in target.disks())


def _ring_distance(model: Disk, target: Disk, ux: float, uy: float, gap: float) -> float:
    """Distance du centre de ``target`` où poser ``model`` dans la direction (ux, uy) pour que les
    empreintes soient à ``gap`` l'une de l'autre (la distance croît le long du rayon : bissection)."""
    if model.hx == 0.0 and model.hy == 0.0 and target.hx == 0.0 and target.hy == 0.0:
        return model.r + target.r + gap
    lo, hi = 0.0, model.extent + target.extent + gap + 0.05
    for _ in range(14):
        mid = (lo + hi) / 2
        if disk_gap(model.moved_to(target.x + ux * mid, target.y + uy * mid), target) < gap:
            lo = mid
        else:
            hi = mid
    return hi


def _as_units(targets) -> List[Unit]:
    return [targets] if isinstance(targets, Unit) else list(targets)


def _model_gap_to_units(disk: Disk, targets: Sequence[Unit]) -> float:
    return min((disk_gap(disk, t) for u in targets for t in u.disks()), default=float("inf"))


def _free_spot_within(model: Disk, targets, reach: float, blockers: Sequence[Disk], gap_max: float, board_ok, angles: int = 16) -> Optional[Point]:
    """Un point où poser ``model`` (orientation inchangée) à au plus ``gap_max`` (socle à socle) d'une
    figurine des unités ``targets``, atteignable en ligne droite en ≤ ``reach``, sans chevaucher
    ``blockers`` ; le moins coûteux, ou None."""
    best, best_d = None, float("inf")
    for target in _as_units(targets):
        for t in target.alive_models:
            td = t.disk
            for ring in (0.0, gap_max / 2, gap_max) if gap_max > 0 else (0.0,):
                for k in range(angles):
                    a = 2 * math.pi * k / angles
                    ux, uy = math.cos(a), math.sin(a)
                    radius = _ring_distance(model, td, ux, uy, min(ring, gap_max) + 0.01)
                    cand = model.moved_to(t.x + radius * ux, t.y + radius * uy)
                    d = dist(model.center, cand.center)
                    if d > reach + 1e-9 or d >= best_d:
                        continue
                    if not board_ok(cand) or any(disks_overlap(cand, b) for b in blockers):
                        continue
                    best, best_d = cand.center, d
    return best


def check_charge_positions(state: GameState, unit: Unit, targets, positions: Dict[str, Point], roll: int) -> Optional[str]:
    """Valide un mouvement de charge placé figurine par figurine ; message d'erreur ou None.

    ``targets`` : l'unité chargée ou la liste des cibles de la charge (V11 : une ou plusieurs).

    * chaque figurine parcourt au plus le jet, reste sur la table, ne chevauche rien, ne traverse pas de
      socle ennemi (on traverse les amis) ; cohérence à l'arrivée ;
    * au moins une figurine finit socle à socle avec une figurine d'une cible (validé avec Valentin) et
      l'unité finit engagée (2") avec chacune des cibles (11.04) ;
    * aucune figurine ne finit à portée d'engagement d'une unité ennemie qui n'est pas une cible ;
    * chaque figurine qui pouvait arriver à 1" d'une cible doit l'avoir fait ; sinon engagée (2") si elle
      pouvait ; sinon elle doit finir plus près d'une cible qu'au départ.
    """
    targets = _as_units(targets)
    rules = state.rules
    er = rules.engagement_range_in
    alive = unit.alive_models
    by_id = {m.id: m for m in alive}
    for mid in positions:
        if mid not in by_id:
            return f"figurine inconnue ou détruite : {mid}"
    finals = {m.id: pose_disk(m, positions.get(m.id)) for m in alive}
    enemies = [e for e in state.enemies_of(unit.side)]
    enemy_disks = [m.disk for e in enemies for m in e.alive_models]
    friendly_others = [m.disk for u in state.units_of(unit.side) if u.id != unit.id for m in u.alive_models if state.on_board(m.disk)]
    if not any(t.alive_models for t in targets):
        return "la cible n'existe plus"
    target_ids = {t.id for t in targets}
    for m in alive:
        start, end = m.disk, finals[m.id]
        length = move_distance(start, end)
        if length > roll + 1e-6:
            return f"{m.id} bouge de {length:.1f}\", plus que le jet de charge ({roll})"
        if not state.on_board(end):
            return f"{m.id} sort de la table"
        if any(disks_overlap(end, o) for o in friendly_others) or any(disks_overlap(end, o) for o in enemy_disks):
            return f"{m.id} chevauche une autre figurine"
        for pos in _path_poses(start, end):
            if _blocked_by_models(pos, enemy_disks):
                return f"{m.id} traverse une figurine ennemie"
    ids = list(finals)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if disks_overlap(finals[a], finals[b]):
                return f"{a} et {b} se chevauchent"
    for e in enemies:
        if e.id in target_ids:
            continue
        for mid, d in finals.items():
            if any(within(d, x, er) for x in e.disks()):
                return f"{mid} finit à portée d'engagement de {e.id}, qui n'est pas une cible de la charge"
    if not any(_model_gap_to_units(d, targets) <= CONTACT_TOL_IN for d in finals.values()):
        return "aucune figurine n'arrive socle à socle avec une cible"
    for t in targets:
        if not any(_model_gap_to_units(d, [t]) <= er + 1e-6 for d in finals.values()):
            return f"l'unité doit finir engagée (2\") avec chaque cible : {t.id} ne l'est pas"
    for m in alive:
        g0 = _model_gap_to_units(m.disk, targets)
        g1 = _model_gap_to_units(finals[m.id], targets)
        if g1 <= 1.0 + CHARGE_STEP_TOL_IN:
            continue  # arrivée à 1" : rien à redire
        # « pouvait arriver » = une place libre existe, compte tenu des autres figurines à leur position finale
        blockers = enemy_disks + friendly_others + [finals[o] for o in finals if o != m.id]
        if _free_spot_within(m.disk, targets, roll, blockers, 1.0, state.on_board) is not None:
            return f"{m.id} pouvait arriver à 1\" d'une cible : elle doit le faire (elle finit à {g1:.1f}\")"
        if g1 <= er + CHARGE_STEP_TOL_IN:
            continue
        if _free_spot_within(m.disk, targets, roll, blockers, er, state.on_board) is not None:
            return f"{m.id} pouvait finir engagée (2\") : elle doit le faire (elle finit à {g1:.1f}\")"
        if g1 >= g0 - 1e-6:
            return f"{m.id} doit finir plus près d'une cible qu'au départ ({g0:.1f}\")"
    if len(alive) > 1:
        for mid, d in finals.items():
            others = [finals[o] for o in finals if o != mid]
            if not any(within(d, o, rules.coherency_range_in) for o in others):
                return f"{mid} n'est à 2\" d'aucune autre figurine (cohérence)"
            if any(disk_gap(d, o) > rules.coherency_max_spread_in for o in others):
                return f"{mid} est à plus de 9\" d'une figurine de l'unité (cohérence)"
    return None


def auto_charge_move(state: GameState, unit: Unit, targets, roll: int) -> Tuple[bool, str]:
    """Mouvement de charge automatique : d'abord, pour chaque cible, la figurine la plus proche va au
    contact ; puis chaque autre figurine cherche une place au contact (puis à 1", puis à 2") d'une
    cible, les plus proches d'abord, sinon avance en ligne droite. Réussite : une figurine socle à socle,
    l'unité engagée avec chaque cible, en cohérence et hors portée d'engagement des autres ennemis ;
    sinon les figurines sont remises en place. Retourne (réussite, raison de l'échec)."""
    targets = _as_units(targets)
    rules = state.rules
    er = rules.engagement_range_in
    saved = [(m, m.x, m.y, m.angle) for m in unit.alive_models]
    others = [m.disk for u in state.on_table_units() if u.id != unit.id for m in u.alive_models]
    enemy_disks = [m.disk for e in state.enemies_of(unit.side) for m in e.alive_models]
    if not any(t.alive_models for t in targets):
        return False, "la cible n'existe plus"
    target_ids = {t.id for t in targets}

    def place_towards(m, goal_units) -> None:
        blockers = list(others) + [x.disk for x in unit.alive_models if x is not m]
        spot = None
        for gap_max in (0.0, 1.0, er):
            spot = _free_spot_within(m.disk, goal_units, roll, blockers, gap_max, state.on_board)
            if spot is not None:
                break
        if spot is not None:
            # trajet en ligne droite : on ne traverse pas de socle ennemi (on traverse les amis)
            if all(not any(disks_overlap(pos, b) for b in enemy_disks) for pos in _path_poses(m.disk, m.disk.moved_to(*spot))):
                m.move_to(*spot)
                return
        tgt = min((t for u in goal_units for t in u.alive_models), key=lambda t: disk_gap(m.disk, t.disk))
        _step_towards(m, tgt.disk, roll, stop_gap=0.0, blockers=blockers, board_ok=state.on_board)

    moved = set()
    # une figurine vers chaque cible (la plus proche de celle-ci) : l'unité doit engager toutes les cibles
    for t in targets:
        free = [m for m in unit.alive_models if m.id not in moved]
        if not free or not t.alive_models:
            continue
        m = min(free, key=lambda mm: _model_gap_to_units(mm.disk, [t]))
        place_towards(m, [t])
        moved.add(m.id)
    for m in sorted(unit.alive_models, key=lambda mm: _model_gap_to_units(mm.disk, targets)):
        if m.id not in moved:
            place_towards(m, targets)
    # cohérence : ramène les traînards vers la figurine de tête si besoin
    if not unit.coherency_ok(rules):
        lead = min(unit.alive_models, key=lambda mm: _model_gap_to_units(mm.disk, targets))
        for m in unit.alive_models:
            if m is lead:
                continue
            if not any(within(m.disk, o.disk, rules.coherency_range_in) for o in unit.alive_models if o is not m):
                blockers = list(others) + [x.disk for x in unit.alive_models if x is not m]
                start = next((x, y) for mm, x, y, _ in saved if mm is m)
                remaining = max(0.0, roll - dist(start, m.position))
                _step_towards(m, lead.disk, remaining, stop_gap=0.1, blockers=blockers, board_ok=state.on_board)
    reason = ""
    if not any(_model_gap_to_units(m.disk, targets) <= CONTACT_TOL_IN for m in unit.alive_models):
        reason = "aucune figurine n'arrive socle à socle"
    elif not unit.coherency_ok(rules):
        reason = "cohérence impossible à conserver"
    else:
        for t in targets:
            if not unit.in_engagement_range_of(t, rules):
                reason = f"impossible de finir engagée avec {t.id}"
                break
        if not reason:
            for e in state.enemies_of(unit.side):
                if e.id not in target_ids and unit.in_engagement_range_of(e, rules):
                    reason = f"impossible sans finir à portée d'engagement de {e.id}, qui n'est pas une cible"
                    break
    if reason:
        for m, x, y, a in saved:
            m.move_to(x, y, a)
        return False, reason
    unit.remained_stationary = False
    unit.moved_in = max(unit.moved_in, float(roll))
    return True, ""


def charge_move(state: GameState, unit: Unit, target, roll: int) -> bool:
    """Compatibilité : charge automatique (voir :func:`auto_charge_move`)."""
    targets = _as_units(target)
    if roll <= 2 or not all(roll + 1e-9 >= charge_gap(unit, t) for t in targets):
        return False
    return auto_charge_move(state, unit, targets, roll)[0]


def _move_towards_units(state: GameState, unit: Unit, targets: Sequence[Unit], max_distance: float, lock_contact: bool = True) -> None:
    """Chaque figurine qui n'est pas socle à socle avec un ennemi (``lock_contact``) avance vers la
    figurine la plus proche des unités ``targets`` : elle finit plus près, engagée si possible, sans
    chevaucher personne ; un pas qui casserait la cohérence est annulé."""
    others = [m.disk for u in state.on_table_units() if u.id != unit.id for m in u.alive_models]
    enemy_disks = [m.disk for e in state.enemies_of(unit.side) for m in e.alive_models]
    target_models = [m for t in targets for m in t.alive_models]
    if not target_models:
        return
    for m in unit.alive_models:
        md = m.disk
        if lock_contact and any(disk_gap(md, e) <= CONTACT_TOL_IN for e in enemy_disks):
            continue  # 12.03 / 12.08 : une figurine socle à socle avec un ennemi ne bouge pas
        tgt = min(target_models, key=lambda e: disk_gap(md, e.disk))
        blockers = others + [x.disk for x in unit.alive_models if x is not m]
        before = (m.x, m.y)
        _step_towards(m, tgt.disk, max_distance, stop_gap=0.05, blockers=blockers, board_ok=state.on_board)
        if not unit.coherency_ok(state.rules):
            m.move_to(*before)


def _engagements(state: GameState, unit: Unit) -> Dict[str, set]:
    """Pour chaque figurine : les unités ennemies avec lesquelles elle est engagée."""
    er = state.rules.engagement_range_in
    enemies = state.enemies_of(unit.side)
    return {m.id: {e.id for e in enemies if any(within(m.disk, d, er) for d in e.disks())} for m in unit.alive_models}


def _keeps_engagements(state: GameState, unit: Unit, before: Dict[str, set]) -> bool:
    """Chaque figurine qui était engagée avec une unité ennemie l'est encore."""
    after = _engagements(state, unit)
    return all(before[mid] <= after.get(mid, set()) for mid in before)


def pile_in(state: GameState, unit: Unit, max_distance: Optional[float] = None, targets: Optional[Sequence[Unit]] = None) -> bool:
    """Pile-in (V11 12.03) : cibles = toutes les unités ennemies engagées, sinon l'unité ennemie la
    plus proche à 5" ; les figurines socle à socle ne bougent pas ; chaque figurine qui bouge finit
    plus près de la cible la plus proche, engagée si possible ; ensuite l'unité doit être engagée et
    chaque figurine engagée au départ l'être encore avec la même unité. Sinon rien ne bouge.
    Retourne True si l'unité a fait son pile-in."""
    rules = state.rules
    dist_max = rules.pile_in_in if max_distance is None else max_distance
    engaged_with = state.enemies_in_engagement_range(unit)
    if engaged_with:
        targets = engaged_with
    else:
        near = [e for e in state.enemies_of(unit.side) if unit.min_gap_to(e) <= rules.pile_in_target_range_in + 1e-9]
        if targets is not None:
            near = [e for e in near if e in targets]
        if not near:
            return False
        targets = [min(near, key=lambda e: unit.min_gap_to(e))]
    saved = [(m, m.x, m.y) for m in unit.alive_models]
    before = _engagements(state, unit)
    _move_towards_units(state, unit, targets, dist_max)
    if not state.is_engaged(unit) or not _keeps_engagements(state, unit, before):
        for m, x, y in saved:
            m.move_to(x, y)
        return False
    return True


def consolidate(state: GameState, unit: Unit) -> Tuple[str, List[str]]:
    """Consolidation (V11 12.08), mode imposé :

    * ``ongoing`` (unité engagée) : vers les unités engagées, figurines socle à socle immobiles,
      chaque figurine engagée au départ l'est encore avec la même unité ;
    * ``engaging`` (un ennemi à 3") : vers l'unité ennemie la plus proche, l'unité doit finir engagée
      avec elle ;
    * ``objective`` (un objectif à 3") : chaque figurine finit à portée de l'objectif si possible,
      sinon plus près ; l'unité doit finir à portée.

    Si les conditions d'arrivée ne sont pas remplies, rien ne bouge. Retourne (mode, identifiants des
    unités ennemies nouvellement engagées) ; mode vide si pas de consolidation."""
    rules = state.rules
    dist_max = rules.consolidate_in
    saved = [(m, m.x, m.y) for m in unit.alive_models]

    def revert() -> None:
        for m, x, y in saved:
            m.move_to(x, y)

    engaged_with = state.enemies_in_engagement_range(unit)
    if engaged_with:
        before = _engagements(state, unit)
        _move_towards_units(state, unit, engaged_with, dist_max)
        if not _keeps_engagements(state, unit, before):
            revert()
            return "", []
        new = [e.id for e in state.enemies_in_engagement_range(unit) if e.id not in {x.id for x in engaged_with}]
        return "ongoing", new
    near = [e for e in state.enemies_of(unit.side) if unit.min_gap_to(e) <= rules.consolidate_in + 1e-9]
    if near:
        target = min(near, key=lambda e: unit.min_gap_to(e))
        _move_towards_units(state, unit, [target], dist_max, lock_contact=False)
        if not unit.in_engagement_range_of(target, rules):
            revert()
            return "", []
        return "engaging", [e.id for e in state.enemies_in_engagement_range(unit)]
    objs = [o for o in state.layout.objectives if state.unit_objective_distance(unit, o) <= rules.consolidate_in + 1e-9]
    if objs:
        obj = min(objs, key=lambda o: state.unit_objective_distance(unit, o))
        if state.unit_in_objective_range(unit, obj):
            pass
        for m in unit.alive_models:
            if state.model_in_objective_range(m, obj):
                continue
            blockers = [x.disk for u in state.on_table_units() for x in u.alive_models if x is not m]
            _step_to_objective(state, m, obj, dist_max, blockers)
            if not unit.coherency_ok(rules):
                x0, y0 = next((x, y) for mm, x, y in saved if mm is m)
                m.move_to(x0, y0)
        if not state.unit_in_objective_range(unit, obj):
            revert()
            return "", []
        return "objective", []
    return "", []


def _step_to_objective(state: GameState, model: Model, objective, max_distance: float, blockers: Sequence[Disk]) -> None:
    """Avance la figurine vers l'objectif (point le plus proche de sa zone), d'au plus ``max_distance``,
    jusqu'à être à portée, sans chevaucher ``blockers``."""
    goal = state.objective_anchor(objective, model.position)
    ux, uy = unit_vector(model.position, goal)
    if (ux, uy) == (0.0, 0.0):
        return
    d = model.disk
    total = min(max_distance, dist(model.position, goal))
    steps = max(1, int(math.ceil(total / 0.25)))
    best = None
    for k in range(1, steps + 1):
        t = total * k / steps
        cand = d.translated(ux * t, uy * t)
        if not state.on_board(cand) or any(disks_overlap(cand, b) for b in blockers):
            continue
        best = cand
        if state.disk_in_objective_range(cand, objective):
            break
    if best is not None:
        model.move_to(best.x, best.y)
