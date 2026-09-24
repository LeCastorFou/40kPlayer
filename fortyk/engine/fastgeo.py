"""Géométrie vectorisée (numpy) pour les calculs chauds du moteur.

Deux familles de calculs dominent le temps d'une partie : les lignes de vue (des centaines de
traits socle → socle contre les décors) et la légalité des mouvements candidats (chaque figurine,
chaque translation, contre chaque ennemi et chaque ami). Ce module les fait par lots avec numpy ;
:mod:`fortyk.engine.geometry` reste l'implémentation de référence en Python pur, et les tests
vérifient que les deux donnent les mêmes réponses.

* :class:`LineOfSight` — index des décors (rectangles alignés sur les axes) avec cache : pour une
  paire de socles, visible (au moins un trait libre) et fraction du périmètre de la cible visible
  (« pleinement visible » = 1). Même échantillonnage que la référence : centre + 12 points du
  périmètre de chaque côté, marge de dégagement de 0,25" hors du premier et du dernier pouce, une
  empreinte ne bloque pas un socle qui la touche. Le cache est partagé par toutes les parties qui
  utilisent les mêmes décors (clones de la recherche arborescente comprises).
* :func:`translations_legal` — pour une unité et C translations, lesquelles sont légales (table,
  chevauchements, socles ennemis traversés, fin hors portée d'engagement) ; les trajectoires sont
  testées de façon exacte (distance d'un segment à un point) et non par échantillonnage.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import EPS, LOS_TRIM_IN, Disk, Polygon, Terrain, disk_polygon_distance as _ref_disk_polygon_distance
from .geometry import is_visible as _ref_is_visible, perimeter_points as _ref_perimeter, visibility as _ref_visibility

__all__ = ["LineOfSight", "line_of_sight_for", "translations_legal", "points_in_polygon", "deployment_grid_legal", "PERIMETER_SAMPLES",
           "shape_arrays", "core_distances", "swept_core_distances", "touches_any", "point_box_distances"]

PERIMETER_SAMPLES = 12
_ANG = 2 * math.pi * np.arange(PERIMETER_SAMPLES) / PERIMETER_SAMPLES
# point 0 = centre, puis le périmètre (même ordre que geometry.perimeter_points)
_UNIT_PTS = np.concatenate([np.zeros((1, 2)), np.stack([np.cos(_ANG), np.sin(_ANG)], axis=1)])


def _is_axis_rect(poly) -> bool:
    x0, y0, x1, y1 = poly.bbox
    return len(poly.vertices) == 4 and abs(poly.area - (x1 - x0) * (y1 - y0)) <= 1e-9


def _seg_hits_rects(p0: np.ndarray, p1: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Segments [p0, p1] (…, 2) contre rectangles (R, 4) = x0, y0, x1, y1 → touche (…, R).
    Liang–Barsky, bords inclus (tolérance EPS). Une composante de direction nulle est remplacée par
    une valeur infime : les paramètres d'entrée / sortie deviennent ±inf, ce qui classe correctement
    le segment (dedans ou dehors de la bande) sans branche."""
    d = p1 - p0
    dx = d[..., 0, None]
    dy = d[..., 1, None]
    dx = np.where(dx == 0.0, 1e-300, dx)
    dy = np.where(dy == 0.0, 1e-300, dy)
    px, py = p0[..., 0, None], p0[..., 1, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ix = 1.0 / dx
        iy = 1.0 / dy
        tx_a = (rects[:, 0] - EPS - px) * ix
        tx_b = (rects[:, 2] + EPS - px) * ix
        ty_a = (rects[:, 1] - EPS - py) * iy
        ty_b = (rects[:, 3] + EPS - py) * iy
    lo = np.maximum(np.maximum(np.minimum(tx_a, tx_b), np.minimum(ty_a, ty_b)), 0.0)
    hi = np.minimum(np.minimum(np.maximum(tx_a, tx_b), np.maximum(ty_a, ty_b)), 1.0)
    return lo <= hi


def points_in_polygon(pts: np.ndarray, vertices) -> np.ndarray:
    """Points (N, 2) dans un polygone (pair-impair), bord inclus à EPS près — version vectorisée de
    :func:`fortyk.engine.geometry.point_in_polygon`."""
    v = np.asarray(vertices, dtype=float)
    a, b = v, np.roll(v, -1, axis=0)  # arêtes (E, 2)
    x, y = pts[:, 0, None], pts[:, 1, None]
    on_edge = (_seg_point_dist(a[None], b[None], pts[:, None, :]) <= EPS).any(axis=1)
    cond = (a[None, :, 1] > y) != (b[None, :, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xin = a[None, :, 0] + (y - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / (b[None, :, 1] - a[None, :, 1])
    crossings = (cond & (x < xin)).sum(axis=1)
    return on_edge | (crossings % 2 == 1)


def _disks_touch_rects(xy: np.ndarray, r: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Socles (N, 2), rayons (N,) contre rectangles (R, 4) → le socle touche l'empreinte (N, R)."""
    cx, cy = xy[:, 0, None], xy[:, 1, None]
    ddx = np.maximum(np.maximum(rects[:, 0] - cx, 0.0), cx - rects[:, 2])
    ddy = np.maximum(np.maximum(rects[:, 1] - cy, 0.0), cy - rects[:, 3])
    return np.hypot(ddx, ddy) <= r[:, None] + EPS


def _sample_points(disks: Sequence[Disk]) -> np.ndarray:
    """Centre + 12 points du bord de chaque empreinte (même échantillonnage que geometry.py)."""
    if all(d.hx == 0.0 and d.hy == 0.0 for d in disks):
        xy = np.array([(d.x, d.y) for d in disks], dtype=float)
        r = np.array([d.r for d in disks], dtype=float)
        return xy[:, None, :] + r[:, None, None] * _UNIT_PTS[None]
    out = np.empty((len(disks), PERIMETER_SAMPLES + 1, 2))
    for i, d in enumerate(disks):
        if d.hx == 0.0 and d.hy == 0.0:
            out[i] = np.array([d.x, d.y]) + d.r * _UNIT_PTS
        else:
            out[i, 0] = (d.x, d.y)
            out[i, 1:] = np.array(_ref_perimeter(d, PERIMETER_SAMPLES))
    return out


def _touch_matrix(disks: Sequence[Disk], rects: np.ndarray) -> np.ndarray:
    """(N, R) : l'empreinte n touche le rectangle de décor r (« un orteil suffit »)."""
    xy = np.array([(d.x, d.y) for d in disks], dtype=float)
    r = np.array([d.r for d in disks], dtype=float)
    out = _disks_touch_rects(xy, r, rects)
    for i, d in enumerate(disks):
        if not (d.hx == 0.0 and d.hy == 0.0):
            out[i] = [_ref_disk_polygon_distance(d, Polygon.rect(*rc)) <= EPS for rc in rects]
    return out


class LineOfSight:
    """Lignes de vue entre socles pour un jeu de décors donné, avec cache."""

    MAX_CACHE = 200_000  #: ~50 Mo au plus ; vidé d'un coup quand il est plein

    def __init__(self, terrain: Sequence[Terrain]):
        self.terrain = list(terrain)
        self.vectorized = all(_is_axis_rect(t.footprint) for t in self.terrain)
        blockers = [t for t in self.terrain if t.opaque]
        self.rects = np.array([t.footprint.bbox for t in blockers], dtype=float).reshape(-1, 4)
        self.grown = np.array([[x0 - t.clearance_in, y0 - t.clearance_in, x1 + t.clearance_in, y1 + t.clearance_in]
                               for t, (x0, y0, x1, y1) in zip(blockers, (b.footprint.bbox for b in blockers))], dtype=float).reshape(-1, 4)
        self.has_clearance = np.array([t.clearance_in > 0 for t in blockers], dtype=bool)
        self.see_through = np.array([t.see_through_from_inside for t in blockers], dtype=bool)
        self.cache: Dict[Tuple[float, ...], Tuple[bool, float]] = {}
        self.hits = 0
        self.misses = 0

    # ----------------------------------------------------------- API

    def visible(self, a: Disk, b: Disk) -> bool:
        """Au moins un trait libre entre les deux socles (vraie ligne de vue)."""
        return self._lookup(a, b)[0]

    def fraction(self, a: Disk, b: Disk) -> float:
        """Fraction des points du périmètre de ``b`` visibles depuis ``a``."""
        return self._lookup(a, b)[1]

    def fully_visible(self, a: Disk, b: Disk) -> bool:
        return self._lookup(a, b)[1] >= 1.0 - EPS

    def prefetch(self, observers: Sequence[Disk], targets: Sequence[Disk]) -> None:
        """Calcule d'un coup toutes les paires (observateur, cible) absentes du cache."""
        if not self.vectorized or not observers or not targets:
            return
        todo_a = [a for a in observers if any(tuple(a) + tuple(b) not in self.cache for b in targets)]
        if not todo_a:
            return
        self._compute_block(todo_a, list(targets))

    # ----------------------------------------------------------- interne

    def _lookup(self, a: Disk, b: Disk) -> Tuple[bool, float]:
        key = tuple(a) + tuple(b)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        if self.vectorized:
            self._compute_block([a], [b])
            return self.cache[key]
        val = (_ref_is_visible(a, b, self.terrain), _ref_visibility(a, b, self.terrain))
        self._store(key, val)
        return val

    def _store(self, key, val) -> None:
        if len(self.cache) >= self.MAX_CACHE:
            self.cache.clear()
        self.cache[key] = val

    def _compute_block(self, A: List[Disk], B: List[Disk]) -> None:
        na, nb = len(A), len(B)
        axy = np.array([(d.x, d.y) for d in A], dtype=float)
        ar = np.array([d.extent for d in A], dtype=float)
        bxy = np.array([(d.x, d.y) for d in B], dtype=float)
        br = np.array([d.extent for d in B], dtype=float)
        # seuls les décors qui coupent la boîte englobante de tous les socles peuvent bloquer
        lo = np.minimum((axy - ar[:, None]).min(0), (bxy - br[:, None]).min(0))
        hi = np.maximum((axy + ar[:, None]).max(0), (bxy + br[:, None]).max(0))
        keep = ~((self.grown[:, 2] < lo[0]) | (self.grown[:, 0] > hi[0]) | (self.grown[:, 3] < lo[1]) | (self.grown[:, 1] > hi[1]))
        if not keep.any():
            for a in A:
                for b in B:
                    self._store(tuple(a) + tuple(b), (True, 1.0))
            return
        rects, grown = self.rects[keep], self.grown[keep]
        has_clearance, see_through = self.has_clearance[keep], self.see_through[keep]
        # empreintes qui bloquent pour chaque paire : opaques, sauf si l'un des socles la touche (ruines)
        ta = _touch_matrix(A, rects) & see_through  # (na, R)
        tb = _touch_matrix(B, rects) & see_through  # (nb, R)
        blocks = ~(ta[:, None, :] | tb[None, :, :])  # (na, nb, R)
        pa = _sample_points(A)  # (na, 13, 2) : centre + 12 points du bord
        pb = _sample_points(B)
        p0 = np.broadcast_to(pa[:, None, :, None, :], (na, nb, 13, 13, 2))
        p1 = np.broadcast_to(pb[None, :, None, :, :], (na, nb, 13, 13, 2))
        hit = _seg_hits_rects(p0, p1, rects)  # (na, nb, 13, 13, R)
        # marge de dégagement : partie centrale du trait contre l'empreinte élargie
        d = p1 - p0
        length = np.hypot(d[..., 0], d[..., 1])
        long_enough = length > 2 * LOS_TRIM_IN
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = np.where(long_enough, LOS_TRIM_IN / length, 0.0)[..., None]
        q0 = p0 + d * t0
        q1 = p0 + d * (1 - t0)
        hit_grown = _seg_hits_rects(q0, q1, grown) & long_enough[..., None] & has_clearance
        blocked = ((hit | hit_grown) & blocks[:, :, None, None, :]).any(axis=-1)  # (na, nb, 13, 13)
        free = ~blocked
        vis = free.any(axis=(2, 3))
        frac = free[:, :, :, 1:].any(axis=2).mean(axis=-1)
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                self._store(tuple(a) + tuple(b), (bool(vis[i, j]), float(frac[i, j])))


_LOS_BY_TERRAIN: Dict[tuple, LineOfSight] = {}


def line_of_sight_for(terrain: Sequence[Terrain]) -> LineOfSight:
    """Index de lignes de vue partagé par toutes les parties sur les mêmes décors."""
    key = tuple((t.footprint.vertices, t.opaque, t.see_through_from_inside, t.clearance_in) for t in terrain)
    los = _LOS_BY_TERRAIN.get(key)
    if los is None:
        los = LineOfSight(terrain)
        _LOS_BY_TERRAIN[key] = los
    return los


# ------------------------------------------------------------------ mouvements


def _seg_point_dist(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distance des segments [a, b] (…, 2) aux points p (…, 2) (diffusion numpy)."""
    ab = b - a
    ap = p - a
    l2 = (ab * ab).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(l2 > EPS, (ap * ab).sum(-1) / l2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[..., None]
    q = p - proj
    return np.hypot(q[..., 0], q[..., 1])


# ------------------------------------------------------------------ empreintes non rondes (vectorisé)

_CORNER_SIGNS = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])


def shape_arrays(disks: Sequence[Disk]) -> np.ndarray:
    """Tableau (N, 6) : x, y, r, hx, hy, angle."""
    return np.array([tuple(d) if len(d) == 6 else (d[0], d[1], d[2], 0.0, 0.0, 0.0) for d in disks], dtype=float).reshape(-1, 6)


def _corners(shapes: np.ndarray, centers: Optional[np.ndarray] = None) -> np.ndarray:
    """Sommets du cœur (…, 4, 2) ; ``centers`` (…, 2) remplace les centres (positions d'arrivée)."""
    cx = shapes[..., 0] if centers is None else centers[..., 0]
    cy = shapes[..., 1] if centers is None else centers[..., 1]
    hx, hy, a = shapes[..., 3], shapes[..., 4], shapes[..., 5]
    ca, sa = np.cos(a), np.sin(a)
    lx = _CORNER_SIGNS[:, 0] * hx[..., None]
    ly = _CORNER_SIGNS[:, 1] * hy[..., None]
    x = cx[..., None] + lx * ca[..., None] - ly * sa[..., None]
    y = cy[..., None] + lx * sa[..., None] + ly * ca[..., None]
    return np.stack([x, y], axis=-1)


def _segbox(p0: np.ndarray, p1: np.ndarray, cx, cy, hx, hy, a) -> np.ndarray:
    """Distance des segments [p0, p1] (…, 2) aux rectangles orientés (centre, demi-côtés, angle),
    tout en diffusion numpy ; 0 si le segment touche le rectangle (Liang–Barsky en repère local)."""
    ca, sa = np.cos(a), np.sin(a)
    def local(p):
        dx, dy = p[..., 0] - cx, p[..., 1] - cy
        return dx * ca + dy * sa, -dx * sa + dy * ca
    x0, y0 = local(p0)
    x1, y1 = local(p1)
    dxs, dys = x1 - x0, y1 - y0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ix = 1.0 / np.where(dxs == 0.0, 1e-300, dxs)
        iy = 1.0 / np.where(dys == 0.0, 1e-300, dys)
        ta, tb = (-hx - EPS - x0) * ix, (hx + EPS - x0) * ix
        tc, td = (-hy - EPS - y0) * iy, (hy + EPS - y0) * iy
    lo = np.maximum(np.maximum(np.minimum(ta, tb), np.minimum(tc, td)), 0.0)
    hi = np.minimum(np.minimum(np.maximum(ta, tb), np.maximum(tc, td)), 1.0)
    hit = lo <= hi
    def pbox(x, y):
        return np.hypot(np.maximum(np.abs(x) - hx, 0.0), np.maximum(np.abs(y) - hy, 0.0))
    best = np.minimum(pbox(x0, y0), pbox(x1, y1))
    l2 = dxs * dxs + dys * dys
    for sx, sy in _CORNER_SIGNS:
        qx, qy = sx * hx, sy * hy
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.clip(np.where(l2 > EPS, ((qx - x0) * dxs + (qy - y0) * dys) / np.where(l2 > EPS, l2, 1.0), 0.0), 0.0, 1.0)
        best = np.minimum(best, np.hypot(qx - (x0 + dxs * t), qy - (y0 + dys * t)))
    return np.where(hit, 0.0, best)


def _box(shapes: np.ndarray, centers: Optional[np.ndarray] = None):
    cx = shapes[..., 0] if centers is None else centers[..., 0]
    cy = shapes[..., 1] if centers is None else centers[..., 1]
    return cx, cy, shapes[..., 3], shapes[..., 4], shapes[..., 5]


def core_distances(A: np.ndarray, B: np.ndarray, a_centers: Optional[np.ndarray] = None) -> np.ndarray:
    """Distances entre cœurs, A (…, 6) contre B (…, 6) en diffusion : min des distances de chaque
    arête de l'un au rectangle de l'autre (les empreintes rondes ont des arêtes de longueur nulle)."""
    ca = _corners(A, a_centers)  # (…, 4, 2)
    cb = _corners(B)
    ea0, ea1 = ca, np.roll(ca, -1, axis=-2)
    eb0, eb1 = cb, np.roll(cb, -1, axis=-2)
    bx = tuple(v[..., None] for v in _box(B))
    ax = tuple(v[..., None] for v in _box(A, a_centers))
    d1 = _segbox(ea0, ea1, *bx).min(axis=-1)
    d2 = _segbox(eb0, eb1, *ax).min(axis=-1)
    return np.minimum(d1, d2)


def swept_core_distances(A: np.ndarray, vecs: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance minimale entre les cœurs de A translaté de 0 à ``vecs`` et B immobile : chaque sommet
    de A balaie un segment contre B, et chaque sommet de B balaie le segment opposé contre A (au premier
    contact, un sommet de l'un touche le bord de l'autre). Formes : A (1, n, 1, 6), vecs (C, 1, 1, 2),
    B (1, 1, m, 6) → (C, n, m)."""
    ca = _corners(A)[..., :, :]  # (1, n, 1, 4, 2)
    cb = _corners(B)  # (1, 1, m, 4, 2)
    v = vecs[..., None, :]  # (C, 1, 1, 1, 2)
    bx = tuple(x[..., None] for x in _box(B))
    ax = tuple(x[..., None] for x in _box(A))
    d1 = _segbox(ca, ca + v, *bx).min(axis=-1)
    d2 = _segbox(cb, cb - v, *ax).min(axis=-1)
    return np.minimum(d1, d2)


def _round_rows(a: np.ndarray) -> np.ndarray:
    return (a[:, 3] == 0.0) & (a[:, 4] == 0.0) if len(a) else np.zeros(0, dtype=bool)


def _all_round(a: np.ndarray) -> bool:
    return bool(_round_rows(a).all())


def _point_box_dist(p: np.ndarray, cx, cy, hx, hy, a) -> np.ndarray:
    """Distance des points p (…, 2) aux rectangles orientés (0 à l'intérieur)."""
    ca, sa = np.cos(a), np.sin(a)
    dx, dy = p[..., 0] - cx, p[..., 1] - cy
    lx, ly = dx * ca + dy * sa, -dx * sa + dy * ca
    return np.hypot(np.maximum(np.abs(lx) - hx, 0.0), np.maximum(np.abs(ly) - hy, 0.0))


def _pair_distances(movers: np.ndarray, vecs: np.ndarray, others: np.ndarray, swept: bool):
    """Distances cœur à cœur (C, n, m) entre l'unité translatée de ``vecs`` et ``others`` : à l'arrivée,
    et (``swept``) minimum sur le trajet. Chemins courts quand l'un des deux côtés est rond (un point ou
    un segment contre un rectangle au lieu de 4 × 4 arêtes)."""
    C, n, m = len(vecs), len(movers), len(others)
    V = vecs[:, None, None, :]  # (C, 1, 1, 2)
    start = movers[None, :, None, :2]  # (1, n, 1, 2)
    end = start + V  # (C, n, 1, 2)
    if _all_round(movers):
        bx = _box(others[None, None, :, :])
        d_end = _point_box_dist(end, *bx)
        d_path = _segbox(np.broadcast_to(start, end.shape), end, *bx) if swept else None
        return d_end, d_path
    if _all_round(others):
        oc = others[None, None, :, :2]  # (1, 1, m, 2)
        A = movers[None, :, None, :]
        ax_end = _box(A, np.broadcast_to(end, (C, n, 1, 2)))
        d_end = _point_box_dist(oc, *ax_end)
        d_path = _segbox(np.broadcast_to(oc, (C, n, m, 2)), oc - V, *_box(A)) if swept else None
        return d_end, d_path
    A = movers[None, :, None, :]
    B = others[None, None, :, :]
    end_centers = np.broadcast_to(end, (C, n, m, 2))
    d_end = core_distances(np.broadcast_to(A, (C, n, m, 6)), np.broadcast_to(B, (C, n, m, 6)), a_centers=end_centers)
    d_path = swept_core_distances(A, V, B) if swept else None
    return d_end, d_path


#: types de mouvement dont le trajet ne peut pas traverser un socle ennemi (03.01) ; le Desperate Escape
#: (09.07) le peut
PATH_BLOCKED_KINDS = ("normal", "advance", "fall_back")
#: types de mouvement qui doivent finir désengagés (09.05-09.07)
END_UNENGAGED_KINDS = ("normal", "advance", "fall_back", "desperate")


def _general_translations_legal(movers: np.ndarray, vecs: np.ndarray, kind: str, board, friends: np.ndarray, enemies: np.ndarray, er: float,
                                enemy_blocks: Optional[np.ndarray] = None) -> np.ndarray:
    C = len(vecs)
    ok = np.ones(C, dtype=bool)
    end_centers = movers[None, :, :2] + vecs[:, None, :]  # (C, n, 2)
    # table : sommets du cœur ± rayon dans les 4 directions
    corners = _corners(movers[None, :, :].repeat(C, axis=0), end_centers)  # (C, n, 4, 2)
    r = movers[None, :, None, 2]
    w, h = board
    ok &= ((corners[..., 0] - r >= -1e-9) & (corners[..., 1] - r >= -1e-9) & (corners[..., 0] + r <= w + 1e-9) & (corners[..., 1] + r <= h + 1e-9)).all(axis=(1, 2))
    if enemy_blocks is None:
        enemy_blocks = np.ones(len(enemies), dtype=bool)
    for others, is_enemy in ((friends, False), (enemies, True)):
        if not len(others):
            continue
        rows = _round_rows(others)
        for mask in (rows, ~rows):  # paires rondes / non rondes traitées séparément (chemins courts)
            sub = others[mask]
            if not len(sub):
                continue
            rsum = movers[None, :, None, 2] + sub[None, None, :, 2]
            blocks = enemy_blocks[mask] if is_enemy else None
            swept = is_enemy and kind in PATH_BLOCKED_KINDS and bool(blocks.any())
            d_end, d_path = _pair_distances(movers, vecs, sub, swept)
            ok &= ~(d_end < rsum - EPS).any(axis=(1, 2))  # pas de chevauchement à l'arrivée
            if not is_enemy:
                continue  # 03.01 : on traverse les figurines amies
            if swept:
                ok &= ~((d_path < rsum - EPS) & blocks[None, None, :]).any(axis=(1, 2))
            if kind in END_UNENGAGED_KINDS:
                ok &= ~(d_end - rsum <= er + EPS).any(axis=(1, 2))
    return ok


def translations_legal(
    models_xy: np.ndarray,
    models_r: np.ndarray,
    vecs: np.ndarray,
    kind: str,
    max_distance: float,
    board: Tuple[float, float],
    friends_xy: np.ndarray,
    friends_r: np.ndarray,
    enemies_xy: np.ndarray,
    enemies_r: np.ndarray,
    engagement_range: float,
    shapes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    enemy_blocks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Légalité de C translations (``vecs`` (C, 2)) de toute une unité (n figurines), règles V11.

    Toujours : longueur ≤ ``max_distance``, arrivée sur la table, pas de chevauchement à l'arrivée.
    Les figurines amies se traversent (03.01). ``normal`` / ``advance`` / ``fall_back`` : le trajet ne
    traverse pas de socle ennemi — sauf ceux que ``enemy_blocks`` (masque aligné sur les ennemis)
    déclare franchissables (M/V qui passent sur l'infanterie, 17.01) ; ``desperate`` : on traverse les
    ennemis. Tous ces mouvements doivent finir hors de portée d'engagement ; passer à portée pendant
    le trajet est permis. Translation nulle : toujours légale."""
    vecs = np.asarray(vecs, dtype=float).reshape(-1, 2)
    length = np.hypot(vecs[:, 0], vecs[:, 1])
    ok = length <= max_distance + 1e-9
    zero = length <= 1e-9
    if len(models_xy) == 0:
        return ok | zero
    n_enemies = len(shapes[2]) if shapes is not None else len(enemies_xy)
    blocks_all = np.ones(n_enemies, dtype=bool) if enemy_blocks is None else np.asarray(enemy_blocks, dtype=bool)
    # seuls les socles assez proches de la zone balayée par l'unité peuvent gêner
    if shapes is not None:
        ms, fs, es = shapes  # (n, 6), (f, 6), (e, 6) : x, y, r, hx, hy, angle
        ext = lambda a: a[:, 2] + np.hypot(a[:, 3], a[:, 4])  # noqa: E731
        reach = float(length.max()) + float(ext(ms).max()) + engagement_range + 2.0
        lo = ms[:, :2].min(0) - reach
        hi = ms[:, :2].max(0) + reach

        def keep_s(a):
            if not len(a):
                return np.zeros(0, dtype=bool)
            e = ext(a)
            return (a[:, 0] >= lo[0] - e) & (a[:, 0] <= hi[0] + e) & (a[:, 1] >= lo[1] - e) & (a[:, 1] <= hi[1] + e)

        kf, ke = keep_s(fs), keep_s(es)
        fs, es, blocks = fs[kf], es[ke], blocks_all[ke]
        if not _all_round(ms):
            return (ok & _general_translations_legal(ms, vecs, kind, board, fs, es, engagement_range, blocks)) | zero
        # unité de socles ronds : calcul rapide contre les socles ronds, général contre les autres
        fr, er_ = _round_rows(fs), _round_rows(es)
        fast = translations_legal(ms[:, :2], ms[:, 2], vecs, kind, max_distance, board, fs[fr, :2], fs[fr, 2], es[er_, :2], es[er_, 2], engagement_range,
                                  enemy_blocks=blocks[er_])
        if (~fr).any() or (~er_).any():
            fast &= _general_translations_legal(ms, vecs, kind, board, fs[~fr], es[~er_], engagement_range, blocks[~er_]) | zero
        return fast
    reach = float(length.max()) + float(models_r.max()) + engagement_range + 2.0
    lo = models_xy.min(0) - reach
    hi = models_xy.max(0) + reach

    def keep(xy, rr):
        if not len(xy):
            return np.zeros(0, dtype=bool)
        return (xy[:, 0] >= lo[0] - rr) & (xy[:, 0] <= hi[0] + rr) & (xy[:, 1] >= lo[1] - rr) & (xy[:, 1] <= hi[1] + rr)

    kf = keep(friends_xy, friends_r)
    ke = keep(enemies_xy, enemies_r)
    friends_xy, friends_r = friends_xy[kf], friends_r[kf]
    enemies_xy, enemies_r, blocks = enemies_xy[ke], enemies_r[ke], blocks_all[ke]
    start = models_xy[None, :, :]  # (1, n, 2)
    end = start + vecs[:, None, :]  # (C, n, 2)
    r = models_r[None, :]
    w, h = board
    ok &= ((end[..., 0] - r >= -1e-9) & (end[..., 1] - r >= -1e-9) & (end[..., 0] + r <= w + 1e-9) & (end[..., 1] + r <= h + 1e-9)).all(axis=1)
    if len(friends_xy):  # arrivée seulement : on traverse les amis (03.01)
        fxy = friends_xy[None, None, :, :]
        lim = r[..., None] + friends_r[None, None, :] - EPS  # chevauchement strict
        ok &= ~(_dist2(end[:, :, None, :], fxy) < np.square(lim) * (lim > 0)).any(axis=(1, 2))
    if len(enemies_xy):
        exy = enemies_xy[None, None, :, :]
        er = enemies_r[None, None, :]
        lim = r[..., None] + er - EPS
        d2_end = _dist2(end[:, :, None, :], exy)
        ok &= ~(d2_end < np.square(lim) * (lim > 0)).any(axis=(1, 2))
        if kind in PATH_BLOCKED_KINDS and blocks.any():
            lim2 = np.square(lim) * (lim > 0)
            cross = _seg_point_dist2(start[:, :, None, :], end[:, :, None, :], exy) < lim2
            already = _dist2(start[:, :, None, :], exy) < lim2  # socles déjà imbriqués au départ : on peut s'en dégager
            ok &= ~(cross & ~already & blocks[None, None, :]).any(axis=(1, 2))
        if kind in END_UNENGAGED_KINDS:
            lim_er = np.square(r[..., None] + er + engagement_range + EPS)
            ok &= ~(d2_end <= lim_er).any(axis=(1, 2))
    return ok | zero


def _dist2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]


def _seg_point_dist2(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Carré de la distance des segments [a, b] aux points p (diffusion numpy)."""
    abx, aby = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    apx, apy = p[..., 0] - a[..., 0], p[..., 1] - a[..., 1]
    l2 = abx * abx + aby * aby
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(l2 > EPS, (apx * abx + apy * aby) / np.where(l2 > EPS, l2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx = apx - abx * t
    qy = apy - aby * t
    return qx * qx + qy * qy


def deployment_grid_legal(
    centers: np.ndarray,
    offsets: np.ndarray,
    radii: np.ndarray,
    zone_vertices,
    board: Tuple[float, float],
    others_xy: np.ndarray,
    others_r: np.ndarray,
    shapes: Optional[np.ndarray] = None,
    others_shapes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pour G centres de formation (G, 2) : la formation (décalages (n, 2), rayons (n,)) tient-elle
    entièrement dans la zone (4 points cardinaux de chaque socle), sur la table, sans toucher les
    figurines déjà posées ? Version vectorisée de ``Engine.deployment_ok``."""
    if shapes is not None:
        others_shapes = np.zeros((0, 6)) if others_shapes is None else others_shapes
        if not _all_round(shapes):
            return _general_deployment_legal(centers, offsets, shapes, zone_vertices, board, others_shapes)
        # formation de socles ronds : calcul rapide, plus le contact avec les empreintes non rondes
        rr = _round_rows(others_shapes)
        ok = deployment_grid_legal(centers, offsets, shapes[:, 2], zone_vertices, board, others_shapes[rr, :2], others_shapes[rr, 2])
        if (~rr).any():
            ok &= ~_touches_any(centers, offsets, shapes, others_shapes[~rr])
        return ok
    pos = centers[:, None, :] + offsets[None, :, :]  # (G, n, 2)
    r = radii[None, :]
    w, h = board
    ok = ((pos[..., 0] - r >= -1e-9) & (pos[..., 1] - r >= -1e-9) & (pos[..., 0] + r <= w + 1e-9) & (pos[..., 1] + r <= h + 1e-9)).all(axis=1)
    card = np.array([(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)])
    corners = pos[:, :, None, :] + radii[None, :, None, None] * card[None, None, :, :]  # (G, n, 4, 2)
    inside = points_in_polygon(corners.reshape(-1, 2), zone_vertices).reshape(corners.shape[:3])
    ok &= inside.all(axis=(1, 2))
    if len(others_xy):
        d = np.hypot(*(pos[:, :, None, :] - others_xy[None, None, :, :]).transpose(3, 0, 1, 2))
        ok &= ~(d - r[..., None] - others_r[None, None, :] <= 0.0).any(axis=(1, 2))
    return ok


def _general_deployment_legal(centers, offsets, shapes, zone_vertices, board, others) -> np.ndarray:
    """Déploiement avec empreintes non rondes : sommets du cœur (± rayon) dans la zone et sur la table,
    aucun contact avec les figurines déjà posées."""
    G, n = len(centers), len(offsets)
    pos = centers[:, None, :] + offsets[None, :, :]  # (G, n, 2)
    corners = _corners(np.broadcast_to(shapes[None], (G, n, 6)), pos)  # (G, n, 4, 2)
    r = shapes[None, :, None, 2]
    card = np.array([(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)])
    pts = corners[:, :, :, None, :] + r[..., None, None] * card  # (G, n, 4, 4, 2)
    w, h = board
    ok = ((pts[..., 0] >= -1e-9) & (pts[..., 1] >= -1e-9) & (pts[..., 0] <= w + 1e-9) & (pts[..., 1] <= h + 1e-9)).all(axis=(1, 2, 3))
    inside = points_in_polygon(pts.reshape(-1, 2), zone_vertices).reshape(pts.shape[:4])
    ok &= inside.all(axis=(1, 2, 3))
    if others is not None and len(others):
        ok &= ~_touches_any(centers, offsets, shapes, others)
    return ok


def _touches_any(centers, offsets, shapes, others) -> np.ndarray:
    """(G,) : une figurine de la formation posée en ``centers + offsets`` touche-t-elle une empreinte
    de ``others`` ? (les formations clairement éloignées sont écartées sans calcul exact)"""
    G, n, m = len(centers), len(offsets), len(others)
    pos = centers[:, None, :] + offsets[None, :, :]  # (G, n, 2)
    ext_a = shapes[:, 2] + np.hypot(shapes[:, 3], shapes[:, 4])
    ext_b = others[:, 2] + np.hypot(others[:, 3], others[:, 4])
    far = np.hypot(pos[:, :, None, 0] - others[None, None, :, 0], pos[:, :, None, 1] - others[None, None, :, 1]) > ext_a[None, :, None] + ext_b[None, None, :] + EPS
    out = np.zeros(G, dtype=bool)
    rows = np.flatnonzero(~far.all(axis=(1, 2)))
    if not len(rows):
        return out
    k = len(rows)
    A = np.broadcast_to(shapes[None, :, None, :], (k, n, m, 6))
    B = np.broadcast_to(others[None, None, :, :], (k, n, m, 6))
    d = core_distances(A, B, a_centers=np.broadcast_to(pos[rows, :, None, :], (k, n, m, 2)))
    rsum = shapes[None, :, None, 2] + others[None, None, :, 2]
    out[rows] = (d - rsum <= 0.0).any(axis=(1, 2))
    return out


def touches_any(centers: np.ndarray, offsets: np.ndarray, shapes: np.ndarray, others: np.ndarray, grow: float = 0.0) -> np.ndarray:
    """(G,) : la formation (``shapes`` (n, 6) décalées de ``offsets``) posée en chaque centre touche-t-elle
    une empreinte de ``others`` (m, 6) élargie de ``grow`` (portée d'engagement…) ?"""
    if not len(others):
        return np.zeros(len(centers), dtype=bool)
    if grow:
        others = others.copy()
        others[:, 2] += grow
    return _touches_any(centers, offsets, shapes, others)


def point_box_distances(points: np.ndarray, shape) -> np.ndarray:
    """Distance des points (…, 2) au cœur d'une empreinte (x, y, r, hx, hy, a) (sans son rayon)."""
    x, y, _r, hx, hy, a = shape
    return _point_box_dist(points, x, y, hx, hy, a)
