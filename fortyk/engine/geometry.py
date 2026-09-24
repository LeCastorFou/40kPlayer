"""Géométrie 2D du toy model : figurines = disques (socles), décors = polygones.

Unité de longueur : le pouce (inch), comme dans les règles. Pur Python, sans
dépendance : le volume de calcul du toy model ne le justifie pas encore, et le
code reste simple à porter vers numpy / Rust si la recherche arborescente l'exige.

Conventions 40k reprises ici :

* les distances entre figurines se mesurent entre les points les plus proches
  des socles (:func:`disk_gap`), pas entre les centres ;
* la ligne de vue est « vraie » : il suffit qu'un segment quelconque entre un
  point du socle observateur et un point du socle observé ne soit bloqué par
  aucun décor opaque (:func:`visibility`) ; on échantillonne les périmètres ;
* un décor peut être « transparent depuis l'intérieur » (ruines : on voit
  dedans/dehors quand l'un des deux socles est dans l'empreinte).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    "EPS",
    "Point",
    "Disk",
    "Polygon",
    "Terrain",
    "dist",
    "disk_gap",
    "disks_overlap",
    "within",
    "within_of_point",
    "segments_intersect",
    "point_segment_distance",
    "point_in_polygon",
    "segment_intersects_polygon",
    "disk_polygon_distance",
    "disk_intersects_polygon",
    "perimeter_points",
    "core_points",
    "core_edges",
    "core_distance",
    "point_core_distance",
    "segment_core_distance",
    "segment_segment_distance",
    "extreme_points",
    "segment_blocked",
    "visibility",
    "is_visible",
    "is_fully_visible",
    "swept_disk_hits_polygon",
    "move_is_legal",
    "MM_PER_INCH",
    "mm_to_in",
]

MM_PER_INCH = 25.4
EPS = 1e-9
LOS_CLEARANCE_IN = 0.25  #: marge minimale d'une ligne de vue par rapport au bord d'un décor opaque
LOS_TRIM_IN = 1.0  #: la marge ne s'applique pas au premier / dernier pouce du trait (socle collé à un mur)

Point = Tuple[float, float]


def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH


# ------------------------------------------------------------------ primitives


class Disk(NamedTuple):
    """L'empreinte d'une figurine, en pouces : un « rectangle arrondi » centré en (x, y).

    Le cœur est un rectangle de demi-côtés ``hx`` (le long de l'orientation ``a``, en radians) et
    ``hy``, dilaté du rayon ``r`` :

    * socle rond : ``hx = hy = 0`` → disque de rayon ``r`` (le cas courant, chemins rapides) ;
    * socle ovale : ``hy = 0`` → « stade » (segment de longueur 2·hx dilaté de r), bonne approximation
      d'une ellipse ;
    * coque de véhicule sans socle (Rhino) : ``r = 0`` → rectangle 2·hx × 2·hy orienté.

    (NamedTuple : immuable, hachable, et bien plus rapide à créer qu'une dataclass gelée — le moteur
    en crée des centaines de milliers.)"""

    x: float
    y: float
    r: float
    hx: float = 0.0
    hy: float = 0.0
    a: float = 0.0

    @property
    def center(self) -> Point:
        return (self.x, self.y)

    @property
    def is_round(self) -> bool:
        return self.hx == 0.0 and self.hy == 0.0

    @property
    def extent(self) -> float:
        """Rayon du cercle circonscrit (distance max du centre à un point de l'empreinte)."""
        return self.r + math.hypot(self.hx, self.hy)

    def moved_to(self, x: float, y: float) -> "Disk":
        return self._replace(x=x, y=y)

    def translated(self, dx: float, dy: float) -> "Disk":
        return self._replace(x=self.x + dx, y=self.y + dy)

    def rotated_to(self, a: float) -> "Disk":
        return self._replace(a=a)


@dataclass(frozen=True)
class Polygon:
    """Polygone simple (non auto-intersectant), sommets dans l'ordre, fermé implicitement."""

    vertices: Tuple[Point, ...]

    def __post_init__(self):
        if len(self.vertices) < 3:
            raise ValueError("un polygone a au moins 3 sommets")

    @classmethod
    def rect(cls, x0: float, y0: float, x1: float, y1: float) -> "Polygon":
        """Rectangle aligné sur les axes, de (x0, y0) à (x1, y1)."""
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        return cls(((xa, ya), (xb, ya), (xb, yb), (xa, yb)))

    @classmethod
    def box(cls, cx: float, cy: float, w: float, h: float, angle_deg: float = 0.0) -> "Polygon":
        """Rectangle centré en (cx, cy), largeur w, hauteur h, tourné de ``angle_deg``."""
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        pts = []
        for sx, sy in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            dx, dy = sx * w, sy * h
            pts.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
        return cls(tuple(pts))

    @property
    def edges(self) -> Tuple[Tuple[Point, Point], ...]:
        v = self.vertices
        return tuple((v[i], v[(i + 1) % len(v)]) for i in range(len(v)))

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def centroid(self) -> Point:
        xs = [p[0] for p in self.vertices]
        ys = [p[1] for p in self.vertices]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @property
    def area(self) -> float:
        s = 0.0
        for (x0, y0), (x1, y1) in self.edges:
            s += x0 * y1 - x1 * y0
        return abs(s) / 2

    def contains(self, p: Point) -> bool:
        return point_in_polygon(p, self)

    def contains_disk(self, d: "Disk") -> bool:
        """Le socle est-il entièrement dans le polygone (supposé convexe pour les empreintes non rondes) ?"""
        for p in core_points(d):
            if not point_in_polygon(p, self):
                return False
            if d.r > 0 and min(point_segment_distance(p, e0, e1) for e0, e1 in self.edges) < d.r - EPS:
                return False
        return True

    def expanded(self, margin: float) -> "Polygon":
        """Polygone élargi de ``margin`` : exact pour un rectangle aligné (le cas des empreintes GW),
        approximé par la boîte englobante élargie sinon."""
        x0, y0, x1, y1 = self.bbox
        return Polygon.rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin)


@dataclass(frozen=True)
class Terrain:
    """Un élément de décor : une empreinte polygonale et ses propriétés de jeu.

    ``opaque``                  bloque les lignes de vue qui traversent l'empreinte.
    ``see_through_from_inside`` (ruines) l'empreinte ne bloque pas si l'observateur ou
                                la cible touche l'empreinte (un socle partiellement dedans compte).
    ``impassable``              les socles ne peuvent ni entrer ni traverser (murs pleins).
    ``clearance_in``            une ligne de vue doit passer à au moins cette distance du bord
                                (hors du premier et du dernier pouce du trait) : évite les lignes
                                « au rasoir » entre deux décors dont les bords sont alignés.
    """

    footprint: Polygon
    name: str = ""
    opaque: bool = True
    see_through_from_inside: bool = True
    impassable: bool = False
    clearance_in: float = LOS_CLEARANCE_IN

    def blocks_sight_between(self, a: Disk, b: Disk) -> bool:
        """Ce décor doit-il être considéré comme bloquant pour la paire (a, b) ?"""
        if not self.opaque:
            return False
        if self.see_through_from_inside and (disk_intersects_polygon(a, self.footprint) or disk_intersects_polygon(b, self.footprint)):
            return False  # « un orteil suffit » : un socle qui touche l'empreinte est dedans
        return True


# ------------------------------------------------------------------ distances


def dist(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# ------------------------------------------------------------------ empreintes non rondes


def core_points(d: Disk) -> List[Point]:
    """Sommets du cœur de l'empreinte : 1 (rond), 2 (ovale) ou 4 (rectangle), dans l'ordre."""
    if d.hx == 0.0 and d.hy == 0.0:
        return [(d.x, d.y)]
    c, s = math.cos(d.a), math.sin(d.a)
    if d.hy == 0.0:
        return [(d.x - d.hx * c, d.y - d.hx * s), (d.x + d.hx * c, d.y + d.hx * s)]
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = sx * d.hx, sy * d.hy
        pts.append((d.x + lx * c - ly * s, d.y + lx * s + ly * c))
    return pts


def core_edges(d: Disk) -> List[Tuple[Point, Point]]:
    pts = core_points(d)
    if len(pts) == 1:
        return [(pts[0], pts[0])]
    if len(pts) == 2:
        return [(pts[0], pts[1])]
    return [(pts[i], pts[(i + 1) % 4]) for i in range(4)]


def _to_local(p: Point, d: Disk) -> Point:
    c, s = math.cos(d.a), math.sin(d.a)
    dx, dy = p[0] - d.x, p[1] - d.y
    return (dx * c + dy * s, -dx * s + dy * c)


def point_core_distance(p: Point, d: Disk) -> float:
    """Distance du point au cœur de l'empreinte (0 s'il est dedans)."""
    if d.hx == 0.0 and d.hy == 0.0:
        return math.hypot(p[0] - d.x, p[1] - d.y)
    qx, qy = _to_local(p, d)
    return math.hypot(max(abs(qx) - d.hx, 0.0), max(abs(qy) - d.hy, 0.0))


def _segment_hits_box(q0: Point, q1: Point, hx: float, hy: float) -> bool:
    """Le segment (repère local) touche-t-il la boîte [-hx, hx] × [-hy, hy] ? (Liang–Barsky)"""
    lo, hi = 0.0, 1.0
    for p0, d, b in ((q0[0], q1[0] - q0[0], hx), (q0[1], q1[1] - q0[1], hy)):
        if abs(d) < 1e-15:
            if p0 < -b - EPS or p0 > b + EPS:
                return False
            continue
        t1, t2 = (-b - EPS - p0) / d, (b + EPS - p0) / d
        if t1 > t2:
            t1, t2 = t2, t1
        lo, hi = max(lo, t1), min(hi, t2)
        if lo > hi:
            return False
    return True


def segment_core_distance(p0: Point, p1: Point, d: Disk) -> float:
    """Distance du segment [p0, p1] au cœur de l'empreinte (0 s'ils se touchent)."""
    if d.hx == 0.0 and d.hy == 0.0:
        return point_segment_distance(d.center, p0, p1)
    q0, q1 = _to_local(p0, d), _to_local(p1, d)
    if _segment_hits_box(q0, q1, d.hx, d.hy):
        return 0.0
    best = min(math.hypot(max(abs(q[0]) - d.hx, 0.0), max(abs(q[1]) - d.hy, 0.0)) for q in (q0, q1))
    for cx, cy in ((-d.hx, -d.hy), (d.hx, -d.hy), (d.hx, d.hy), (-d.hx, d.hy)):
        best = min(best, point_segment_distance((cx, cy), q0, q1))
    return best


def core_distance(a: Disk, b: Disk) -> float:
    """Distance entre les cœurs de deux empreintes (0 s'ils se chevauchent)."""
    if a.hx == 0.0 and a.hy == 0.0:
        return point_core_distance(a.center, b)
    if b.hx == 0.0 and b.hy == 0.0:
        return point_core_distance(b.center, a)
    best = min(segment_core_distance(e0, e1, b) for e0, e1 in core_edges(a))
    if best > 0.0:
        best = min(best, min(segment_core_distance(e0, e1, a) for e0, e1 in core_edges(b)))
    return best


def disk_gap(a: Disk, b: Disk) -> float:
    """Distance socle-à-socle (0 si les socles se touchent ou se chevauchent)."""
    if a.hx == 0.0 and a.hy == 0.0 and b.hx == 0.0 and b.hy == 0.0:
        return max(0.0, math.hypot(a.x - b.x, a.y - b.y) - a.r - b.r)
    return max(0.0, core_distance(a, b) - a.r - b.r)


def disks_overlap(a: Disk, b: Disk) -> bool:
    """Chevauchement strict (deux socles qui se touchent exactement ne se chevauchent pas)."""
    if a.hx == 0.0 and a.hy == 0.0 and b.hx == 0.0 and b.hy == 0.0:
        return math.hypot(a.x - b.x, a.y - b.y) < a.r + b.r - EPS
    return core_distance(a, b) < a.r + b.r - EPS


def within(a: Disk, b: Disk, inches: float) -> bool:
    """« a est à moins de X" de b » au sens 40k : distance socle-à-socle ≤ X."""
    return disk_gap(a, b) <= inches + EPS


def within_of_point(a: Disk, p: Point, inches: float) -> bool:
    """« a est à moins de X" du point p » (pion d'objectif, centre de table…), mesuré depuis le socle."""
    return point_core_distance(p, a) - a.r <= inches + EPS


def extreme_points(d: Disk) -> List[Point]:
    """Points de l'empreinte les plus éloignés dans les 4 directions des axes, à partir de chaque
    sommet du cœur (tester qu'ils sont dans une zone rectiligne = l'empreinte y est entièrement)."""
    pts = []
    for cx, cy in core_points(d):
        if d.r > 0:
            pts.extend([(cx - d.r, cy), (cx + d.r, cy), (cx, cy - d.r), (cx, cy + d.r)])
        else:
            pts.append((cx, cy))
    return pts


# ------------------------------------------------------------------ segments


def _orient(p: Point, q: Point, r: Point) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """r est-il sur le segment [p, q] (r supposé colinéaire) ?"""
    return min(p[0], q[0]) - EPS <= r[0] <= max(p[0], q[0]) + EPS and min(p[1], q[1]) - EPS <= r[1] <= max(p[1], q[1]) + EPS


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Les segments [p1, p2] et [q1, q2] se touchent-ils (contact inclus) ?"""
    d1 = _orient(q1, q2, p1)
    d2 = _orient(q1, q2, p2)
    d3 = _orient(p1, p2, q1)
    d4 = _orient(p1, p2, q2)
    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and ((d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)):
        return True
    if abs(d1) <= EPS and _on_segment(q1, q2, p1):
        return True
    if abs(d2) <= EPS and _on_segment(q1, q2, p2):
        return True
    if abs(d3) <= EPS and _on_segment(p1, p2, q1):
        return True
    if abs(d4) <= EPS and _on_segment(p1, p2, q2):
        return True
    return False


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 <= EPS:
        return dist(p, a)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / l2
    t = max(0.0, min(1.0, t))
    return dist(p, (ax + t * dx, ay + t * dy))


# ------------------------------------------------------------------ polygones


def point_in_polygon(p: Point, poly: Polygon) -> bool:
    """Test pair-impair (ray casting) ; un point sur le bord compte comme dedans."""
    x, y = p
    inside = False
    for (x0, y0), (x1, y1) in poly.edges:
        if point_segment_distance(p, (x0, y0), (x1, y1)) <= EPS:
            return True
        if (y0 > y) != (y1 > y):
            xin = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xin:
                inside = not inside
    return inside


def segment_intersects_polygon(a: Point, b: Point, poly: Polygon) -> bool:
    """Le segment [a, b] touche-t-il l'intérieur ou le bord du polygone ?"""
    x0, y0, x1, y1 = poly.bbox
    if max(a[0], b[0]) < x0 - EPS or min(a[0], b[0]) > x1 + EPS or max(a[1], b[1]) < y0 - EPS or min(a[1], b[1]) > y1 + EPS:
        return False
    if point_in_polygon(a, poly) or point_in_polygon(b, poly):
        return True
    return any(segments_intersect(a, b, e0, e1) for e0, e1 in poly.edges)


def segment_segment_distance(p0: Point, p1: Point, q0: Point, q1: Point) -> float:
    if segments_intersect(p0, p1, q0, q1):
        return 0.0
    return min(point_segment_distance(p0, q0, q1), point_segment_distance(p1, q0, q1),
               point_segment_distance(q0, p0, p1), point_segment_distance(q1, p0, p1))


def disk_polygon_distance(d: Disk, poly: Polygon) -> float:
    """Distance entre le bord du socle et le polygone (0 si contact ou chevauchement)."""
    if d.hx == 0.0 and d.hy == 0.0:
        if point_in_polygon(d.center, poly):
            return 0.0
        edge_dist = min(point_segment_distance(d.center, e0, e1) for e0, e1 in poly.edges)
        return max(0.0, edge_dist - d.r)
    pts = core_points(d)
    if any(point_in_polygon(p, poly) for p in pts) or any(point_core_distance(v, d) <= EPS for v in poly.vertices):
        return 0.0
    best = min(segment_segment_distance(e0, e1, f0, f1) for e0, e1 in core_edges(d) for f0, f1 in poly.edges)
    return max(0.0, best - d.r)


def disk_intersects_polygon(d: Disk, poly: Polygon) -> bool:
    return disk_polygon_distance(d, poly) <= EPS


# ------------------------------------------------------------- lignes de vue


def perimeter_points(d: Disk, n: int) -> List[Point]:
    """``n`` points régulièrement répartis sur le bord de l'empreinte (le centre si n == 0).
    Socle rond : angles réguliers ; autres formes : abscisse curviligne régulière le long du bord
    (côtés droits puis arrondis), en partant du milieu du côté avant."""
    if n <= 0 or (d.r <= EPS and d.hx == 0.0 and d.hy == 0.0):
        return [d.center]
    if d.hx == 0.0 and d.hy == 0.0:
        return [(d.x + d.r * math.cos(2 * math.pi * k / n), d.y + d.r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    hx, hy, r = d.hx, d.hy, d.r
    # bord local parcouru dans le sens trigonométrique : côté +x, coin, côté +y, coin, côté -x, …
    pieces = []  # (longueur, fonction t∈[0,1] → point)
    corners = ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))
    starts = ((hx + r, -hy), (hx, hy + r), (-hx - r, hy), (-hx, -hy - r))
    ends = ((hx + r, hy), (-hx, hy + r), (-hx - r, -hy), (hx, -hy - r))
    for i in range(4):
        (sx, sy), (ex, ey) = starts[i], ends[i]
        length = math.hypot(ex - sx, ey - sy)
        pieces.append((length, lambda t, sx=sx, sy=sy, ex=ex, ey=ey: (sx + (ex - sx) * t, sy + (ey - sy) * t)))
        if r > 0:
            cx, cy = corners[i]
            a0 = math.pi / 2 * i
            pieces.append((math.pi / 2 * r, lambda t, cx=cx, cy=cy, a0=a0: (cx + r * math.cos(a0 + t * math.pi / 2), cy + r * math.sin(a0 + t * math.pi / 2))))
    total = sum(length for length, _ in pieces)
    c, s = math.cos(d.a), math.sin(d.a)
    out = []
    offset = pieces[0][0] / 2  # départ au milieu du côté avant (+x)
    for k in range(n):
        u = (offset + total * k / n) % total
        for length, f in pieces:
            if u <= length + 1e-12 and length > 0:
                lx, ly = f(u / length)
                break
            u -= length
        else:
            lx, ly = pieces[-1][1](1.0)
        out.append((d.x + lx * c - ly * s, d.y + lx * s + ly * c))
    return out


def segment_blocked(a: Point, b: Point, blockers: Iterable) -> bool:
    """Le trait [a, b] est-il arrêté par un des bloqueurs ?

    ``blockers`` : polygones, ou couples ``(polygone, polygone élargi)``. Avec un couple, le trait
    complet est testé contre le polygone exact, et sa partie centrale (raccourcie de LOS_TRIM_IN à
    chaque bout) contre le polygone élargi de la marge de sécurité.
    """
    for blk in blockers:
        if isinstance(blk, tuple):
            poly, grown = blk
        else:
            poly, grown = blk, None
        if segment_intersects_polygon(a, b, poly):
            return True
        if grown is not None:
            length = dist(a, b)
            if length > 2 * LOS_TRIM_IN:
                t0, t1 = LOS_TRIM_IN / length, 1 - LOS_TRIM_IN / length
                a2 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
                b2 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
                if segment_intersects_polygon(a2, b2, grown):
                    return True
    return False


def _blockers_for(observer: Disk, target: Disk, terrain: Sequence[Terrain]) -> list:
    out = []
    for t in terrain:
        if t.blocks_sight_between(observer, target):
            out.append((t.footprint, t.footprint.expanded(t.clearance_in)) if t.clearance_in > 0 else t.footprint)
    return out


def visibility(observer: Disk, target: Disk, terrain: Sequence[Terrain], samples: int = 12) -> float:
    """Fraction des points du périmètre de ``target`` visibles depuis au moins un point de ``observer``.

    0.0 : invisible ; 1.0 : pleinement visible ; entre les deux : partiellement visible.
    Les décors qui ne bloquent pas pour cette paire (ruine contenant l'un des socles,
    décor non opaque) sont écartés avant le test. Complexité O(samples² · arêtes).
    """
    blockers = _blockers_for(observer, target, terrain)
    tgt_pts = perimeter_points(target, samples)
    if not blockers:
        return 1.0
    obs_pts = [observer.center] + perimeter_points(observer, samples)
    seen = 0
    for tp in tgt_pts:
        if any(not segment_blocked(op, tp, blockers) for op in obs_pts):
            seen += 1
    return seen / len(tgt_pts)


def is_visible(observer: Disk, target: Disk, terrain: Sequence[Terrain], samples: int = 12) -> bool:
    """Vraie ligne de vue : au moins un segment socle→socle non bloqué."""
    blockers = _blockers_for(observer, target, terrain)
    if not blockers:
        return True
    obs_pts = [observer.center] + perimeter_points(observer, samples)
    tgt_pts = [target.center] + perimeter_points(target, samples)
    return any(not segment_blocked(op, tp, blockers) for tp in tgt_pts for op in obs_pts)


def is_fully_visible(observer: Disk, target: Disk, terrain: Sequence[Terrain], samples: int = 12) -> bool:
    """Tout le périmètre de la cible est visible (approximation de « fully visible »)."""
    return visibility(observer, target, terrain, samples) >= 1.0 - EPS


# ------------------------------------------------------------- déplacements


def swept_disk_hits_polygon(start: Disk, end: Point, poly: Polygon, steps: Optional[int] = None) -> bool:
    """Le socle, déplacé en ligne droite de ``start`` vers ``end``, touche-t-il le polygone ?

    Échantillonne le trajet avec un pas ≤ r/2 (au moins 2 positions), ce qui suffit
    pour des empreintes de décor bien plus grandes qu'un socle.
    """
    length = dist(start.center, end)
    if steps is None:
        steps = max(2, int(math.ceil(length / max(start.r / 2, 0.1))) + 1)
    for k in range(steps):
        t = k / (steps - 1)
        pos = start.moved_to(start.x + (end[0] - start.x) * t, start.y + (end[1] - start.y) * t)
        if disk_intersects_polygon(pos, poly):
            return True
    return False


def move_is_legal(
    start: Disk,
    end: Point,
    max_distance: float,
    other_bases: Iterable[Disk] = (),
    terrain: Sequence[Terrain] = (),
    board: Optional[Tuple[float, float]] = None,
) -> bool:
    """Un déplacement en ligne droite est-il légal ?

    * longueur ≤ ``max_distance`` ;
    * arrivée entièrement sur la table (``board`` = (largeur, hauteur)) ;
    * pas de chevauchement final avec un autre socle ;
    * pas de traversée ni d'arrêt dans un décor ``impassable``.

    Les décors franchissables (ruines sans murs, caisses) n'empêchent rien ici :
    c'est au moteur d'appliquer d'éventuels coûts de mouvement.
    """
    if dist(start.center, end) > max_distance + EPS:
        return False
    final = start.moved_to(*end)
    if board is not None:
        w, h = board
        if any(px < -EPS or py < -EPS or px > w + EPS or py > h + EPS for px, py in extreme_points(final)):
            return False
    if any(disks_overlap(final, o) for o in other_bases):
        return False
    for t in terrain:
        if t.impassable and swept_disk_hits_polygon(start, end, t.footprint):
            return False
    return True
