"""Cartes de bataille : table, zones de déploiement, objectifs et décors.

Une carte est décrite dans un JSON (``data/layouts/*.json``) et chargée en
:class:`Layout`. Les décors sont des rectangles alignés sur les axes (les
empreintes GW le sont) convertis en :class:`~fortyk.engine.geometry.Terrain`.

Conventions V11 retenues pour la 2D :

* toute empreinte (ruine ou barricade, dense ou légère) bloque les lignes de vue qui la
  traversent, sauf pour un socle qui la touche (« un orteil suffit ») ; elle est franchissable
  et c'est une zone de terrain (couvert à l'infanterie qui la touche, aux monstres / véhicules
  entièrement dans une ruine ; Hidden pour l'infanterie qui la touche) ;
* la distinction dense (D) / légère (L) de la carte est conservée mais n'a pas encore d'effet
  dans le moteur (à préciser avec les règles V11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .geometry import EPS, Point, Polygon, Terrain, disk_intersects_polygon, disk_polygon_distance, extreme_points, perimeter_points, point_in_polygon
from .rules import DEFAULT_RULES, RulesConfig

__all__ = ["TerrainPiece", "ObjectivePoint", "Layout", "load_layout", "LAYOUTS_DIR"]

LAYOUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "layouts"


@dataclass(frozen=True)
class TerrainPiece:
    id: str
    kind: str  #: "ruin" (zone de terrain) ou "barricade"
    density: str  #: "dense" ou "light"
    rect: Tuple[float, float, float, float]  #: x0, y0, x1, y1 en pouces
    label: str = ""

    def __post_init__(self):
        if self.kind not in ("ruin", "barricade"):
            raise ValueError(f"{self.id}: kind inconnu {self.kind!r}")
        if self.density not in ("dense", "light"):
            raise ValueError(f"{self.id}: density inconnue {self.density!r}")
        x0, y0, x1, y1 = self.rect
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{self.id}: rectangle dégénéré {self.rect}")

    @property
    def polygon(self) -> Polygon:
        return Polygon.rect(*self.rect)

    @property
    def is_area(self) -> bool:
        """Zone de terrain au sens des règles : toute empreinte de décor (ruine ou barricade) donne le
        couvert à l'infanterie qui la touche et compte pour « dans un décor » (Hidden)."""
        return True

    @property
    def center(self) -> Point:
        x0, y0, x1, y1 = self.rect
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def to_terrain(self) -> Terrain:
        """Toute empreinte (dense ou légère, ruine ou barricade) bloque les lignes de vue qui la
        traversent, sauf pour un socle qui la touche (« un orteil suffit »). Validé avec Valentin :
        le prince derrière la barricade légère T8m n'est pas visible. La distinction dense / légère
        est conservée dans ``density`` pour les règles qui en dépendront."""
        return Terrain(
            footprint=self.polygon,
            name=self.id,
            opaque=True,
            see_through_from_inside=True,
            impassable=False,
            clearance_in=DEFAULT_RULES.los_clearance_in,
        )

    def mirrored(self, board: Tuple[float, float]) -> Tuple[float, float, float, float]:
        w, h = board
        x0, y0, x1, y1 = self.rect
        return (w - x1, h - y1, w - x0, h - y0)


@dataclass(frozen=True)
class ObjectivePoint:
    id: str
    kind: str  #: "home", "central" ou "expansion"
    x: float
    y: float
    owner: Optional[str] = None  #: "attacker" / "defender" pour un home objective
    estimated: bool = False  #: position lue sur une image, à confirmer
    terrain_id: Optional[str] = None  #: V11 (14.01) : empreinte de décor qui est l'objectif (désignée par la carte)
    rect: Optional[Tuple[float, float, float, float]] = None  #: rectangle de cette empreinte ; None = pion de 40 mm (3")

    @property
    def center(self) -> Point:
        return (self.x, self.y)

    @property
    def is_home(self) -> bool:
        return self.kind == "home"


@dataclass(frozen=True)
class Layout:
    name: str
    board: Tuple[float, float]
    deployment_zones: Dict[str, Polygon]
    objectives: Tuple[ObjectivePoint, ...]
    terrain: Tuple[TerrainPiece, ...]
    territory_line: Optional[Tuple[Point, Point]] = None
    source: str = ""

    # ----------------------------------------------------------- accès

    @property
    def width(self) -> float:
        return self.board[0]

    @property
    def height(self) -> float:
        return self.board[1]

    @property
    def center(self) -> Point:
        return (self.board[0] / 2, self.board[1] / 2)

    def terrain_objects(self) -> List[Terrain]:
        return [p.to_terrain() for p in self.terrain]

    def areas(self) -> List[TerrainPiece]:
        return [p for p in self.terrain if p.is_area]

    def objective(self, id_: str) -> ObjectivePoint:
        for o in self.objectives:
            if o.id == id_:
                return o
        raise KeyError(id_)

    def home_objective(self, side: str) -> Optional[ObjectivePoint]:
        for o in self.objectives:
            if o.is_home and o.owner == side:
                return o
        return None

    # Les empreintes sont des rectangles alignés sur les axes : les tests ci-dessous travaillent
    # directement sur ``piece.rect`` (arithmétique simple, bords inclus à EPS près), ce qui est bien
    # plus rapide que les polygones génériques de geometry.py et donne les mêmes réponses.

    def piece_at(self, p: Point) -> Optional[TerrainPiece]:
        """Zone de terrain contenant le point, s'il y en a une."""
        x, y = p
        for piece in self.terrain:
            x0, y0, x1, y1 = piece.rect
            if piece.is_area and x0 - EPS <= x <= x1 + EPS and y0 - EPS <= y <= y1 + EPS:
                return piece
        return None

    @staticmethod
    def _rect_contains_disk(rect, d) -> bool:
        x0, y0, x1, y1 = rect
        if len(d) < 6 or (d.hx == 0.0 and d.hy == 0.0):
            return x0 + d.r - EPS <= d.x <= x1 - d.r + EPS and y0 + d.r - EPS <= d.y <= y1 - d.r + EPS
        return all(x0 - EPS <= px <= x1 + EPS and y0 - EPS <= py <= y1 + EPS for px, py in extreme_points(d))

    @staticmethod
    def _rect_touches_disk(rect, d) -> bool:
        x0, y0, x1, y1 = rect
        if len(d) < 6 or (d.hx == 0.0 and d.hy == 0.0):
            dx = max(x0 - d.x, 0.0, d.x - x1)
            dy = max(y0 - d.y, 0.0, d.y - y1)
            lim = d.r + EPS
            return dx * dx + dy * dy <= lim * lim
        return disk_polygon_distance(d, Polygon.rect(x0, y0, x1, y1)) <= EPS

    def piece_containing_disks(self, disks) -> Optional[TerrainPiece]:
        """Zone de terrain contenant entièrement tous les socles donnés (« wholly within »)."""
        disks = list(disks)
        if not disks:
            return None
        for piece in self.terrain:
            if piece.is_area and all(self._rect_contains_disk(piece.rect, d) for d in disks):
                return piece
        return None

    def disk_wholly_in_area(self, d, samples: int = 24) -> bool:
        """Le socle est-il entièrement dans une zone de terrain, en comptant comme une seule zone
        des décors qui se touchent ou se chevauchent (ex. T5m L + T4m D du Layout A) ?

        Une empreinte suffit à contenir le socle → vrai ; sinon on vérifie que le centre et
        ``samples`` points du bord du socle sont chacun dans une zone de terrain."""
        areas = self.areas()
        if any(self._rect_contains_disk(a.rect, d) for a in areas):
            return True
        pts = [d.center] + perimeter_points(d, samples) + (extreme_points(d) if len(d) == 6 and not d.is_round else [])
        return all(any(a.rect[0] - EPS <= p[0] <= a.rect[2] + EPS and a.rect[1] - EPS <= p[1] <= a.rect[3] + EPS for a in areas) for p in pts)

    def disks_wholly_in_area(self, disks) -> bool:
        """Tous les socles donnés sont entièrement dans des zones de terrain (adjacentes ou non)."""
        disks = list(disks)
        return bool(disks) and all(self.disk_wholly_in_area(d) for d in disks)

    def piece_touching(self, d) -> Optional[TerrainPiece]:
        """Zone de terrain que le socle touche (chevauchement partiel suffisant : « un orteil suffit »)."""
        for piece in self.terrain:
            if piece.is_area and self._rect_touches_disk(piece.rect, d):
                return piece
        return None

    def disk_touches_dense_area(self, d) -> bool:
        """Le socle touche une zone de terrain contenant un décor dense (Hidden, V11 13.09)."""
        return any(p.is_area and p.density == "dense" and self._rect_touches_disk(p.rect, d) for p in self.terrain)

    def disk_touches_area(self, d) -> bool:
        return self.piece_touching(d) is not None

    def disks_touch_area(self, disks) -> bool:
        """Chaque socle touche une zone de terrain : c'est le test « dans un décor » de Hidden (V11,
        validé avec Valentin : pas besoin d'être entièrement dedans, un orteil suffit)."""
        disks = list(disks)
        return bool(disks) and all(self.disk_touches_area(d) for d in disks)

    def ruin_containing_disk(self, d) -> Optional[TerrainPiece]:
        """Ruine (pas barricade) contenant entièrement le socle — couvert des monstres et véhicules."""
        for piece in self.terrain:
            if piece.kind == "ruin" and self._rect_contains_disk(piece.rect, d):
                return piece
        return None

    def territory_of(self, p: Point) -> Optional[str]:
        """« attacker » ou « defender » selon le côté de la ligne de séparation (None si absente)."""
        if self.territory_line is None:
            return None
        (ax, ay), (bx, by) = self.territory_line
        side = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
        anchor = self.deployment_zones["attacker"].centroid
        anchor_side = (bx - ax) * (anchor[1] - ay) - (by - ay) * (anchor[0] - ax)
        if abs(side) <= EPS:
            return None
        return "attacker" if (side > 0) == (anchor_side > 0) else "defender"

    def in_deployment_zone(self, p: Point, side: str) -> bool:
        return point_in_polygon(p, self.deployment_zones[side])

    # ----------------------------------------------------------- contrôles

    def symmetry_report(self, tolerance: float = 0.3) -> List[str]:
        """Liste les décors sans image par rotation de 180° (les cartes GW sont symétriques)."""
        issues = []
        rects = [p.rect for p in self.terrain]
        for piece in self.terrain:
            m = piece.mirrored(self.board)
            if not any(all(abs(a - b) <= tolerance for a, b in zip(m, r)) for r in rects):
                issues.append(f"{piece.id} : pas de miroir pour {piece.rect}")
        return issues

    # ----------------------------------------------------------- rendu

    def render(self, path, scale: float = 16.0, rules: RulesConfig = DEFAULT_RULES, models=None, labels=None, margin: int = 22) -> None:
        """Image PNG de contrôle (nécessite Pillow).

        ``models`` : liste de (x, y, r, couleur) à dessiner ; ``labels`` : liste de (x, y, texte).
        Une marge graduée en pouces entoure la table pour lire les coordonnées.
        """
        from PIL import Image, ImageDraw  # import local : Pillow n'est pas une dépendance du moteur

        w, h = int(self.width * scale), int(self.height * scale)
        img = Image.new("RGB", (w + 2 * margin, h + 2 * margin), (250, 248, 244))
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([margin, margin, margin + w, margin + h], fill=(232, 228, 218))

        def P(x, y):
            return (margin + x * scale, margin + y * scale)

        for x in range(int(self.width) + 1):
            major = x % 4 == 0
            d.line([P(x, 0), P(x, self.height)], fill=(200, 196, 186) if major else (215, 211, 202), width=1)
            if major and margin:
                d.text((P(x, 0)[0] - 4, 3), str(x), fill=(60, 60, 60))
                d.text((P(x, self.height)[0] - 4, margin + h + 4), str(x), fill=(60, 60, 60))
        for y in range(int(self.height) + 1):
            major = y % 4 == 0
            d.line([P(0, y), P(self.width, y)], fill=(200, 196, 186) if major else (215, 211, 202), width=1)
            if major and margin:
                d.text((3, P(0, y)[1] - 5), str(y), fill=(60, 60, 60))
                d.text((margin + w + 4, P(0, y)[1] - 5), str(y), fill=(60, 60, 60))

        zone_colors = {"attacker": (150, 35, 30, 120), "defender": (55, 90, 125, 120)}
        for side, poly in self.deployment_zones.items():
            d.polygon([P(*v) for v in poly.vertices], fill=zone_colors.get(side, (120, 120, 120, 100)))

        if self.territory_line:
            (ax, ay), (bx, by) = self.territory_line
            d.line([P(ax, ay), P(bx, by)], fill=(40, 40, 40), width=2)

        for piece in self.terrain:
            x0, y0, x1, y1 = piece.rect
            fill = (190, 190, 192, 235) if piece.is_area else (170, 170, 172, 235)
            edge = (45, 110, 100) if piece.density == "dense" else (190, 140, 30)
            d.rectangle([P(x0, y0), P(x1, y1)], fill=fill, outline=edge, width=4)
            d.text((P(x0, y0)[0] + 5, P(x0, y0)[1] + 4), f"{piece.id} {piece.density[0].upper()}", fill=(30, 30, 30))

        r_marker = rules.objective_marker_radius_in
        r_control = rules.objective_control_radius_in
        for o in self.objectives:
            color = {"home": (200, 60, 60), "central": (40, 130, 110), "expansion": (60, 120, 190)}[o.kind]
            if o.rect is not None:  # V11 : l'empreinte de décor désignée est l'objectif
                x0, y0, x1, y1 = o.rect
                (px0, py0), (px1, py1) = P(x0, y0), P(x1, y1)
                d.rectangle([px0 + 4, py0 + 4, px1 - 4, py1 - 4], outline=color + (230,), width=4)
                d.ellipse([P(o.x - r_marker * 0.6, o.y - r_marker * 0.6), P(o.x + r_marker * 0.6, o.y + r_marker * 0.6)], fill=color + (230,))
                d.text((P(o.x, o.y)[0] + r_marker * 0.6 * scale + 3, P(o.x, o.y)[1] - 6), f"{o.id} ({o.terrain_id})", fill=(20, 20, 20))
                continue
            d.ellipse([P(o.x - r_control, o.y - r_control), P(o.x + r_control, o.y + r_control)], outline=color + (200,), width=2)
            d.ellipse([P(o.x - r_marker, o.y - r_marker), P(o.x + r_marker, o.y + r_marker)], fill=color + (230,))
            d.text((P(o.x, o.y)[0] + r_marker * scale + 3, P(o.x, o.y)[1] - 6), o.id + (" ?" if o.estimated else ""), fill=(20, 20, 20))

        for m in models or ():
            x, y, r, col = m
            d.ellipse([P(x - r, y - r), P(x + r, y + r)], fill=col, outline=(0, 0, 0))
        for x, y, text in labels or ():
            px, py = P(x, y)
            d.rectangle([px - 2, py - 7, px + 7 * len(text), py + 6], fill=(255, 255, 255, 200))
            d.text((px, py - 6), text, fill=(0, 0, 0))

        d.rectangle([margin, margin, margin + w - 1, margin + h - 1], outline=(0, 0, 0), width=3)
        img.save(path)


def _polygon_from_json(points) -> Polygon:
    return Polygon(tuple((float(x), float(y)) for x, y in points))


def load_layout(path) -> Layout:
    """Charge un fichier JSON de carte. ``path`` peut être un nom court (« layout_a »)."""
    p = Path(path)
    if not p.suffix:
        p = LAYOUTS_DIR / f"{p.name}.json"
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    board = (float(data["board"][0]), float(data["board"][1]))
    zones = {side: _polygon_from_json(pts) for side, pts in data["deployment_zones"].items()}
    terrain = tuple(
        TerrainPiece(
            id=t["id"],
            kind=t["kind"],
            density=t["density"],
            rect=tuple(float(v) for v in t["rect"]),
            label=t.get("label", ""),
        )
        for t in data["terrain"]
    )
    by_id = {t.id: t for t in terrain}

    def objective_piece(o):
        # la carte désigne l'empreinte (« terrain ») ; à défaut, celle qui contient le point (14.01) ;
        # « terrain »: null = pion de 40 mm hors décor
        if "terrain" in o:
            return by_id[o["terrain"]] if o["terrain"] else None
        x, y = float(o["x"]), float(o["y"])
        return next((t for t in terrain if t.is_area and t.rect[0] <= x <= t.rect[2] and t.rect[1] <= y <= t.rect[3]), None)

    objectives = []
    for o in data["objectives"]:
        piece = objective_piece(o)
        objectives.append(ObjectivePoint(
            id=o["id"],
            kind=o["kind"],
            x=float(o["x"]),
            y=float(o["y"]),
            owner=o.get("owner"),
            estimated=bool(o.get("estimated", False)),
            terrain_id=piece.id if piece else None,
            rect=piece.rect if piece else None,
        ))
    objectives = tuple(objectives)
    line = data.get("territory_line")
    territory_line = (tuple(map(float, line[0])), tuple(map(float, line[1]))) if line else None
    return Layout(
        name=data["name"],
        board=board,
        deployment_zones=zones,
        objectives=objectives,
        terrain=terrain,
        territory_line=territory_line,
        source=data.get("source", ""),
    )
