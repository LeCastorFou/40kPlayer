"""Empreintes non rondes (ovales, coques de véhicule) : référence Python contre shapely (si présent),
et version numpy contre la référence (distances, trajectoires, déploiement, lignes de vue)."""

import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.engine import fastgeo  # noqa: E402
from fortyk.engine.fastgeo import LineOfSight, core_distances, shape_arrays, swept_core_distances  # noqa: E402
from fortyk.engine.geometry import (  # noqa: E402
    Disk, Polygon, core_distance, disk_gap, disk_polygon_distance, extreme_points, is_visible, point_in_polygon, visibility,
)
from fortyk.engine.layout import load_layout  # noqa: E402

try:
    import shapely.geometry as sg
    from shapely import affinity
    HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    HAS_SHAPELY = False


def random_shape(rng, x0=0.0, x1=44.0, y0=0.0, y1=60.0):
    kind = rng.choice(["round", "oval", "hull", "rrect"])
    x, y, a = rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(-math.pi, math.pi)
    if kind == "round":
        return Disk(x, y, rng.choice([0.5, 0.63, 1.18, 1.97]))
    if kind == "oval":
        return Disk(x, y, rng.uniform(0.6, 1.8), rng.uniform(0.2, 1.2), 0.0, a)
    if kind == "hull":
        return Disk(x, y, 0.0, rng.uniform(1.5, 3.5), rng.uniform(1.0, 2.3), a)
    return Disk(x, y, rng.uniform(0.1, 0.6), rng.uniform(0.5, 2.5), rng.uniform(0.3, 1.5), a)


def to_shapely(d):
    if d.hx == 0 and d.hy == 0:
        return sg.Point(d.x, d.y).buffer(d.r, 256)
    if d.hy == 0:
        core = sg.LineString([(-d.hx, 0), (d.hx, 0)])
    else:
        core = sg.box(-d.hx, -d.hy, d.hx, d.hy)
    g = core.buffer(d.r, 256) if d.r > 0 else core
    g = affinity.rotate(g, d.a, origin=(0, 0), use_radians=True)
    return affinity.translate(g, d.x, d.y)


class ReferenceGeometryTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SHAPELY, "shapely absent")
    def test_gap_matches_shapely(self):
        rng = random.Random(3)
        for _ in range(1500):
            a = random_shape(rng, 10, 20, 10, 20)
            b = random_shape(rng, 10, 20, 10, 20)
            expect = to_shapely(a).distance(to_shapely(b))
            self.assertAlmostEqual(disk_gap(a, b), expect, delta=2e-3, msg=(a, b))

    @unittest.skipUnless(HAS_SHAPELY, "shapely absent")
    def test_polygon_distance_matches_shapely(self):
        rng = random.Random(4)
        for _ in range(800):
            d = random_shape(rng, 5, 25, 5, 25)
            x0, y0 = rng.uniform(5, 20), rng.uniform(5, 20)
            poly = Polygon.rect(x0, y0, x0 + rng.uniform(1, 8), y0 + rng.uniform(1, 8))
            expect = to_shapely(d).distance(sg.Polygon(poly.vertices))
            self.assertAlmostEqual(disk_polygon_distance(d, poly), expect, delta=2e-3, msg=(d, poly))

    def test_extreme_points_bound_the_shape(self):
        d = Disk(10, 10, 0.0, 3.0, 1.5, math.pi / 2)  # coque 6×3 tournée d'un quart de tour
        xs = [p[0] for p in extreme_points(d)]
        ys = [p[1] for p in extreme_points(d)]
        self.assertAlmostEqual(min(xs), 8.5)
        self.assertAlmostEqual(max(ys), 13.0)


class NumpyShapeEquivalenceTests(unittest.TestCase):
    def test_core_distances_match_reference(self):
        rng = random.Random(11)
        A = [random_shape(rng, 10, 18, 10, 18) for _ in range(40)]
        B = [random_shape(rng, 10, 18, 10, 18) for _ in range(40)]
        sa, sb = shape_arrays(A), shape_arrays(B)
        got = core_distances(sa[:, None, :], sb[None, :, :])
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                self.assertAlmostEqual(got[i, j], core_distance(a, b), delta=1e-6, msg=(a, b))

    def test_swept_distance_is_min_over_path(self):
        rng = random.Random(12)
        for _ in range(150):
            a = random_shape(rng, 10, 18, 10, 18)
            b = random_shape(rng, 10, 18, 10, 18)
            v = (rng.uniform(-6, 6), rng.uniform(-6, 6))
            got = swept_core_distances(shape_arrays([a])[None, :, None, :], np.array(v)[None, None, None, :], shape_arrays([b])[None, None, :, :])[0, 0, 0]
            steps = 200
            ref = min(core_distance(a.translated(v[0] * k / steps, v[1] * k / steps), b) for k in range(steps + 1))
            step_len = math.hypot(*v) / steps
            self.assertLessEqual(got, ref + 1e-6, (a, b, v))
            self.assertGreaterEqual(got, ref - step_len - 1e-6, (a, b, v))

    def test_translations_legal_general_matches_bruteforce(self):
        rng = random.Random(13)
        board = (44.0, 60.0)
        er = 2.0
        for trial in range(30):
            mover = random_shape(rng, 15, 29, 20, 40)
            friends = [random_shape(rng, 8, 36, 12, 48) for _ in range(3)]
            enemies = [random_shape(rng, 8, 36, 12, 48) for _ in range(3)]
            vecs = np.array([(rng.uniform(-8, 8), rng.uniform(-8, 8)) for _ in range(20)])
            kind = rng.choice(["normal", "advance", "fall_back", "desperate", "charge"])
            got = fastgeo.translations_legal(
                np.array([[mover.x, mover.y]]), np.array([mover.r]), vecs, kind, 20.0, board,
                np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2)), np.zeros(0), er,
                shapes=(shape_arrays([mover]), shape_arrays(friends), shape_arrays(enemies)))
            for c, v in enumerate(vecs):
                end = mover.translated(*v)
                ok = all(-1e-9 <= px <= board[0] + 1e-9 and -1e-9 <= py <= board[1] + 1e-9 for px, py in extreme_points(end))
                ok &= not any(core_distance(end, o) < end.r + o.r - 1e-9 for o in friends + enemies)
                steps = 80
                path = [mover.translated(v[0] * k / steps, v[1] * k / steps) for k in range(steps + 1)]
                # V11 : amis traversables ; socles ennemis non traversables (sauf desperate escape) ; fin désengagée
                if kind in ("normal", "advance", "fall_back"):
                    ok &= not any(core_distance(p, e) < p.r + e.r - 1e-9 for p in path for e in enemies)
                if kind in ("normal", "advance", "fall_back", "desperate"):
                    ok &= not any(disk_gap(end, e) <= er + 1e-9 for e in enemies)
                if bool(got[c]) != ok:
                    # désaccord toléré seulement à la frontière (échantillonnage du trajet de référence)
                    margin = math.hypot(*v) / steps + 1e-3
                    near = [abs(core_distance(p, e) - p.r - e.r) < margin for p in path for e in enemies]
                    self.assertTrue(any(near), (trial, kind, mover, v, bool(got[c]), ok))

    def test_deployment_grid_general_matches_reference(self):
        rng = random.Random(14)
        zone = Polygon.rect(0, 0, 44, 12)
        board = (44.0, 60.0)
        shapes = shape_arrays([Disk(0, 0, 0.0, 2.4, 1.5, math.pi / 2)])
        offsets = np.zeros((1, 2))
        others = [random_shape(rng, 0, 44, 0, 14) for _ in range(6)]
        centers = np.array([(rng.uniform(0, 44), rng.uniform(0, 14)) for _ in range(500)])
        got = fastgeo.deployment_grid_legal(centers, offsets, np.array([0.0]), zone.vertices, board,
                                            np.zeros((0, 2)), np.zeros(0), shapes=shapes, others_shapes=shape_arrays(others))
        for g, (cx, cy) in enumerate(centers):
            d = Disk(cx, cy, 0.0, 2.4, 1.5, math.pi / 2)
            ok = all(point_in_polygon(p, zone) for p in extreme_points(d))
            ok &= all(0 <= px <= 44 and 0 <= py <= 60 for px, py in extreme_points(d))
            ok &= all(disk_gap(d, o) > 0.0 for o in others)
            self.assertEqual(bool(got[g]), ok, (cx, cy))

    def test_deployment_grid_round_formation_among_hulls(self):
        rng = random.Random(16)
        zone = Polygon.rect(0, 0, 44, 12)
        offsets = np.array([(-1.4, 0.0), (0.0, 0.0), (1.4, 0.0)])
        radii = np.array([0.63, 0.63, 0.63])
        shapes = np.array([(ox, oy, r, 0.0, 0.0, 0.0) for (ox, oy), r in zip(offsets, radii)])
        others = [random_shape(rng, 0, 44, 0, 14) for _ in range(8)]
        centers = np.array([(rng.uniform(0, 44), rng.uniform(0, 14)) for _ in range(400)])
        got = fastgeo.deployment_grid_legal(centers, offsets, radii, zone.vertices, (44.0, 60.0), np.zeros((0, 2)), np.zeros(0),
                                            shapes=shapes, others_shapes=shape_arrays(others))
        for g, (cx, cy) in enumerate(centers):
            ds = [Disk(cx + ox, cy + oy, 0.63) for ox, oy in offsets]
            ok = all(point_in_polygon(p, zone) for d in ds for p in extreme_points(d))
            ok &= all(disk_gap(d, o) > 0.0 for d in ds for o in others)
            self.assertEqual(bool(got[g]), ok, (cx, cy))

    def test_line_of_sight_with_hulls_matches_reference(self):
        terrain = load_layout("layout_a").terrain_objects()
        los = LineOfSight(terrain)
        rng = random.Random(15)
        for _ in range(250):
            a = random_shape(rng, 2, 42, 2, 58)
            b = random_shape(rng, 2, 42, 2, 58)
            self.assertEqual(los.visible(a, b), is_visible(a, b, terrain), (a, b))
            self.assertAlmostEqual(los.fraction(a, b), visibility(a, b, terrain), places=9, msg=(a, b))


if __name__ == "__main__":
    unittest.main()
