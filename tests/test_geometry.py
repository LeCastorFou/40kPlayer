"""Tests analytiques de la géométrie (aucune donnée externe)."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.engine.geometry import (  # noqa: E402
    Disk,
    Polygon,
    Terrain,
    disk_gap,
    disk_intersects_polygon,
    disk_polygon_distance,
    disks_overlap,
    is_fully_visible,
    is_visible,
    mm_to_in,
    move_is_legal,
    perimeter_points,
    point_in_polygon,
    segment_intersects_polygon,
    segments_intersect,
    swept_disk_hits_polygon,
    visibility,
    within,
    within_of_point,
)

R32 = mm_to_in(32) / 2  # rayon d'un socle de 32 mm, en pouces (≈ 0.63")


class DistanceTests(unittest.TestCase):
    def test_gap_is_base_to_base(self):
        a = Disk(0, 0, R32)
        b = Disk(3, 0, R32)
        self.assertAlmostEqual(disk_gap(a, b), 3 - 2 * R32)
        self.assertAlmostEqual(disk_gap(a, Disk(0, 0, 1)), 0.0)  # chevauchement → 0

    def test_within_uses_edges(self):
        a = Disk(0, 0, R32)
        b = Disk(2 + 2 * R32, 0, R32)  # exactement 2" socle à socle
        self.assertTrue(within(a, b, 2.0))
        self.assertFalse(within(a, b, 1.99))
        self.assertTrue(within(a, b, 2.01))

    def test_overlap(self):
        self.assertTrue(disks_overlap(Disk(0, 0, 1), Disk(1.5, 0, 1)))
        self.assertFalse(disks_overlap(Disk(0, 0, 1), Disk(2, 0, 1)))  # contact exact
        self.assertFalse(disks_overlap(Disk(0, 0, 1), Disk(2.5, 0, 1)))

    def test_within_of_point(self):
        obj = (10.0, 10.0)
        m = Disk(10 + 3 + R32, 10, R32)  # bord du socle à 3" du centre du pion
        self.assertTrue(within_of_point(m, obj, 3.0))
        self.assertFalse(within_of_point(m.translated(0.01, 0), obj, 3.0))


class SegmentPolygonTests(unittest.TestCase):
    def test_segments(self):
        self.assertTrue(segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)))
        self.assertFalse(segments_intersect((0, 0), (1, 1), (2, 2), (3, 3)))  # colinéaires disjoints
        self.assertTrue(segments_intersect((0, 0), (2, 0), (1, 0), (1, 5)))  # contact en T
        self.assertFalse(segments_intersect((0, 0), (1, 0), (0, 1), (1, 1)))  # parallèles

    def test_point_in_polygon(self):
        sq = Polygon.rect(0, 0, 4, 4)
        self.assertTrue(point_in_polygon((2, 2), sq))
        self.assertFalse(point_in_polygon((5, 2), sq))
        self.assertTrue(point_in_polygon((4, 2), sq))  # sur le bord
        concave = Polygon(((0, 0), (6, 0), (6, 6), (3, 6), (3, 3), (0, 3)))  # en L
        self.assertTrue(concave.contains((1, 1)))
        self.assertFalse(concave.contains((1, 5)))  # dans l'encoche
        self.assertTrue(concave.contains((5, 5)))

    def test_segment_polygon(self):
        sq = Polygon.rect(2, 2, 4, 4)
        self.assertTrue(segment_intersects_polygon((0, 3), (6, 3), sq))  # traverse
        self.assertFalse(segment_intersects_polygon((0, 0), (6, 1), sq))  # passe dessous
        self.assertTrue(segment_intersects_polygon((3, 3), (3, 3.5), sq))  # entièrement dedans
        self.assertFalse(segment_intersects_polygon((0, 5), (5, 5), sq))  # passe au-dessus (bbox croisée)

    def test_disk_polygon(self):
        sq = Polygon.rect(0, 0, 4, 4)
        self.assertAlmostEqual(disk_polygon_distance(Disk(6, 2, 1), sq), 1.0)
        self.assertEqual(disk_polygon_distance(Disk(2, 2, 0.5), sq), 0.0)  # centre dedans
        self.assertTrue(disk_intersects_polygon(Disk(4.5, 2, 0.6), sq))
        self.assertFalse(disk_intersects_polygon(Disk(4.5, 2, 0.4), sq))
        self.assertAlmostEqual(disk_polygon_distance(Disk(7, 7, 1), sq), math.sqrt(18) - 1)  # vers un coin

    def test_box_and_area(self):
        b = Polygon.box(5, 5, 4, 2, angle_deg=90)
        x0, y0, x1, y1 = b.bbox
        self.assertAlmostEqual(x1 - x0, 2)
        self.assertAlmostEqual(y1 - y0, 4)
        self.assertAlmostEqual(b.area, 8)
        self.assertEqual(len(perimeter_points(Disk(0, 0, 1), 8)), 8)


def wall(x0, y0, x1, y1, **kw):
    return Terrain(Polygon.rect(x0, y0, x1, y1), **kw)


class VisibilityTests(unittest.TestCase):
    def test_open_ground(self):
        a, b = Disk(0, 0, R32), Disk(20, 0, R32)
        self.assertTrue(is_visible(a, b, []))
        self.assertTrue(is_fully_visible(a, b, []))
        self.assertEqual(visibility(a, b, []), 1.0)

    def test_wall_blocks(self):
        a, b = Disk(0, 0, R32), Disk(20, 0, R32)
        big = [wall(9, -10, 11, 10, see_through_from_inside=False)]
        self.assertFalse(is_visible(a, b, big))
        self.assertEqual(visibility(a, b, big), 0.0)

    def test_partial_cover_is_visible_not_fully(self):
        a, b = Disk(0, 0, R32), Disk(20, 0, R32)
        # un mur qui masque la moitié inférieure de la cible
        half = [wall(9, -10, 11, 0.0, see_through_from_inside=False)]
        self.assertTrue(is_visible(a, b, half))
        self.assertFalse(is_fully_visible(a, b, half))
        frac = visibility(a, b, half)
        self.assertTrue(0.3 < frac < 0.8, frac)

    def test_edge_of_wall_true_los(self):
        # La cible dépasse à peine du coin du mur : la « vraie » ligne de vue passe.
        a = Disk(0, 0, R32)
        b = Disk(20, 0.5 + R32 * 0.5, R32)
        w = [wall(9, -10, 11, 0.5, see_through_from_inside=False)]
        self.assertTrue(is_visible(a, b, w, samples=24))
        self.assertFalse(is_fully_visible(a, b, w, samples=24))

    def test_ruin_transparent_from_inside(self):
        ruin = [wall(8, -3, 12, 3)]  # see_through_from_inside=True par défaut
        outside = Disk(0, 0, R32)
        inside = Disk(10, 0, R32)
        far = Disk(20, 0, R32)
        self.assertTrue(is_visible(outside, inside, ruin))  # on voit dans la ruine
        self.assertTrue(is_visible(inside, far, ruin))  # et depuis la ruine
        self.assertFalse(is_visible(outside, far, ruin))  # mais pas au travers

    def test_non_opaque_terrain(self):
        crates = [wall(9, -10, 11, 10, opaque=False)]
        self.assertTrue(is_fully_visible(Disk(0, 0, R32), Disk(20, 0, R32), crates))

    def test_symmetry(self):
        a, b = Disk(0, 0, R32), Disk(20, 4, R32)
        w = [wall(9, -10, 11, 2, see_through_from_inside=False)]
        self.assertEqual(is_visible(a, b, w), is_visible(b, a, w))


class MovementTests(unittest.TestCase):
    def test_distance_limit(self):
        m = Disk(0, 0, R32)
        self.assertTrue(move_is_legal(m, (6, 0), 6.0))
        self.assertFalse(move_is_legal(m, (6.01, 0), 6.0))
        self.assertTrue(move_is_legal(m, (3, 4), 5.0))  # 3-4-5

    def test_board_edges(self):
        m = Disk(1, 1, R32)
        self.assertTrue(move_is_legal(m, (R32, R32), 6.0, board=(44, 30)))  # collé au coin
        self.assertFalse(move_is_legal(m, (0.1, 1), 6.0, board=(44, 30)))  # socle hors table

    def test_other_bases(self):
        m = Disk(0, 0, R32)
        others = [Disk(3, 0, R32)]
        self.assertFalse(move_is_legal(m, (3 - R32, 0), 6.0, others))  # chevauche
        self.assertTrue(move_is_legal(m, (3 - 2 * R32, 0), 6.0, others))  # contact socle à socle : OK

    def test_impassable_terrain(self):
        m = Disk(0, 0, R32)
        blockwall = [wall(3, -1, 3.2, 1, impassable=True)]
        self.assertFalse(move_is_legal(m, (6, 0), 6.0, terrain=blockwall))  # traverse le mur
        self.assertTrue(move_is_legal(m, (0, 5), 6.0, terrain=blockwall))  # passe à côté
        self.assertTrue(swept_disk_hits_polygon(m, (6, 0), blockwall[0].footprint))
        ruin = [wall(3, -1, 3.2, 1)]  # franchissable
        self.assertTrue(move_is_legal(m, (6, 0), 6.0, terrain=ruin))


if __name__ == "__main__":
    unittest.main()
