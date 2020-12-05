from utils import compute_convex_hull


def test_convex_hull():
  p = [(0, 0), (2, 0), (1, 0), (3, 5), (-3, 5)]
  o = compute_convex_hull(p)
  assert o == [0, 4, 3, 1, 0]
