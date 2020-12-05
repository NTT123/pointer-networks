from typing import List, NamedTuple

import matplotlib.pyplot as plt


class Point(NamedTuple):
  x: float
  y: float


def plot_points(points, outfile=None):
  plt.figure(figsize=(3, 3))
  x, y, = zip(*points)
  plt.scatter(x, y, s=20, alpha=0.5, c='red')
  if outfile is not None:
    plt.savefig(outfile)
  else:
    plt.show()
  plt.close('all')


def plot_points_and_hull(points, hull, outfile=None):
  plt.figure(figsize=(3, 3))
  x, y = zip(*points)
  plt.scatter(x, y, s=20, alpha=0.5, c='red')
  xs = []
  ys = []
  for i in hull:
    xs.append(x[i])
    ys.append(y[i])
  plt.plot(xs, ys)
  if outfile is not None:
    plt.savefig(outfile)
  else:
    plt.show()
  plt.close('all')


def get_left_most_point(points: List[Point]) -> int:
  x, _ = zip(*points)
  return x.index(min(x))


def on_left(c, a, b):
  """
  if point C is on the left side of the road from A to B.
  ccw algorithm
  """
  x, y = zip(*[c, a, b])
  p = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0])
  return p > 0


def compute_convex_hull(points: List[Point]) -> List[int]:
  """Fift wrapping algorithm.
  https://en.wikipedia.org/wiki/Gift_wrapping_algorithm
  """
  c = get_left_most_point(points)
  point_on_hull = points[c]
  out = [c]
  start_point = point_on_hull
  while True:
    end_point = points[0]
    c = 0
    for j in range(len(points)):
      if end_point == point_on_hull or on_left(points[j], point_on_hull, end_point):
        end_point = points[j]
        c = j
    point_on_hull = end_point
    out.append(c)
    if end_point == start_point:
      break

  # return out # left most point first

  # minimum index first
  idx = out.index(min(out))
  return out[idx:-1] + out[:idx] + [out[idx]]
