from data import ConvexHullDataLoader


def test_data_loader():
  data_loader = ConvexHullDataLoader()
  it = data_loader.data_iter(32)
  batch = next(it)
  assert batch[0].shape == (32, 102, 3)
  assert batch[1].shape == (32, 102)
  assert batch[2].shape == (32, 52)
  assert batch[3].shape == (32, 52)
