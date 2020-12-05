"""Script to generate random 2d convex hull dataset. Also, a dataloader.
"""

import pickle
import random
from typing import List, NamedTuple

import numpy as np
from tqdm.auto import tqdm

from utils import compute_convex_hull, plot_points, plot_points_and_hull

N = 1_000_000
seq_len = [5, 50]
seed = 42
data_file = '/tmp/convex_hull.dat'


class InputRecord(NamedTuple):
  seq: List
  seq_mask: List
  output: List
  output_mask: List


class ConvexHullDataLoader:
  """A dataloader for convex hull dataset."""

  def __init__(self, random_seed: int = 42, data_filepath: str = '/tmp/convex_hull.dat'):
    with open(data_filepath, 'rb') as f:
      self.data = pickle.load(f)

    self.data = list(map(_encode_convex_hull, tqdm(self.data)))

    self.rng = random.Random(random_seed)

  def data_iter(self, batch_size: int, mode='train'):
    L = len(self.data) * 9 // 10
    data = self.data[:L] if mode == 'train' else self.data[L:]
    while True:
      self.rng.shuffle(data)
      for i in range(batch_size, len(data), batch_size):
        batch = data[i-batch_size: i]
        batch = zip(*batch)
        batch = [np.asarray(e) for e in batch]
        yield batch


def _encode_convex_hull(record):
  """encode convex hulls to network input format"""
  max_encode_len = max(seq_len)
  max_decode_len = max(seq_len) + 1 + 1
  total_len = max_encode_len + max_decode_len
  encoder_seq, hull = record
  encoder_seq_len = len(encoder_seq)

  # add new dimension for the [start] token
  encoder_seq = [(0., *e) for e in encoder_seq]

  # create decoder sequence
  decoder_seq = [encoder_seq[i] for i in hull]
  # insert [start] token
  decoder_seq = [(1.0, 0., 0.)] + decoder_seq
  decoder_seq = decoder_seq + [(0., 0., 0.)] * (max_decode_len - len(decoder_seq))

  # pad encoder seq
  pad_len = max_encode_len - encoder_seq_len
  encoder_seq = [[0., 0., 0.]] * pad_len + encoder_seq

  # create seq mask
  seq_mask = [False] * pad_len
  seq_mask = seq_mask + [True] * (total_len - len(seq_mask))

  # input sequence to the network = encoder inputs + [start] + decoder inputs
  input_seq = encoder_seq + decoder_seq

  # network prediction
  output = [pad_len + i for i in hull]
  # [end] token is at `max_encode_len` position.
  output = output + [max_encode_len]
  output_mask = [True] * len(output) + [False] * (max_decode_len - len(output))
  output = output + [0] * (max_decode_len - len(output))

  return InputRecord(input_seq, seq_mask, output, output_mask)


def generate_data_record(rng, l):
  seq = []
  for _ in range(l):
    x = rng.gauss(0., 1)
    y = rng.gauss(0., 1)
    seq.append((x, y))
  hull = compute_convex_hull(seq)
  return [seq, hull]


def _create_dataset(N, seq_len, seed=42):
  data = []
  rng = random.Random(seed)
  for _ in tqdm(range(N), desc='generating', ascii=True, ncols=80):
    l = rng.randint(*seq_len)
    data.append(generate_data_record(rng, l))

  return data


if __name__ == "__main__":
  data = _create_dataset(N, seq_len, seed)
  plot_points_and_hull(*data[-21], 'imgs/sample.png')
  with open(data_file, 'wb') as f:
    pickle.dump(data, f)
