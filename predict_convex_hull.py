import pickle
import random
from argparse import ArgumentParser

import haiku as hk
import jax
import jax.numpy as jnp
import math
from data import generate_data_record, plot_points_and_hull
from run_training import create_network


@hk.transform_with_state
def generate(seq, args):
  net = create_network(args)
  init_state = net.lstm.initial_state(1)
  seq_ = jnp.asarray([(0., *e) for e in seq])[None]
  seq_ = jnp.swapaxes(seq_, 0, 1)
  encoder_hx, state = hk.dynamic_unroll(net.lstm.core, seq_, init_state)
  encoder_hx = jnp.concatenate([encoder_hx, init_state.hidden[None]], axis=0)
  hull = []
  #  = out[-1]
  encoder_value = net.enc_att_fc(encoder_hx)

  with open('/tmp/encoded_value.pk', 'wb') as f:
    pickle.dump(jax.device_get(encoder_value), f)

  inp = jnp.asarray([[1., 0., 0.]])  # [start] token

  qq = []

  for i in range(len(seq) + 1):
    hidden, state = net.lstm.core(inp, state)
    decoder_query = net.dec_att_fc(hidden)
    qq.append(decoder_query)
    logits = net.energy_fc(jnp.tanh(encoder_value + decoder_query[None]))
    if False:
      # import pdb; pdb.set_trace()
      logits = encoder_value * decoder_query[None]
      logits = logits / math.sqrt(logits.shape[-1])
      logits = jnp.sum(logits, axis=-1)
    idx = jnp.argmax(logits, axis=0).item()
    if idx == len(seq):
      break
    hull.append(idx)
    inp = seq_[idx]

  qq = jnp.concatenate(qq, axis=0)
  # import pdb; pdb.set_trace()

  with open('/tmp/decoder_query.pk', 'wb') as f:
    pickle.dump(jax.device_get(qq), f)


  plot_points_and_hull(seq, hull, 'imgs/prediction.png')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-c', '--checkpoint-filepath', default=None, type=str)
  parser.add_argument('-d', '--rnn-hidden-size', default=256, type=int)
  parser.add_argument('-s', '--random-seed', default=1111, type=int)
  parser.add_argument('-l', '--num-vertex', default=20, type=int)
  args = parser.parse_args()
  print(args)

  with open(args.checkpoint_filepath, 'rb') as f:
    state = pickle.load(f)

  rng = random.Random(args.random_seed)

  seq, hull = generate_data_record(rng, args.num_vertex)
  del hull
  with open('/tmp/points.pk', 'wb') as f:
    pickle.dump(seq, f)

  generate.apply(state.params, state.aux, state.rng, seq, args)
