
from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from haiku_trainer import Trainer

import wandb
from data import ConvexHullDataLoader, seq_len
from model import PointerNet


def create_network(hparams):
  return PointerNet(hparams.rnn_hidden_size, padded_input_len=max(seq_len))


def plot_attention(trainer: Trainer, hparams, wandb=None):
  net = create_network(hparams)
  seq, seq_mask, out, out_mask = next(trainer.val_iter)
  del out, out_mask
  seq = seq[0]
  seq_mask = seq_mask[0]
  logits = net((seq[None], seq_mask[None]))

  plt.figure(figsize=(3, 10))
  size = jnp.clip(jnp.exp(logits) * 200, a_min=10, a_max=100)

  for i in range(5):
    plt.subplot(5, 1, i+1)
    plt.scatter(seq[:51, 1], seq[:51, 2], s=size[0, i], alpha=0.3, c='red')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.title(f'step {i}')
  plt.savefig('/tmp/attention.png')
  if wandb is not None:
    wandb.log({'attention': wandb.Image('/tmp/attention.png')}, commit=False)
  plt.close('all')


def _loss_fn(inputs, hparams):
  net = create_network(hparams)
  seq, seq_mask, out, out_mask = inputs
  logits = net((seq, seq_mask))
  target = jax.nn.one_hot(out, max(seq_len) + 1)
  loss = logits * target
  loss = jnp.sum(loss, axis=-1)
  loss = loss * out_mask
  return -(jnp.sum(loss) / jnp.sum(out_mask))


def main():
  parser = ArgumentParser()
  parser.add_argument('-b', '--batch-size', default=32, type=int)
  parser.add_argument('-d', '--rnn-hidden-size', default=256, type=int)
  parser.add_argument('-f', '--data-file', default='/tmp/convex_hull.dat', type=str)
  parser.add_argument('-l', '--lr', default=1e-3, type=float)
  parser.add_argument('-r', '--resume-training', default=False, action='store_true')
  parser.add_argument('-t', '--training-steps', default=100_000, type=int)
  parser.add_argument('-w', '--use-wandb', default=False, action='store_true')
  parser.add_argument('-wd', '--wd', default=1e-2, type=float)
  hparams = parser.parse_args()
  if hparams.use_wandb:
    wandb.init(project='pointer-networks', dir='/tmp')
    wandb.config.update(hparams)

  print(hparams)
  dataloader = ConvexHullDataLoader(data_filepath=hparams.data_file)

  loss_fn = partial(_loss_fn, hparams=hparams)
  optimizer = optax.chain(
      optax.adamw(hparams.lr, weight_decay=hparams.wd),
      optax.clip_by_global_norm(10.)
  )

  train_iter = dataloader.data_iter(hparams.batch_size, 'train')
  val_iter = dataloader.data_iter(hparams.batch_size, 'val')

  wandb_obj = wandb if hparams.use_wandb else None

  trainer = Trainer(
      train_loss_fn=loss_fn,
      train_data_iter=train_iter,
      val_loss_fn=loss_fn,
      val_data_iter=val_iter,
      optimizer=optimizer,
      wandb=wandb_obj,
      resume=hparams.resume_training
  )

  plot_att_fn = partial(plot_attention, hparams=hparams, wandb=wandb_obj)
  trainer.register_callback(1000, plot_att_fn)

  trainer.fit(total_steps=hparams.training_steps)


if __name__ == "__main__":
  main()
