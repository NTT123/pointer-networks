import haiku as hk
import jax
import jax.numpy as jnp
from model import PointerNet


@hk.testing.transform_and_run
def test_pointer_net():

  inp = jnp.asarray([
      [[0., 0., 1., 2.],
       [0., 0., 2., 3.]],
      [[0., 1., 2., 3.],
       [0., 2., 3., 4.]]
  ])

  inp_mask = jnp.asarray([
      [0., 0., 1., 1.],
      [0., 1., 1., 1.]
  ])
  inp = jnp.swapaxes(inp, 1, 2)
  out = jnp.flip(inp, axis=1)[:, :-1]

  inp = jnp.pad(inp, ((0, 0), (0, 0), (0, 1)))
  out = jnp.pad(out, ((0, 0), (0, 0), (0, 1)))
  start_token = jnp.asarray([0., 0., 1.])
  start_token = jnp.broadcast_to(start_token, (2, 3))

  io = jnp.concatenate([inp, start_token[:, None, :], out], axis=1)
  mask = jnp.pad(inp_mask, ((0, 0), (0, 4)), constant_values=True)

  net = PointerNet(64, 4)

  logits = net((io, mask))
  assert logits.shape == (2, 4, 5)
