
import math
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax._src.numpy.lax_numpy import ndarray


class MyLSTM(hk.LSTM):
  """A LSTM with learnable initial state."""

  def __init__(self, hidden_size: int):
    super().__init__(hidden_size=hidden_size)

    h0 = hk.get_parameter('h0', [hidden_size], init=hk.initializers.Constant(0.))
    c0 = hk.get_parameter('c0', [hidden_size], init=hk.initializers.Constant(0.))
    self.h0c0 = hk.LSTMState(h0, c0)

  def initial_state(self, batch_size: int) -> hk.LSTMState:
    def broadcast(x): return jnp.broadcast_to(x, (batch_size,) + x.shape)
    return jax.tree_map(broadcast, self.h0c0)


class PointerNet(hk.Module):
  def __init__(self, hidden_size: int = 512, padded_input_len: int = -1):
    super().__init__()

    self.lstm = hk.ResetCore(MyLSTM(hidden_size))
    self.enc_att_fc = hk.Linear(128, with_bias=False)
    self.dec_att_fc = hk.Linear(128, with_bias=False)
    # self.energy_fc = hk.Linear(1, with_bias=False)
    self.padded_input_len = padded_input_len

  def __call__(self, inputs: Tuple[ndarray, ndarray]):
    input_seq, input_mask = inputs
    B, L, D = input_seq.shape
    del L, D

    input_seq = jnp.swapaxes(input_seq, 0, 1)
    input_mask = jnp.swapaxes(input_mask, 0, 1)
    reset_mask = jnp.logical_not(input_mask)
    reset_mask = reset_mask.at[1:].set(reset_mask[:-1])  # move the mask to the right
    h0c0: hk.LSTMState = self.lstm.initial_state(B)  # type: ignore
    hx, state = hk.dynamic_unroll(self.lstm, (input_seq, reset_mask), h0c0)
    del state

    # split encoder/decoder states
    encoder_hx = hx[:self.padded_input_len]
    decoder_hx = hx[self.padded_input_len:]

    # append the initial hidden state.
    # this will be the encoder state for the [end] token.
    encoder_hx = jnp.concatenate([encoder_hx, h0c0.hidden[None]], axis=0)

    # create query and value for attention mechanism
    encoder_value = self.enc_att_fc(encoder_hx)[None]
    decoder_query = self.dec_att_fc(decoder_hx)[:, None]

    # energy function
    energy = encoder_value * decoder_query
    energy = jnp.sum(energy, axis=-1) / math.sqrt(energy.shape[-1])

    # apply input sequence mask
    input_mask = input_mask[:self.padded_input_len+1][None]
    energy = jnp.where(input_mask, energy, float('-inf'))

    # normalize
    energy = jax.nn.log_softmax(energy, axis=1)

    # batch first, logit last
    return jnp.transpose(energy, [2, 0, 1])
