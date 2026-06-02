from torch import nn
import torch.nn.functional as F

from modules.attention import CausalSelfAttention


class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.self_attention = CausalSelfAttention(config)
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    return input + dropout(dense_layer(output))

  def forward(self, hidden_states, attention_mask):
    residual = hidden_states
    hidden_states = self.add(
      residual,
      self.self_attention(self.attention_layer_norm(hidden_states), attention_mask),
      self.attention_dense, self.attention_dropout)
    residual = hidden_states
    hidden_states = self.add(
      residual,
      self.interm_af(self.interm_dense(self.out_layer_norm(hidden_states))),
      self.out_dense, self.out_dropout)
    return hidden_states


