from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 

    # normalize the scores
    # multiply the attention scores to the value and get back V'
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]

    attention = (query @ key.transpose(-2, -1))/(1.0 / math.sqrt(key.size(-1)))  #(B, nh, T, hs) #(B, nh, hs, T) -> (B, nh, T, T) 
    #mask= torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    attention=attention.masked_fill(attention_mask[:,:,:,:seq_len]==0,-1e10)
    attention = F.softmax(attention, dim=-1)
    V_out = attention @ value                                       # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    V_out = V_out.transpose(2,1).view(-1, seq_len, hidden_size)    #(B, nh, T, hs) -> (B, T, nh*hs) 
    return V_out


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


#==========================Transformer forward
hidden_size=768
num_attention_heads=12
seq_len=10
bs=32
attention_head_size = int(hidden_size / num_attention_heads)  #head for each head 64
all_head_size = num_attention_heads * attention_head_size #768
key=torch.randn(32,12,seq_len,attention_head_size)   # 768,768  [bs, num_attention_heads, seq_len, attention_head_size]
query=torch.randn(32,12,seq_len,attention_head_size)    #(B, nh, T, hs)
value=torch.randn(32,12,seq_len,attention_head_size)    #(B, nh, T, hs)
attention_before = (query @ key.transpose(-2, -1))/(1.0 / math.sqrt(key.size(-1)))  #(B, nh, T, hs) #(B, nh, hs, T) -> (B, nh, T, T) 
mask= torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
attention=attention_before.masked_fill(mask[:,:,:seq_len,:seq_len]==0,-1e10)
attention = F.softmax(attention, dim=-1)

V_out = attention @ value     # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#print(V_out.shape)
V_out = V_out.transpose(2,1).contiguous().view(bs, seq_len, hidden_size)    #(B, nh, T, hs) -> (B, T, nh*hs) 
#print(V_out.shape)

mask1=torch.rand(32, 1, 1, seq_len)
attention=mask1+attention_before
attention = F.softmax(attention, dim=-1)
V_out = attention @ value     # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
V_c = V_out.transpose(2,1).contiguous()
print(V_c.shape)
V_c_new = V_c.size()[:-2] #+ (hidden_size,)
print(V_c_new)
output = torch.reshape(V_c, V_c_new)
print(output.shape)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = torch.reshape(context_layer, new_context_layer_shape)