"""Data utils.
  Provide functions to create regression datasets.
"""

import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfinv
number_of_D=25

seed=1


class DocWeight(hk.Module):
    def __init__(self, d_y=1, d_d=10, d_q=10, name=None):
        super().__init__(name=name)
        self.d_y = d_y
        self.d_d = d_d
        self.d_q = d_q
        self.out_dim=10

    def __call__(self, docs):
    
        W1b = hk.get_parameter("W1b2", shape=(self.d_y, self.d_d),
                               init=hk.initializers.RandomNormal(0.1))
        M   = hk.get_parameter("M2", shape=(self.d_q, self.out_dim),
                               init=hk.initializers.RandomNormal(0.1))

       
        D = jnp.einsum("ki,kj->ij", docs, docs)   # (d_d, d_d)

    
        W2 = W1b @ D @ M.T    # (d_y, d_q)
        return W2


def forward_fn(docs):
    doc_layer = DocWeight()
    return doc_layer(docs)

forward = hk.transform(forward_fn)

import jax
import jax.numpy as jnp
key = jax.random.PRNGKey(seed)

# d_d=10
# docs = 0.1+jax.random.normal(key, (number_of_D, 10)) * 0.5
# docs = jax.random.normal(key, (number_of_D, 10))  # k=2, d_d=10
c_size=number_of_D
i_size=10
docs = jax.random.uniform(key, shape=[c_size, i_size],
                         minval=-1 / 2, maxval=1 / 2)

params = forward.init(key, docs)   
doc_weight = forward.apply(params, None, docs)   # # W2 ∈ R^{dy × dq}


"""-----------need to change--0-------"""
document_number=1


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data(rng, i_size, c_size, size_distract, input_range, w_scale):
  """Create a linear regression data set: X*w where x ~ U(-1, 1), w ~ N(0,1)."""
  """
  input_size, dataset_size, size_distract,input_range
  """
  # print("i_size",i_size)
 
  i_size=i_size*(document_number+1)   
  sentence_dim=doc_dim=10
  num_docs=document_number
  rng, new_rng, new_rng2, new_rng3, new_rng4,new_rng5 = jax.random.split(rng, 6)


  rng, rng_W1, rng_W2,rng_W3 = jax.random.split(rng, 4)
  x = jax.random.uniform(new_rng, shape=[c_size, i_size],
                         minval=-input_range / 2, maxval=input_range / 2)

  W1 = jax.random.normal(rng_W1, shape=(1, 20)) * w_scale  # sentence part
    
  # W1 = W1.at[:, 10:20].set(doc_weight)

  W2 = jax.random.normal(rng_W2, shape=(1, num_docs*10)) * w_scale  # document part
  query = x[:, :int(i_size/(num_docs+1))]  # [c_size, 10]
  doc_all=query
  # doc_all = x[:, int(i_size/(num_docs+1)):]  # [c_size, 20]
  docs = doc_all.reshape(c_size, num_docs, doc_dim)  # [c_size, 2, 10]

  We = jax.random.normal(rng_W3, shape=(doc_dim, doc_dim)) * w_scale
  query_proj = query @ We  # [c_size, 10]
  docs_proj = jnp.einsum('bnd,df->bnf', docs, We)  # [c_size, 2, 10]
  weight = jnp.einsum('bd,bnd->bn', query_proj, docs_proj)  # [c_size, 2]
  # print("weight",weight.shape)  # 10,2 
 
  eps = 1e-6
  # weight = W1 @ weight
  # D = jnp.linalg.pinv(weight) @ W2  # shape: (2, 20)
  # W2 = weight @ D  #  

  w = W1#jnp.concatenate([W1,W2], axis=1)  # shape: (1, 30)

    
  w=jnp.squeeze(w)  # shape = (30,)

  x_querry = jax.random.uniform(new_rng2, shape=[1, i_size],
                                minval=-input_range/2, maxval=input_range/2)

  y_data = jnp.squeeze(x@w)
  choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                             replace=False)
  y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                   shape=[size_distract]))

  y_target = x_querry@w
  y_target = y_target[..., None]

  seq = jnp.concatenate([x, y_data[..., None]], -1)
  target = jnp.concatenate([x_querry, y_target], -1)
  x_querry_init = -1*x_querry.dot(jnp.ones_like(x_querry).T*0.0)
  zero = jnp.concatenate([x_querry, x_querry_init], -1)
  seq = jnp.concatenate([seq, zero], 0)

  return jnp.squeeze(seq), jnp.squeeze(target), w


"""
w = [0.1, -0.2, 0.05, 0.3, -0.1, 0.15, 0.2, -0.05, 0.4, -0.3]

seq (11 x 11):

[
 [ 0.10, -0.20,  0.30, -0.10, 0.05, -0.05, 0.40, -0.30, 0.20, -0.10,  0.12],
 [-0.40,  0.20, -0.10,  0.10, 0.30, -0.30, 0.05, -0.20, 0.15,  0.05, -0.08],
 [ 0.20, -0.10,  0.05, -0.30, 0.40, -0.20, 0.30, -0.10, 0.05, -0.05,  0.15],
 [ 0.30,  0.10, -0.05,  0.40, -0.10,  0.05, -0.30,  0.20, -0.20, 0.10,  0.09],
 [ 0.15, -0.05,  0.30, -0.20, 0.10,  0.40, -0.10,  0.05, -0.30, -0.20, 0.11],
 [-0.10,  0.40, -0.30,  0.05, -0.20, 0.15, -0.05,  0.30,  0.20, -0.10, -0.04],
 [ 0.05, -0.10,  0.20, -0.30, 0.40, -0.20, 0.30, -0.10,  0.05, -0.05,  0.13],
 [ 0.40, -0.30,  0.10, -0.20, 0.05,  0.30, -0.10,  0.20, -0.05,  0.10,  0.07],
 [-0.30,  0.20, -0.10,  0.30, -0.05, 0.40, -0.20,  0.15, -0.10,  0.05, -0.06],
 [ 0.20, -0.05,  0.30, -0.10, 0.05, -0.20,  0.40, -0.30,  0.10, -0.10,  0.10],
 # 查询样本 (y 初始化为 0)
 [ 0.05, -0.10,  0.20, -0.30, 0.40, -0.20,  0.30, -0.10,  0.05, -0.05,  0.00]
]
target (11,):
[0.05, -0.10, 0.20, -0.30, 0.40, -0.20, 0.30, -0.10, 0.05, -0.05, 0.14]


"""


# data_creator = vmap(create_reg_data,
#                     in_axes=(0, None, None, None, None, None), out_axes=0)
#
# rng = jax.random.PRNGKey(0)
# rng, test_rng_avg = jax.random.split(rng, 2)
# test_data = data_creator(jax.random.split(rng, num=1), 3, 10, 0, 2, 1)


@partial(jax.jit, static_argnums=(1, 2))


def create_ood_data(rng, i_size, c_size, input_range, w_scale):
    """Create a ood data set: X*w where X ~ Normal, Exponential, Poisson."""

    rng, new_rng, new_rng2, new_rng3 = jax.random.split(rng, 4)
    w = jax.random.normal(rng, shape=[i_size]) * w_scale

    selector = jnp.zeros([3])
    choice = jax.random.choice(new_rng3, 3, replace=False)
    selector = selector.at[choice].set(1)

    x_sample = jax.random.exponential(new_rng, shape=[c_size, i_size])
    norm_x_sample = jnp.linalg.norm(x_sample)
    x = x_sample / norm_x_sample * input_range * selector[0]
    x_q_sample = jax.random.exponential(new_rng2, shape=[1, i_size])
    x_querry = x_q_sample / norm_x_sample * input_range * selector[0]

    x_sample = jax.random.normal(new_rng, shape=[c_size, i_size])
    norm_x_sample = jnp.linalg.norm(x_sample)
    x += x_sample / norm_x_sample * input_range * selector[1]
    x_q_sample = jax.random.normal(new_rng2, shape=[1, i_size])
    x_querry += x_q_sample / norm_x_sample * input_range * selector[1]

    x_sample = jax.random.laplace(new_rng, shape=[c_size, i_size])
    norm_x_sample = jnp.linalg.norm(x_sample)
    x += x_sample / norm_x_sample * input_range * selector[2]
    x_q_sample = jax.random.laplace(new_rng2, shape=[1, i_size])
    x_querry += x_q_sample / norm_x_sample * input_range * selector[2]

    y_data = jnp.squeeze(x @ w)

    y_target = x_querry @ w
    y_target = y_target[..., None]

    seq = jnp.concatenate([x, y_data[..., None]], -1)
    target = jnp.concatenate([x_querry, y_target], -1)
    x_querry_init = -1 * x_querry.dot(jnp.ones_like(x_querry).T * 0.0)
    zero = jnp.concatenate([x_querry, x_querry_init], -1)
    seq = jnp.concatenate([seq, zero], 0)
    return jnp.squeeze(seq), jnp.squeeze(target), w


data_creator = vmap(create_ood_data,
                    in_axes=(0, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(0)
rng, test_rng_avg = jax.random.split(rng, 2)
test_data = data_creator(jax.random.split(rng, num=1), 2, 4, 1, 1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data_sin(rng, i_size, c_size, size_distract,
                        input_range=10, w_scale=1):
    """Create a sin wave regression data set."""

    rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
    amp = jax.random.uniform(rng, shape=[1], minval=0.1, maxval=0.5) * w_scale
    phase = jax.random.uniform(rng, shape=[1], minval=0.0,
                               maxval=1) * jnp.pi * w_scale

    x = jax.random.uniform(new_rng, shape=[c_size, 1],
                           minval=-input_range / 2, maxval=input_range / 2)
    x_querry = jax.random.uniform(new_rng2, shape=[1, 1],
                                  minval=-input_range / 2, maxval=input_range / 2)

    y_data = jnp.sin(x + phase) * amp
    choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                               replace=False)
    y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                     shape=[size_distract, 1]))

    y_target = jnp.sin(x_querry + phase) * amp
    seq = jnp.concatenate([x, y_data], -1)
    target = jnp.concatenate([x_querry, y_target], -1)
    y_querry_init = jnp.zeros_like(y_target)

    zero = jnp.concatenate([x_querry, y_querry_init], -1)
    seq = jnp.concatenate([seq, zero], 0)
    return jnp.squeeze(seq), jnp.squeeze(target), (phase, amp)


data_creator = vmap(create_reg_data_sin,
                    in_axes=(0, None, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(0)
rng, test_rng_avg = jax.random.split(rng, 2)
test_data = data_creator(jax.random.split(rng, num=1), 1, 10, 2, 10, 1)


@partial(jax.jit, static_argnums=(2, 3))
def create_reg_data_sin_test(rng, rng2, c_size, input_range, w_scale):
    """Dublicate of the obove - TODO."""

    amp = jax.random.uniform(rng2, shape=[1], minval=0.1, maxval=0.5) * w_scale
    phase = jax.random.uniform(rng2, shape=[1], minval=0.0,
                               maxval=1) * jnp.pi * w_scale

    x = jax.random.uniform(rng2, shape=[c_size, 1],
                           minval=-input_range / 2, maxval=input_range / 2)
    x_querry = jax.random.uniform(rng, shape=[1, 1],
                                  minval=-input_range / 2, maxval=input_range / 2)
    y_data = jnp.sin(x + phase) * amp
    y_target = jnp.sin(x_querry + phase) * amp
    seq = jnp.concatenate([x, y_data], -1)
    target = jnp.concatenate([x_querry, y_target], -1)
    y_querry_init = jnp.zeros_like(y_target)

    zero = jnp.concatenate([x_querry, y_querry_init], -1)
    seq = jnp.concatenate([seq, zero], 0)
    return jnp.squeeze(seq), jnp.squeeze(target), (phase, amp)




@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data_classic_token(rng, i_size, c_size, size_distract,
                                  input_range, w_scale):
    """Create a linear regression data set: X*w where x ~ U[-1,1], w ~ N(0,1)."""

    rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
    w = jax.random.normal(rng, shape=[i_size]) * w_scale

    x = jax.random.uniform(new_rng,
                           shape=[c_size, i_size]) * input_range - (input_range / 2)
    x_querry = jax.random.uniform(new_rng2,
                                  shape=[1, i_size]) * input_range - (input_range / 2)
    y_data = jnp.squeeze(x @ w)
    y_data_zero = jnp.zeros_like(x[:, :-1])
    y_data = jnp.concatenate([y_data_zero, y_data[..., None]], axis=-1)
    y_target = x_querry @ w
    choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                               replace=False)

    y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                     shape=[size_distract,
                                                            i_size]))
    y_target_zero = jnp.zeros_like(x_querry[:, :-1])
    y_target = y_target[..., None]

    seq = jnp.concatenate([x, y_data], 1)
    seq = seq.reshape(-1, i_size)
    target = jnp.concatenate([y_target_zero, y_target], -1)
    seq = jnp.concatenate([seq, x_querry], 0)
    return jnp.squeeze(seq), jnp.squeeze(target), w




# print("doc_weight",doc_weight)
def create_weights(i_size, o_size, c_size, lr, w_init, second_zero=False,
                   lin_diag=False, gd_deq=False, num_layers=1,
                   input_mlp_rnd=None, in_proj=False):
  """Create linear regression gradient descent weights for self-attention
  layer."""
  i_size=i_size*2
  one = jnp.ones([i_size+o_size])
  one_in_size = jnp.ones([i_size])
  zero_out_size = jnp.zeros([o_size])
  one_out_size = jnp.ones([o_size])

  # Value matrix
  value_upper = jnp.zeros([i_size, i_size+o_size])
  # value_lower_left = doc_weight#w_init[0]
  #print(value_lower_left)
  value_lower_left=jnp.concatenate([w_init[0], doc_weight], axis=1)
  if lin_diag:
    value_lower_right = jnp.diag(one_out_size)*-2
  else:
    value_lower_right = jnp.diag(one_out_size)*-1

  if second_zero:
    value_lower_right = jnp.diag(zero_out_size)

  value_lower_part = jnp.concatenate([value_lower_left, value_lower_right],
                                     axis=1)
  value_matrix = jnp.concatenate([value_upper, value_lower_part], axis=0).T
  if lin_diag:
    value_matrix += jnp.diag(one)

  #value_bias = jnp.zeros([i_size + o_size])

  # Query and Key matrix
  query_upper_part = jnp.zeros([o_size, i_size+o_size])
  query_lower_left = jnp.diag(one_in_size)
  query_lower_right = jnp.zeros([i_size, o_size])
  query_lower_part = jnp.concatenate([query_lower_left, query_lower_right],
                                     axis=1)
  query_matrix = jnp.concatenate([query_lower_part, query_upper_part], axis=0)
  key_matrix = query_matrix

  #query_bias = jnp.zeros([i_size + o_size])
  #key_bias = query_bias

  # Projection matrix
  projection_upper_part = jnp.zeros([i_size, i_size+o_size])
  projection_lower_left = jnp.zeros([o_size, i_size])

  projection_lower_right = jnp.diag(one_out_size)*((1/c_size)*lr)

  if lin_diag:
    shifted_lr = jnp.diag(one_out_size)*(1/c_size)*(1/c_size)*lr
    projection_lower_right += shifted_lr

  projection_lower_part = jnp.concatenate([projection_lower_left,
                                           projection_lower_right], axis=1)
  projection_matrix = jnp.concatenate([projection_upper_part,
                                       projection_lower_part], axis=0)
  if lin_diag:
    projection_matrix -= jnp.diag(one)*(1/c_size)*(1/c_size)*lr

  #projection_bias = jnp.zeros([i_size + o_size])

  params_new = {}
  for l in range(num_layers):
    if num_layers == 1 or gd_deq:
      tra_name = 'Transformer_gd/multi_head_attention/'
    else:
      tra_name = 'Transformer_gd/~trans_block/layer_'+str(l)+'/'
    params_new[tra_name+ 'query'] = {'w': jnp.array(query_matrix)}
    params_new[tra_name+ 'value'] = {'w': jnp.array(value_matrix)}
    params_new[tra_name+ 'key'] = {'w': jnp.array(key_matrix)}
    params_new[tra_name+ 'linear'] = {'w': jnp.array(projection_matrix)}

  if in_proj:
    rng1, rng2, rng3 = jax.random.split(input_mlp_rnd, 3)
    w_embedding = jax.random.normal(rng1, shape=[11, 11])*jnp.sqrt(0.002/11)
    params_new['Transformer_gd/emb'] = {'w': w_embedding}
  elif input_mlp_rnd is not None:
    rng1, rng2, rng3 = jax.random.split(input_mlp_rnd, 3)
    w1 = jax.random.normal(rng1, shape=[40, 160])*jnp.sqrt(0.002/2)
    w2 = jax.random.normal(rng2, shape=[160, 40])*jnp.sqrt(0.002/40)
    #w3 = jax.random.normal(rng3, shape=[40, 41])*jnp.sqrt(0.002/40)
    b1 = jax.random.normal(rng1, shape=[160])*0
    b2 = jax.random.normal(rng2, shape=[40])*0
    #b3 = jax.random.normal(rng3, shape=[41])*0
    w_embedding = jax.random.normal(rng1, shape=[2, 40])*jnp.sqrt(0.002/2)
    params_new['Transformer_gd/input_mlp/linear'] = {'w': w1, 'b': b1}
    params_new['Transformer_gd/input_mlp/linear_1'] = {'w': w2, 'b': b2}
    params_new['Transformer_gd/emb'] = {'w': w_embedding}
    #params_new['Transformer_gd/mlp/linear_2'] = {'w': w3, 'b': b3}
    #params_new['Transformer_gd/mlp/linear_3'] = {'w': w3, 'b': b3}
  
  return params_new
