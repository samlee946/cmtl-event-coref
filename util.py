from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import json
import shutil
import sys

import numpy as np
import tensorflow as tf
import pyhocon
import random

import independent_event_coref

def get_model(config):
    if config['model_type'] == 'independent':
      return independent_event_coref.CorefModel(config)
    else:
      raise NotImplementedError('Undefined model type')

def initialize_from_env(conf_file="experiments.conf", eval_test=False):
  if "GPU" in os.environ:
    set_gpus(int(os.environ["GPU"]))

  name = sys.argv[1]
  print("Running experiment: {}".format(name))

  if eval_test:
    config = pyhocon.ConfigFactory.parse_file("test.experiments.conf")[name]
  else:
    config = pyhocon.ConfigFactory.parse_file(conf_file)[name]
  config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

  print(pyhocon.HOCONConverter.convert(config, "hocon"))

  with open(os.path.join(config["log_dir"], "config.txt"), "w") as outf:
    keys = sorted(list(config.keys()))
    for key in keys:
      outf.write(key + "\t" + str(config[key]) + "\n")

  return config

def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])

def flatten(l):
  return [item for sublist in l for item in sublist]

def set_gpus(*gpus):
  # pass
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
  print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path

def load_arg_role_dict(arg_role_path):
  roles = ["NULL"]
  with open(arg_role_path, 'r') as f:
    roles.extend(l.strip() for l in f.readlines())

  arg_role_dict = {}
  arg_role_dict.update({r:i for i, r in enumerate(roles)})
  return arg_role_dict

def load_coref_cluster_dict(coref_cluster_path):
  coref_cluster_dict = {}
  with open(coref_cluster_path, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
      items = line.split("\t")
      for trigger in items[0].split("#"):
        coref_cluster_dict[trigger] = i + 1
  return coref_cluster_dict

def load_event_subtype_dict(event_subtype_path):
  subtype_dict = {}
  with open(event_subtype_path, 'r') as f:
    lines = f.readlines()
    subtype_dict = {m.strip(): i + 1 for i, m in enumerate(lines)}
  return subtype_dict

def load_char_dict(char_vocab_path):
  vocab = [u"<unk>"]
  with codecs.open(char_vocab_path, encoding="utf-8") as f:
    vocab.extend(l.strip() for l in f.readlines())
  char_dict = collections.defaultdict(int)
  char_dict.update({c:i for i, c in enumerate(vocab)})
  return char_dict

def load_subtype_dict(subtype_path):
  subtype_dict = {}
  type_dict = {}
  subtype_type_map = []

  with open(subtype_path, 'r') as f:
    lines = f.readlines()
    subtype_dict = {m.strip(): i + 1 for i, m in enumerate(lines)}

    for i, m in enumerate(lines):
      etype = m.strip().split("_")[0]
      if etype in type_dict:
        etype_id = type_dict[etype]
      else:
        etype_id = len(type_dict) + 1
        type_dict[etype] = etype_id
      subtype_type_map.append(etype_id)

  subtype_type_map = np.array(subtype_type_map)

  return subtype_dict, type_dict, subtype_type_map
  
def maybe_divide(x, y):
  return 0 if y == 0 else x / float(y)

def projection(inputs, output_size, initializer=tf.truncated_normal_initializer(stddev=0.02), efficient=False, activate_fn="relu"):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer, efficient=efficient, activate_fn=activate_fn)

def highway(inputs, num_layers, dropout):
  for i in range(num_layers):
    with tf.variable_scope("highway_{}".format(i)):
      j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
      f = tf.sigmoid(f)
      j = tf.nn.relu(j)
      if dropout is not None:
        j = tf.nn.dropout(j, dropout)
      inputs = f * j + (1 - f) * inputs
  return inputs

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, 
  output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02), 
  hidden_initializer=tf.truncated_normal_initializer(stddev=0.02),
  efficient=False, activate_fn="relu"):
  if len(inputs.get_shape()) > 3:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

  if len(inputs.get_shape()) == 3:
    batch_size = shape(inputs, 0)
    seqlen = shape(inputs, 1)
    emb_size = shape(inputs, 2)
    current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
  else:
    current_inputs = inputs

  for i in range(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size], initializer=hidden_initializer)
    hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size], initializer=tf.zeros_initializer())
    if activate_fn == "leaky_relu":
      current_outputs = tf.nn.leaky_relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))
    else:
      current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
  output_bias = tf.get_variable("output_bias", [output_size], initializer=tf.zeros_initializer())
  outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])

  if efficient:
    outputs = tf.contrib.layers.recompute_grad(outputs)
  return outputs

def linear(inputs, output_size, efficient=False):
  if len(inputs.get_shape()) == 3:
    batch_size = shape(inputs, 0)
    seqlen = shape(inputs, 1)
    emb_size = shape(inputs, 2)
    current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
  else:
    current_inputs = inputs
  hidden_weights = tf.get_variable("linear_w", [shape(current_inputs, 1), output_size])
  hidden_bias = tf.get_variable("bias", [output_size])
  current_outputs = tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias)

  if efficient:
    current_outputs = tf.contrib.layers.recompute_grad(current_outputs)

  return current_outputs

def cnn(inputs, filter_sizes, num_filters, efficient=False):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv_{}".format(i)):
      w = tf.get_variable("w", [filter_size, input_size, num_filters])
      b = tf.get_variable("b", [num_filters])
    conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
    h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
    pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
    outputs.append(pooled)

  cnn_output = tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]
  if efficient:
    cnn_output = tf.contrib.layers.recompute_grad(cnn_output)
  return cnn_output

def batch_gather(emb, indices):
  batch_size = shape(emb, 0)
  seqlen = shape(emb, 1)
  if len(emb.get_shape()) > 2:
    emb_size = shape(emb, 2)
  else:
    emb_size = 1
  flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
  offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
  gathered = tf.gather(flattened_emb, indices + offset) # [batch_size, num_indices, emb]
  if len(emb.get_shape()) == 2:
    gathered = tf.squeeze(gathered, 2) # [batch_size, num_indices]
  return gathered

class RetrievalEvaluator(object):
  def __init__(self):
    self._num_correct = 0
    self._num_gold = 0
    self._num_predicted = 0

  def update(self, gold_set, predicted_set):
    self._num_correct += len(gold_set & predicted_set)
    self._num_gold += len(gold_set)
    self._num_predicted += len(predicted_set)

  def recall(self):
    return maybe_divide(self._num_correct, self._num_gold)

  def precision(self):
    return maybe_divide(self._num_correct, self._num_predicted)

  def metrics(self):
    recall = self.recall()
    precision = self.precision()
    f1 = maybe_divide(2 * recall * precision, precision + recall)
    return recall, precision, f1

class EmbeddingDictionary(object):
  def __init__(self, info, normalize=True, maybe_cache=None):
    self._size = info["size"]
    self._normalize = normalize
    self._path = info["path"]
    if maybe_cache is not None and maybe_cache._path == self._path:
      assert self._size == maybe_cache._size
      self._embeddings = maybe_cache._embeddings
    else:
      self._embeddings = self.load_embedding_dict(self._path)

  @property
  def size(self):
    return self._size

  def load_embedding_dict(self, path):
    print("Loading word embeddings from {}...".format(path))
    default_embedding = np.zeros(self.size)
    embedding_dict = collections.defaultdict(lambda:default_embedding)
    if len(path) > 0:
      vocab_size = None
      with open(path) as f:
        for i, line in enumerate(f.readlines()):
          word_end = line.find(" ")
          word = line[:word_end]
          embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
          assert len(embedding) == self.size
          embedding_dict[word] = embedding
      if vocab_size is not None:
        assert vocab_size == len(embedding_dict)
      print("Done loading word embeddings.")
    return embedding_dict

  def __getitem__(self, key):
    embedding = self._embeddings[key]
    if self._normalize:
      embedding = self.normalize(embedding)
    return embedding

  def normalize(self, v):
    norm = np.linalg.norm(v)
    if norm > 0:
      return v / norm
    else:
      return v

class CustomLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, batch_size, dropout):
    self._num_units = num_units
    self._dropout = dropout
    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
    initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
    initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
      c, h = state
      h *= self._dropout_mask
      concat = projection(tf.concat([inputs, h], 1), 3 * self.output_size, initializer=self._initializer)
      i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
      i = tf.sigmoid(i)
      new_c = (1 - i) * c  + i * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer


def error_delta(subtype_labels, num_subtypes):
  # used to be trigger_delta
  one_hot_subtype_labels = tf.one_hot(subtype_labels, num_subtypes)
  gold_index = one_hot_subtype_labels > 0

  gold_index = tf.to_float(tf.logical_not(gold_index))

  delta_0 = tf.zeros([shape(subtype_labels, 0), 1]) + self.config["false_null"] # [num_mentions, 1]
  delta_1 = tf.zeros([shape(subtype_labels, 0), num_subtypes - 1]) + self.config["false_subtype"] # [num_mentions, max_ant]

  delta = tf.concat([delta_0, delta_1], 1) # [num_mentions, num_subtypes]
  delta = tf.multiply(delta, gold_index) # [num_mentions, num_subtypes]
  return delta


def error_delta_v2(subtype_labels, num_subtypes, false_null_weight, false_subtype_weight, false_nonnull_weight):
  # # used to be trigger_delta_v2
  # for multi-class models
  # Example: gold labels: m1: null, m2: subtype1, m3: subtype2
  # delta
  #     null  subtype1       subtype2
  # m1  0     false_nonnull  false_nonnull
  # m2  f_null  0            false_subtype
  # m3  f_null  false_subtype 0
  one_hot_subtype_labels = tf.one_hot(subtype_labels, num_subtypes)
  gold_index = one_hot_subtype_labels > 0
  gold_index = tf.to_float(tf.logical_not(gold_index)) #[num_mentions, num_subtypes]

  k = shape(subtype_labels, 0)

  null_label = tf.slice(one_hot_subtype_labels, [0, 0], [k, 1]) # 1 if null, 0 if not-null
  null_label = tf.reshape(null_label, [k])
  null_label = null_label > 0

  false_nonnull =  tf.zeros([k]) + false_nonnull_weight
  false_subtype =  tf.zeros([k]) + false_subtype_weight

  delta_1 = tf.where(null_label, false_nonnull, false_subtype)
  delta_1 = tf.reshape(delta_1, [k, 1])
  delta_1 = tf.tile(delta_1, [1, num_subtypes - 1]) #[k, num_subtypes-1]

  delta_0 = tf.zeros([k, 1]) + false_null_weight # [num_mentions, 1]

  delta = tf.concat([delta_0, delta_1], 1) # [num_mentions, num_subtypes]
  delta = tf.multiply(delta, gold_index) # [num_mentions, num_subtypes]
  return delta

def delta(antecedent_labels, antecedent_mask):
  #antecedent_labels: [k, c+1]
  #antecedent_mask: [k, c]

  gold_index = tf.to_float(tf.logical_not(antecedent_labels)) # [num_mentions, max_ant+1]
  delta_0 = tf.zeros([shape(antecedent_labels, 0), 1]) + self.config["false_new"] # [num_mentions, 1]
  delta_1 = tf.zeros([shape(antecedent_labels, 0), shape(antecedent_labels, 1) - 1]) + self.config["false_link"] # [num_mentions, max_ant]

  delta_1 = tf.multiply(delta_1, tf.to_float(antecedent_mask)) # [num_mentions, max_ant]

  delta = tf.concat([delta_0, delta_1], 1) # [num_mentions, max_ant+1]
  delta = tf.multiply(delta, gold_index) # [num_mentions, max_ant+1]
  return delta

def delta_v2(antecedent_labels, antecedent_mask,  false_new_weight, false_link_weight, wrong_link_weight):
  #antecedent_labels: [k, c+1]
  #antecedent_mask: [k, c]
  k = shape(antecedent_labels, 0)
  c = shape(antecedent_mask, 1)

  dummy_label = tf.slice(antecedent_labels, [0, 0], [k, 1])
  #dummy_label = dummy_label > 0
  dummy_label = tf.reshape(dummy_label, [k])

  false_link =  tf.zeros([k]) + false_link_weight
  wrong_link =  tf.zeros([k]) + wrong_link_weight
  delta_1 = tf.where(dummy_label, false_link, wrong_link)
  delta_1 = tf.reshape(delta_1, [k, 1])
  delta_1 = tf.tile(delta_1, [1, c]) #[k, c]

  gold_index = tf.to_float(tf.logical_not(antecedent_labels)) # [num_mentions, max_ant+1]
  delta_0 = tf.zeros([k, 1]) + false_new_weight # [num_mentions, 1]
  delta_1 = tf.multiply(delta_1, tf.to_float(antecedent_mask)) # [num_mentions, max_ant]

  delta = tf.concat([delta_0, delta_1], 1) # [num_mentions, max_ant+1]
  delta = tf.multiply(delta, gold_index) # [num_mentions, max_ant+1]
  return delta

def set_seeds(seed=11):
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  tf.compat.v1.random.set_random_seed(11)
  np.random.seed(seed)

def set_global_determinism(seed=11):
  set_seeds(seed=seed)

  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.config.threading.set_intra_op_parallelism_threads(1)