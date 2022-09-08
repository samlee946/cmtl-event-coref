#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import util

def read_doc_keys(fname):
    keys = set()
    with open(fname) as f:
        for line in f:
            keys.add(line.strip())
    return keys

if __name__ == "__main__":
  if len(sys.argv) > 3:
    config = util.initialize_from_env(conf_file = sys.argv[3])
  else:
    config = util.initialize_from_env()

  os.environ["GPU"] = sys.argv[2]
  util.set_gpus(int(os.environ["GPU"]))

  model = util.get_model(config)
  saver = tf.train.Saver()
  log_dir = config["log_dir"]
  with tf.Session() as session:
    
    checkpoint_path = os.path.join(config["log_dir"], "model.max.ckpt")
    model.restore(session, checkpoint_path)
    # Make sure eval mode is True if you want official conll results
    scores = model.evaluate(session, checkpoint_path + "-output", True)

    print(scores)

    #for score in scores['coref']:
    #  print (score)
