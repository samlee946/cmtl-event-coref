#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf

import util

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.95

if __name__ == "__main__":
    if len(sys.argv) > 3:
        config = util.initialize_from_env(conf_file=sys.argv[3])
    else:
        config = util.initialize_from_env()

    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    os.environ["GPU"] = sys.argv[2]
    util.set_gpus(int(os.environ["GPU"]))

    model = util.get_model(config)
    saver = tf.train.Saver(max_to_keep=2)

    log_dir = config["log_dir"]
    max_steps = config['num_epochs'] * config['num_docs']
    writer = tf.summary.FileWriter(log_dir, flush_secs=20)

    max_f1 = 0
    max_step = 0
    mode = 'w'

    with tf.Session(config=tfconfig) as session:
        session.run(tf.global_variables_initializer())
        model.start_enqueue_thread(session)
        accumulated_loss = 0.0
        accumulated_subtype_loss = 0.0
        accumulated_argument_loss = 0.0
        accumulated_realis_loss = 0.0
        accumulated_type_loss = 0.0
        accumulated_anaphoricity_loss = 0.0

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(session, ckpt.model_checkpoint_path)
            mode = 'a'
        fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
        fh.setFormatter(logging.Formatter(format))
        logger.addHandler(fh)

        initial_time = time.time()
        while True:
            tf_loss, tf_global_step, _, subtype_loss, realis_loss, argument_loss, type_loss, anaphoricity_loss, encoref_loss, en_subtype_loss, predictions = session.run(
                [model.loss, model.global_step, model.train_op, model.subtype_loss, model.realis_loss, model.arg_loss,
                 model.type_loss, model.anaphoricity_loss, model.encoref_loss, model.en_subtype_loss,
                 model.predictions])
            accumulated_loss += tf_loss

            if config["model_type"] == "independent":
                candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_span_mention_scores, coref_output, \
                subtype_output, realis_output, anaphoricity_output, type_output = predictions
            else:
                candidate_starts, candidate_ends, top_span_starts, top_span_ends, coref_output, subtype_output, realis_output, anaphoricity_output, type_output = predictions

            accumulated_subtype_loss += subtype_loss
            accumulated_realis_loss += realis_loss
            accumulated_argument_loss += argument_loss
            accumulated_type_loss += type_loss
            accumulated_anaphoricity_loss += anaphoricity_loss

            if tf_global_step % report_frequency == 0:
                total_time = time.time() - initial_time
                steps_per_second = tf_global_step / total_time

                average_loss = accumulated_loss / report_frequency
                average_subtype_loss = accumulated_subtype_loss / report_frequency
                average_realis_loss = accumulated_realis_loss / report_frequency
                average_argument_loss = accumulated_argument_loss / report_frequency
                average_type_loss = accumulated_type_loss / report_frequency
                average_anaphoricity_loss = accumulated_anaphoricity_loss / report_frequency
                logger.info("[{}] loss={:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} steps/s={:.2f}".format( \
                    tf_global_step, average_loss, average_subtype_loss, average_realis_loss, average_argument_loss,
                    average_type_loss, average_anaphoricity_loss, steps_per_second))
                writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
                accumulated_loss = 0.0
                accumulated_subtype_loss = 0.0
                accumulated_argument_loss = 0.0
                accumulated_realis_loss = 0.0
                accumulated_type_loss = 0.0
                accumulated_anaphoricity_loss = 0.0

            if tf_global_step > 0 and tf_global_step % eval_frequency == 0:  # and tf_global_step > 2000:
                saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                path = os.path.join(log_dir, "model-" + str(tf_global_step))
                scores = model.evaluate(session, path + "-output")

                eval_f1 = 0
                if config["add_coref_loss"]:
                    for score in scores['coref']:
                        print(score)
                        if score[0] == 'avg':  # 'ave_f':
                            eval_f1 = float(score[-1])
                elif config["add_subtype_loss"]:
                    score = scores["mention_type"]
                    print("subtype:", score)
                    eval_f1 = score[2]
                elif config["add_realis_loss"]:
                    score = scores["realis_status"]
                    print("realis", score)
                    eval_f1 = score[2]
                elif config["add_anaphoricity_loss"]:
                    score = scores["anaphoricity"]
                    print("anaphoricity score:", score)
                    eval_f1 = score[2]
                    logger.info(
                        "[{}] p={:.4f}, r={:.4f}, f1={:.4f}".format(tf_global_step, score[0], score[1], score[2]))
                elif config["add_span_argument_loss"]:
                    score = scores["argument"]
                    print("argument:", score)
                    # print("arg wo subtype:", score["argument_wo_subtype"])
                    eval_f1 = score[2]
                elif config["add_modality_loss"] or config["add_tense_loss"] or config["add_genericity_loss"] or config[
                    "add_polarity_loss"]:
                    score = scores["attribute"]
                    # print("attribute:", score)
                    # input(" ")
                    eval_f1 = score[2]
                else:
                    eval_f1 = 0

                if eval_f1 > max_f1:
                    max_f1 = eval_f1
                    max_step = tf_global_step

                    if eval_f1 > 47.0:
                        util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)),
                                             os.path.join(log_dir, "model.max.ckpt"))

                # writer.add_summary(eval_summary, tf_global_step)
                writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

                logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}, {}".format(tf_global_step, eval_f1, max_f1, max_step))
