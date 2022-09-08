from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import random
import threading

import numpy as np
import tensorflow as tf

import coref_ops
import coref_scorer
import evaluate_scorer
import metrics
import optimization
import util
from bert import modeling
from bert import tokenization
from pytorch_to_tf import load_from_pytorch_checkpoint


class CorefModel(object):
    def __init__(self, config):
        self.config = config
        self.max_segment_len = config['max_segment_len']
        self.max_span_width = config["max_span_width"]
        self.subtoken_maps = {}
        self.gold = {}
        self.eval_data = None  # Load eval data lazily.
        self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=config['vocab_file'], do_lower_case=False)
        self.subtype_dict, self.type_dict, self.subtype_type_map = util.load_subtype_dict(config["event_subtype_list"])
        self.arg_role_dict = util.load_arg_role_dict(config["arg_role_path"])
        self.realis_dict = {"NULL": 0, "actual": 1, "other": 2, "generic": 3, "unknown": 4}
        self.dev_data = None
        self.test_data = None

        input_props = []
        input_props.append((tf.int32, [None, None]))  # input_ids.
        input_props.append((tf.int32, [None, None]))  # input_mask
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.bool, []))  # Is training coref.
        input_props.append((tf.int32, [None]))  # Gold starts.
        input_props.append((tf.int32, [None]))  # Gold ends.
        input_props.append((tf.int32, [None]))  # Cluster ids.
        input_props.append((tf.int32, [None]))  # Sentence Map
        input_props.append((tf.int32, [None]))  # subtypes.
        input_props.append((tf.int32, [None]))  # types.
        input_props.append((tf.int32, [None]))  # realis.
        input_props.append((tf.int32, [None]))  # Gold Arg mention starts.
        input_props.append((tf.int32, [None]))  # Gold Arg mention ends.
        input_props.append((tf.int32, [None]))  # Gold Arg starts
        input_props.append((tf.int32, [None]))  # Gold Arg ends.
        input_props.append((tf.int32, [None]))  # Gold role ids.
        input_props.append((tf.int32, [None]))  # Gold anaphoricity.
        input_props.append((tf.bool, []))  # Not E73,94, if the doc is from corpus E73, 94, false
        input_props.append((tf.int32, [None]))  # subtype type mapping.

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)

        # bert stuff
        tvars = tf.trainable_variables()
        # If you're using TF weights only, tf_checkpoint and init_checkpoint can be the same
        # Get the assignment map from the tensorflow checkpoint. Depending on the extension, use TF/Pytorch to load weights.
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, config[
            'tf_checkpoint'])
        init_from_checkpoint = tf.train.init_from_checkpoint if config['init_checkpoint'].endswith(
            'ckpt') else load_from_pytorch_checkpoint
        init_from_checkpoint(config['init_checkpoint'], assignment_map)
        print("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            # init_string)
            print("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))

        num_train_steps = int(
            self.config['num_docs'] * self.config['num_epochs'])
        num_warmup_steps = int(num_train_steps * 0.1)
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = optimization.create_custom_optimizer(tvars,
                                                             self.loss, self.config['bert_learning_rate'],
                                                             self.config['task_learning_rate'],
                                                             num_train_steps, num_warmup_steps, False, self.global_step,
                                                             freeze=-1,
                                                             task_opt=self.config['task_optimizer'],
                                                             eps=config['adam_eps'])

    def start_enqueue_thread(self, session):
        with open(self.config["train_path"]) as f:
            # train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
            jsonlines = f.readlines()
            total_num_docs = len(jsonlines)
            num_docs = int(round(total_num_docs * self.config["lc_ratio"]))
            num_docs_train_coref = int(round(num_docs * self.config["coref_ratio"]))
            is_training_coref_map = {}
            train_examples = []
            for i in range(0, num_docs):
                jsonline = jsonlines[i]
                example = json.loads(jsonline)
                train_examples.append(example)

                if i <= num_docs_train_coref:
                    is_training_coref = True
                else:
                    is_training_coref = False
                is_training_coref_map[example["doc_key"]] = is_training_coref

            print("Loaded {} train examples. lc_ratio {}. total_num_docs {}, num_docs_train_coref {}".format(
                len(train_examples), self.config["lc_ratio"], total_num_docs, num_docs_train_coref))

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                if self.config['single_example']:
                    for example in train_examples:
                        # print(example["doc_key"])
                        is_training_coref = is_training_coref_map[example["doc_key"]]
                        tensorized_example = self.tensorize_example(example, is_training=True,
                                                                    is_training_coref=is_training_coref)
                        feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                        session.run(self.enqueue_op, feed_dict=feed_dict)
                else:
                    examples = []
                    for example in train_examples:
                        tensorized = self.tensorize_example(example, is_training=True, is_training_coref=True)
                        if type(tensorized) is not list:
                            tensorized = [tensorized]
                        examples += tensorized
                    random.shuffle(examples)
                    print('num examples', len(examples))
                    for example in examples:
                        feed_dict = dict(zip(self.queue_input_tensors, example))
                        session.run(self.enqueue_op, feed_dict=feed_dict)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def restore(self, session, checkpoint_path):
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables()]
        saver = tf.train.Saver(vars_to_restore)

        print("log_dir:", self.config["log_dir"])

        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends, _ = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def tensorize_example(self, example, is_training, is_training_coref):
        not_E7394 = True
        if "LDC2015E94" in example['doc_key'] or "LDC2015E73" in example['doc_key']:
            not_E7394 = False

        event_mention_map = {}
        for c in example["gold_clusters"]:
            for m in c:
                start, end, mtype = m
                if "_" in mtype:
                    event_mention_map[(start, end)] = mtype

        if self.config["train_wo_subtype"]:
            all_clusters = []
            for cluster in example["gold_clusters"]:
                new_cluster = []
                for m in cluster:
                    if "_" in m[2]:
                        new_cluster.append((m[0], m[1], "event"))
                    else:
                        new_cluster.append((m[0], m[1], "entity"))
                all_clusters.append(new_cluster)
        else:
            all_clusters = []  # remove overlapped trigger and entity mention
            for cluster in example["gold_clusters"]:
                new_cluster = []
                for m in cluster:
                    if "_" in m[2]:
                        new_cluster.append(m)
                    else:
                        if (m[0], m[1]) in event_mention_map:
                            # print(m[0], m[1], m[2], event_mention_map[(m[0], m[1])])
                            continue
                        else:
                            new_cluster.append(m)
                all_clusters.append(new_cluster)

        all_mentions = sorted(tuple(m) for m in util.flatten(all_clusters))

        if self.config["add_entity_coref_only"]:
            gold_mentions = []
            for mention in all_mentions:
                if (mention[2] == "entity" and self.config["train_wo_subtype"]) or (
                        "_" not in mention[2] and not self.config["train_wo_subtype"]):
                    gold_mentions.append(mention)

            clusters = []
            for c in all_clusters:
                if len(c) > 0 and (("_" not in c[0][2] and not self.config["train_wo_subtype"]) or (
                        c[0][2] == "entity" and self.config["train_wo_subtype"])):
                    clusters.append(c)
        else:
            if self.config["add_entity_coref"]:
                gold_mentions = all_mentions
                clusters = all_clusters
            else:
                gold_mentions = []
                if self.config["train_wo_subtype"]:
                    for mention in all_mentions:
                        if mention[2] == "event":
                            gold_mentions.append(mention)
                else:
                    for mention in all_mentions:
                        if "_" in mention[2]:
                            gold_mentions.append(mention)

                clusters = []
                for c in all_clusters:
                    if len(c) > 0 and ("_" in c[0][2] or c[0][2] == "event"):
                        clusters.append(c)

        if self.config["pipelined_subtypes"] and not is_training:
            print("evaluating coref using predicted subtypes")
            gold_mentions = []
            for m, mtype in zip(example["top_spans"], example["top_spans_subtypes"]):
                if mtype == "null":
                    continue

                gold_mentions.append((str(m[0]), str(m[1]), mtype))

            gold_mentions = sorted(gold_mentions)
            clusters = [gold_mentions]

        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        gold_anaphoricity = np.zeros(len(gold_mentions))

        # remove overlapped mentions
        mention_set = set()
        span_cluster_map = {}
        for cluster_id, cluster in enumerate(clusters):
            for mid, mention in enumerate(sorted(cluster)):

                if (mention[0], mention[1]) in mention_set:
                    continue

                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
                mention_set.add((mention[0], mention[1]))
                span_cluster_map[(mention[0], mention[1])] = cluster_id

                if mid > 0:
                    gold_anaphoricity[gold_mention_map[tuple(mention)]] = 1

                # print(cluster_id, mid, mention, gold_anaphoricity[gold_mention_map[tuple(mention)]])

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)

        sentence_map = example['sentence_map']

        max_sentence_length = self.max_segment_len
        text_len = np.array([len(s) for s in sentences])

        input_ids, input_mask = [], []
        for i, sentence in enumerate(sentences):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            sent_input_mask = [1] * len(sent_input_ids)
            while len(sent_input_ids) < max_sentence_length:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        doc_key = example["doc_key"]
        self.subtoken_maps[doc_key] = example.get("subtoken_map", None)
        self.gold[doc_key] = example["gold_clusters"]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        gold_types = []
        gold_subtypes = []
        span_set = set()
        for mention in gold_mentions:
            m_start, m_end, m_type = mention

            if (m_start, m_end) in span_set:  # handle multi-subtypes at the same offset
                gold_subtypes.append(0)
                gold_types.append(0)
            else:
                gold_subtypes.append(self.subtype_dict[m_type])
                gold_types.append(self.type_dict[m_type.split("_")[0]])
                span_set.add((m_start, m_end))

        gold_subtypes = np.array(gold_subtypes)
        gold_types = np.array(gold_types)

        gold_realis = []
        if self.config["add_realis_loss"]:
            gold_realis_map = {}

            for mention in example["gold_event_mentions"]:
                mstart, mend, _, mrealis = mention
                gold_realis_map[(mstart, mend)] = self.realis_dict[mrealis]

            span_set = set()
            for mention in gold_mentions:
                if (mention[0], mention[1]) in span_set:  # handle multi-subtypes at same offset
                    gold_realis.append(0)
                else:
                    if (mention[0], mention[1]) in gold_realis_map:
                        gold_realis.append(gold_realis_map[(mention[0], mention[1])])
                        span_set.add((mention[0], mention[1]))
                    else:
                        gold_realis.append(
                            self.realis_dict["unknown"])  # entity mention will be assigned "unknown" for realis
                        span_set.add((mention[0], mention[1]))

            gold_realis = np.array(gold_realis)
        else:
            gold_realis = np.zeros_like(gold_subtypes)

        gold_args = []
        if self.config["add_span_argument_loss"]:
            gold_args = [tuple(arg) for arg in example["gold_arguments"]]

        if len(gold_args) == 0:
            gold_arg_mstart = []
            gold_arg_mend = []
            gold_arg_pstart = []  # arg phrase start
            gold_arg_pend = []  # arg phrase end
            gold_arg_hstart = []  # arg head start
            gold_arg_hend = []  # arg head end
            gold_arg_roles_text = []
        else:
            gold_arg_mstart, gold_arg_mend, gold_arg_pstart, gold_arg_pend, gold_arg_hstart, gold_arg_hend, \
            gold_arg_roles_text, _ = zip(*gold_args)
            gold_arg_mstart = list(gold_arg_mstart)
            gold_arg_mend = list(gold_arg_mend)
            gold_arg_pstart = list(gold_arg_pstart)
            gold_arg_pend = list(gold_arg_pend)
            gold_arg_hstart = list(gold_arg_hstart)
            gold_arg_hend = list(gold_arg_hend)
            gold_arg_roles_text = list(gold_arg_roles_text)

        gold_arg_start = gold_arg_pstart
        gold_arg_end = gold_arg_pend

        gold_arg_roles = []
        arg_set = set()
        for mstart, mend, start, end, role in zip(gold_arg_mstart, gold_arg_mend, gold_arg_start, gold_arg_end,
                                                  gold_arg_roles_text):
            if (mstart, mend, start, end) in arg_set:
                gold_arg_roles.append(0)
                # print(mstart, mend, start, end, role)
            else:
                gold_arg_roles.append(self.arg_role_dict[role.upper()])
                arg_set.add((mstart, mend, start, end))

        gold_arg_mstart = np.array(gold_arg_mstart)
        gold_arg_mend = np.array(gold_arg_mend)
        gold_arg_start = np.array(gold_arg_start)
        gold_arg_end = np.array(gold_arg_end)
        gold_arg_roles = np.array(gold_arg_roles)

        example_tensors = (
            input_ids, input_mask, text_len, is_training, is_training_coref, gold_starts, gold_ends, cluster_ids,
            sentence_map, gold_subtypes, gold_types, gold_realis,
            gold_arg_mstart, gold_arg_mend, gold_arg_start, gold_arg_end, gold_arg_roles, gold_anaphoricity, not_E7394,
            self.subtype_type_map
        )

        if is_training and len(sentences) > self.config["max_training_sentences"]:
            if self.config['single_example']:
                return self.truncate_example(*example_tensors)
            else:
                offsets = range(self.config['max_training_sentences'], len(sentences),
                                self.config['max_training_sentences'])
                tensor_list = [self.truncate_example(*(example_tensors + (offset,))) for offset in offsets]
                return tensor_list
        else:
            return example_tensors

    def truncate_example(self, input_ids, input_mask, text_len, is_training, is_training_coref, gold_starts, gold_ends,
                         cluster_ids, sentence_map, gold_subtypes, gold_types, gold_realis,
                         gold_arg_mstart, gold_arg_mend, gold_arg_start, gold_arg_end, gold_arg_roles,
                         gold_anaphoricity, not_E7394, subtype_type_map,
                         sentence_offset=None):
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0,
                                         num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]
        gold_subtypes = gold_subtypes[gold_spans]
        gold_types = gold_types[gold_spans]
        gold_realis = gold_realis[gold_spans]
        gold_anaphoricity = gold_anaphoricity[gold_spans]

        gold_arg_mspans = np.logical_and(gold_arg_mend >= word_offset, gold_arg_mstart < word_offset + num_words)

        gold_arg_mstart = gold_arg_mstart[gold_arg_mspans] - word_offset
        gold_arg_mend = gold_arg_mend[gold_arg_mspans] - word_offset
        gold_arg_start = gold_arg_start[gold_arg_mspans] - word_offset
        gold_arg_end = gold_arg_end[gold_arg_mspans] - word_offset
        gold_arg_roles = gold_arg_roles[gold_arg_mspans]

        return input_ids, input_mask, text_len, is_training, is_training_coref, gold_starts, gold_ends, cluster_ids, sentence_map, gold_subtypes, gold_types, gold_realis, \
               gold_arg_mstart, gold_arg_mend, gold_arg_start, gold_arg_end, gold_arg_roles, gold_anaphoricity, not_E7394, subtype_type_map

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]

        candidate_mention_labels = tf.to_int32(tf.greater(candidate_labels, 0))

        return candidate_labels, candidate_mention_labels

    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_span_range = tf.range(k)  # [k]
        antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)  # [k, k]
        antecedents_mask = antecedent_offsets >= 1  # [k, k]
        fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores,
                                                                                             0)  # [k, k]
        fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask))  # [k, k]
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)  # [k, k]
        if self.config['use_prior']:
            antecedent_distance_buckets = self.bucket_distance(antecedent_offsets)  # [k, c]
            distance_scores = util.projection(tf.nn.dropout(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)), self.dropout), 1,
                initializer=tf.truncated_normal_initializer(stddev=0.02))  # [10, 1]
            antecedent_distance_scores = tf.gather(tf.squeeze(distance_scores, 1),
                                                   antecedent_distance_buckets)  # [k, c]
            fast_antecedent_scores += antecedent_distance_scores

        _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False)  # [k, c]
        top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents)  # [k, c]
        top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c]
        top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents)  # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def get_predictions_and_loss(self, input_ids, input_mask, text_len, is_training, is_training_coref, gold_starts,
                                 gold_ends, cluster_ids, sentence_map, gold_subtypes, gold_types, gold_realis,
                                 gold_arg_mstart, gold_arg_mend, gold_arg_start, gold_arg_end, gold_arg_roles,
                                 gold_anaphoricity, not_E7394, subtype_type_mapping):

        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=False,
            scope='bert')
        all_encoder_layers = model.get_all_encoder_layers()
        mention_doc = model.get_sequence_output()

        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

        num_sentences = tf.shape(mention_doc)[0]
        max_sentence_length = tf.shape(mention_doc)[1]
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)
        num_words = util.shape(mention_doc, 0)
        antecedent_doc = mention_doc

        flattened_sentence_indices = sentence_map
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1),
                                   [1, self.max_span_width])  # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width),
                                                           0)  # [num_words, max_span_width]
        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                     candidate_starts)  # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends,
                                                                                          num_words - 1))  # [num_words, max_span_width]
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                             candidate_end_sentence_indices))  # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]),
                                           flattened_candidate_mask)  # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]

        candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]),
                                                     flattened_candidate_mask)  # [num_candidates]

        candidate_cluster_ids, candidate_mention_labels = self.get_candidate_labels(candidate_starts, candidate_ends,
                                                                                    gold_starts, gold_ends,
                                                                                    cluster_ids)  # [num_candidates]
        candidate_subtypes, _ = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                          gold_subtypes)
        candidate_types, _ = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                       gold_types)
        candidate_realis, _ = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                        gold_realis)
        candidate_anaphoricity, _ = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                              gold_anaphoricity)

        candidate_span_emb = self.get_span_emb(mention_doc, mention_doc, candidate_starts,
                                               candidate_ends)  # [num_candidates, emb]

        candidate_mention_scores = self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)  # [k]

        # beam size
        k = tf.minimum(3900, tf.to_int32(tf.floor(tf.to_float(num_words) * self.config["top_span_ratio"])))
        # pull from beam

        if self.config["end_to_end"]:
            top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                       tf.expand_dims(candidate_starts, 0),
                                                       tf.expand_dims(candidate_ends, 0),
                                                       tf.expand_dims(k, 0),
                                                       num_words,
                                                       True)  # [1, k]
            top_span_indices.set_shape([1, None])
            top_span_indices = tf.squeeze(top_span_indices, 0)  # [k]
        else:
            k = util.shape(gold_starts, 0)
            top_span_indices = coref_ops.extract_spans(tf.expand_dims(tf.to_float(candidate_mention_labels), 0),
                                                       tf.expand_dims(candidate_starts, 0),
                                                       tf.expand_dims(candidate_ends, 0),
                                                       tf.expand_dims(k, 0),
                                                       num_words,
                                                       True)  # [1, k]
            top_span_indices.set_shape([1, None])
            top_span_indices = tf.squeeze(top_span_indices, 0)  # [k]

        top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)  # [k, emb]
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices)  # [k, emb]
        top_span_subtypes = tf.gather(candidate_subtypes, top_span_indices)
        self.top_span_gold_subtypes = top_span_subtypes

        if self.config["end_to_end"]:
            top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]
        else:
            top_span_mention_scores = tf.zeros_like(top_span_indices, dtype=tf.float32)  # [k]
            self.top_span_pred_subtypes = top_span_subtypes

        top_span_mention_labels = tf.gather(candidate_mention_labels, top_span_indices)  # [k]

        type_output = []
        self.type_loss = tf.constant(0.0)
        # if self.config["add_type_loss"]:
        #   type_null_scores = tf.zeros([k, 1])
        #   with tf.variable_scope("type_scores"):
        #     type_scores = util.ffnn(top_span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], len(self.type_dict), self.dropout) #[num_mentions, num_subtype,1]
        #     type_scores += tf.expand_dims(top_span_mention_scores, 1) # add mention score
        #   top_span_type_scores = tf.concat([type_null_scores, type_scores], 1)
        #   top_span_types = tf.gather(candidate_types, top_span_indices)

        #   self.top_span_pred_types = tf.to_int32(tf.argmax(top_span_type_scores, axis = 1)) #[k]

        #   if self.config["type_margin_loss"]:
        #     self.type_loss = self.trigger_margin_loss(top_span_type_scores, top_span_types, len(self.type_dict)+1, self.config["false_null"], self.config["false_subtype"])
        #   else:
        #     self.type_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = top_span_types, logits = top_span_type_scores)

        #   self.type_loss = tf.reduce_sum(self.type_loss)

        #   type_output = [top_span_type_scores, self.type_loss]

        subtype_output = []
        self.subtype_loss = tf.reduce_sum(tf.zeros([1]))
        if self.config["add_subtype_loss"]:
            subtype_null_scores = tf.zeros([k, 1])
            with tf.variable_scope("subtype_scores"):
                subtype_scores = util.ffnn(top_span_emb, self.config["ffnn_depth"], self.config["ffnn_size"],
                                           self.config["num_subtypes"] - 1,
                                           self.dropout)  # [num_mentions, num_subtype,1]
                subtype_scores += tf.expand_dims(top_span_mention_scores, 1)  # add mention score
            top_span_subtype_scores = tf.concat([subtype_null_scores, subtype_scores], 1)

            self.top_span_gold_subtypes = top_span_subtypes
            self.top_span_pred_subtypes = tf.to_int32(tf.argmax(top_span_subtype_scores, axis=1))  # [k]

            if self.config["subtype_margin_loss"]:
                self.subtype_loss = self.multilabel_margin_loss(top_span_subtype_scores, top_span_subtypes,
                                                                self.config["num_subtypes"], self.config["false_null"],
                                                                self.config["false_subtype"],
                                                                self.config["false_nonnull"])
            else:
                self.subtype_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=top_span_subtypes,
                                                                                   logits=top_span_subtype_scores)

            subtype_output = [top_span_subtype_scores, top_span_subtypes, self.subtype_loss]

        realis_output = []
        self.realis_loss = tf.reduce_sum(tf.zeros([1]))
        if self.config["add_realis_loss"]:
            realis_null_scores = tf.zeros([k, 1])
            with tf.variable_scope("realis_scores"):
                realis_scores = util.ffnn(top_span_emb, self.config["ffnn_depth"], self.config["ffnn_size"],
                                          self.config["num_realis"] - 1, self.dropout)
                realis_scores += tf.expand_dims(top_span_mention_scores, 1)  # add mention score

            top_span_realis_scores = tf.concat([realis_null_scores, realis_scores], 1)
            top_span_realis = tf.gather(candidate_realis, top_span_indices)

            if self.config["realis_margin_loss"]:
                self.realis_loss = self.multilabel_margin_loss(top_span_realis_scores, top_span_realis,
                                                               self.config["num_realis"],
                                                               self.config["realis_false_null"],
                                                               self.config["realis_false_type"],
                                                               self.config["realis_false_nonnull"])
            else:
                self.realis_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=top_span_realis,
                                                                                  logits=top_span_realis_scores)

            self.realis_loss = tf.reduce_sum(self.realis_loss)

            realis_output.append(top_span_realis_scores)
            realis_output.append(self.realis_loss)

        anaphoricity_output = []
        self.anaphoricity_loss = tf.constant(0.0)
        if self.config["add_anaphoricity_loss"]:
            top_span_ana_emb_list = [top_span_emb]

            context_range = tf.range(self.config["max_span_context_len"])
            span_context_left_offsets = tf.expand_dims(top_span_starts, 1) - tf.expand_dims(context_range,
                                                                                            0)  # [k, context_len]
            span_context_left_mask = span_context_left_offsets > -1  # [k, context_len]
            span_context_left_offsets = tf.maximum(0, span_context_left_offsets)
            span_context_left_emb = tf.gather(mention_doc, span_context_left_offsets)  # [k, context_len, emb]
            span_context_left_emb = tf.multiply(span_context_left_emb,
                                                tf.expand_dims(tf.to_float(span_context_left_mask),
                                                               2))  # [k, context_len, emb]
            span_context_left_emb = tf.reduce_sum(span_context_left_emb, 1)  # [k, emb]
            span_context_left_len = tf.reduce_sum(tf.to_float(span_context_left_mask), 1)  # [k]
            span_context_left_len = tf.maximum(1.0, span_context_left_len)  # avoid 0 #[k]
            span_context_left_emb = tf.divide(span_context_left_emb, tf.expand_dims(span_context_left_len,
                                                                                    1))  # [k, emb] divide by context len
            top_span_ana_emb_list.append(span_context_left_emb)

            top_span_ana_emb = tf.concat(top_span_ana_emb_list, 1)  # [k, emb]

            with tf.variable_scope("anaphoric_scores"):
                anaphoric_scores = util.ffnn(top_span_ana_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                             self.dropout)  # [k, 1]

            top_span_anaphoricity = tf.gather(candidate_anaphoricity, top_span_indices)

            if self.config["anaphoricity_margin_loss"]:
                self.anaphoricity_loss = self.binary_margin_loss(anaphoric_scores, top_span_anaphoricity,
                                                                 self.config["ana_false_null"],
                                                                 self.config["ana_false_ana"])
            else:
                self.anaphoricity_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=top_span_anaphoricity,
                                                                                        logits=anaphoric_scores)

            self.anaphoricity_loss = tf.reduce_sum(self.anaphoricity_loss)

            anaphoricity_output.append(anaphoric_scores)

        self.argument_output = []
        self.arg_loss = tf.constant(0.0)
        if self.config["add_span_argument_loss"]:
            a = tf.minimum(self.config["max_top_arguments"], k)
            pruning_subtypes = tf.cond(is_training, self.get_gold_subtypes, self.get_pred_subtypes)
            pruning_subtype_embs = tf.one_hot(pruning_subtypes, self.config["num_subtypes"])

            top_argument, top_argument_mask, top_argument_offsets, top_fast_arg_scores = self.coarse_to_fine_span_argument_pruning( \
                top_span_emb, top_span_mention_scores, top_span_sentence_indices, pruning_subtypes, a)  # [k, a]

            top_argument_emb = tf.gather(top_span_emb, top_argument)
            top_argument_subtype_emb = tf.gather(pruning_subtype_embs, top_argument)
            top_argument_emb = tf.concat([top_argument_emb, top_argument_subtype_emb], 2)

            target_mention_emb = tf.expand_dims(tf.concat([top_span_emb, pruning_subtype_embs], 1), 1)  # [k, 1, emb]
            target_mention_emb = tf.tile(target_mention_emb, [1, a, 1])  # [k, a, emb]

            mention_arg_pair_emb = tf.concat([target_mention_emb, top_argument_emb], 2)  # [k, a, emb]

            mention_arg_pair_emb = tf.reshape(mention_arg_pair_emb, [k * a, util.shape(mention_arg_pair_emb, -1)])
            arg_starts = tf.gather(top_span_starts, top_argument)  # [k, a]
            arg_ends = tf.gather(top_span_ends, top_argument)

            self.top_arg_labels, flatten_top_arg_labels, flatten_top_arg_mstarts, flatten_top_arg_mends, flatten_top_arg_starts, flatten_top_arg_ends = self.get_argument_labels( \
                top_span_starts, top_span_ends, arg_starts, arg_ends, top_argument_mask, \
                gold_arg_mstart, gold_arg_mend, gold_arg_start, gold_arg_end, gold_arg_roles)  # [k*a]

            flatten_arg_mask = tf.reshape(top_argument_mask, [k * a])  # [k*a]

            # for NULL type, score is 0, for other types, score = mention score + role score
            target_mention_scores = tf.expand_dims(top_span_mention_scores, 1)  # [k, 1]
            top_argument_mention_scores = tf.gather(top_span_mention_scores, top_argument)  # [k, a]
            top_arg_scores = target_mention_scores + top_argument_mention_scores  # [k, a]
            top_arg_scores = tf.reshape(top_arg_scores, [k * a, 1])  # [k*a, 1]
            with tf.variable_scope("argument_scores"):
                top_arg_pair_scores = util.ffnn(mention_arg_pair_emb, self.config["ffnn_depth"],
                                                self.config["ffnn_size"], len(self.arg_role_dict) - 1,
                                                self.dropout)  # [num_mentions, num_subtype,1]
                top_arg_scores += top_arg_pair_scores  # [k*a, num, num_roles - 1]

            null_scores = tf.zeros([util.shape(mention_arg_pair_emb, 0), 1])
            top_arg_scores = tf.concat([null_scores, top_arg_scores], 1)  # [k*a, num_roles]
            self.top_arg_pred_labels = tf.to_int32(tf.argmax(top_arg_scores, axis=1))  # [k*a]
            self.top_arg_pred_labels = tf.reshape(self.top_arg_pred_labels, [k, a])

            top_arg_scores = tf.boolean_mask(top_arg_scores, flatten_arg_mask)  # [num arg, 1]

            if self.config["arg_margin_loss"]:
                self.span_arg_loss = self.multilabel_margin_loss(top_arg_scores, flatten_top_arg_labels,
                                                                 len(self.arg_role_dict), self.config["arg_false_null"],
                                                                 self.config["arg_false_role"],
                                                                 self.config["arg_false_nonnull"])
            else:
                self.span_arg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flatten_top_arg_labels,
                                                                                    logits=top_arg_scores)

            self.span_arg_loss = tf.reduce_sum(self.span_arg_loss)
            self.argument_output = [flatten_top_arg_mstarts, flatten_top_arg_mends, flatten_top_arg_starts,
                                    flatten_top_arg_ends, \
                                    top_arg_scores, flatten_top_arg_labels]

            def get_span_argument_loss():
                return self.span_arg_loss

            def get_zero_arg_loss():
                return tf.constant(0.0)

            self.arg_loss = tf.cond(not_E7394, get_span_argument_loss, get_zero_arg_loss)
            self.argument_output.append(self.arg_loss)

        self.coref_loss = tf.constant(0.0)
        coref_output = []
        if self.config["add_coref_loss"]:
            c = tf.minimum(self.config["max_top_antecedents"], k)

            dummy_scores = tf.zeros([k, 1])  # [k, 1]
            if self.config["add_anaphoricity_loss"]:
                dummy_scores = tf.multiply(-1 * self.config["ana_constraint_loss_weight"],
                                           anaphoric_scores)  # tf.slice(anaphoric_scores, [0, 0], [k, 1])
            else:
                ##dummy_scores = anaphoric_scores
                dummy_scores = tf.zeros([k, 1])  # [k, 1]

            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(
                top_span_emb, top_span_mention_scores, c)

            if self.config["use_same_subtype_filter"]:
                top_span_subtypes = tf.cond(is_training, self.get_gold_subtypes, self.get_pred_subtypes)
                top_antecedent_subtypes = tf.gather(top_span_subtypes, top_antecedents)  # [k, c]
                same_subtypes = tf.equal(tf.expand_dims(top_span_subtypes, 1), top_antecedent_subtypes)  # [k, c]
                top_antecedents_mask = tf.logical_and(top_antecedents_mask, same_subtypes)
                top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask))  # [k, k]

            num_segs, seg_len = util.shape(input_ids, 0), util.shape(input_ids, 1)
            word_segments = tf.tile(tf.expand_dims(tf.range(0, num_segs), 1), [1, seg_len])
            flat_word_segments = tf.boolean_mask(tf.reshape(word_segments, [-1]), tf.reshape(input_mask, [-1]))
            mention_segments = tf.expand_dims(tf.gather(flat_word_segments, top_span_starts), 1)  # [k, 1]
            antecedent_segments = tf.gather(flat_word_segments, tf.gather(top_span_starts, top_antecedents))  # [k, c]
            segment_distance = tf.clip_by_value(mention_segments - antecedent_segments, 0,
                                                self.config['max_training_sentences'] - 1) if self.config[
                'use_segment_distance'] else None  # [k, c]

            if self.config["use_subtype_emb"]:
                top_span_subtype_fea = tf.cond(is_training, self.get_gold_subtypes, self.get_pred_subtypes)
                top_span_subtype_emb = tf.one_hot(top_span_subtype_fea, self.config["num_subtypes"])
                top_span_emb = tf.concat([top_span_emb, top_span_subtype_emb], 1)

            if self.config['fine_grained']:
                for i in range(self.config["coref_depth"]):
                    with tf.variable_scope("coref_layer", reuse=(i > 0)):
                        top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]

                        if self.config["use_pred_subtype_fea"]:
                            top_span_subtypes = self.top_span_pred_subtypes
                        else:
                            top_span_subtypes = tf.cond(is_training, self.get_gold_subtypes, self.get_pred_subtypes)

                        top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(
                            top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                            top_span_subtypes, segment_distance)  # [k, c]

                        if self.config["add_violation_scores"]:
                            top_antecedent_scores -= self.config[
                                                         "constraint_loss_weight"] * self.get_coref_violation_scores(
                                top_antecedents, top_span_subtype_scores)
                        elif self.config["add_violation_scores_2"]:
                            if self.config["add_argument_violation_scores"]:
                                def get_gold_arg_label():
                                    return self.top_arg_labels

                                def get_pred_arg_label():
                                    return self.top_arg_pred_labels

                                arg_labels = tf.cond(is_training, get_gold_arg_label, get_pred_arg_label)

                                if self.config["eval_with_gold_argument"]:
                                    arg_labels = self.top_arg_labels

                                top_antecedent_pred_scores = tf.reduce_max(
                                    tf.concat([dummy_scores, top_antecedent_scores], 1), axis=1)  # [k]
                                arg_violation_scores = coref_ops.extract_important_argument_pairs_violation_scores(
                                    top_argument, arg_labels, tf.to_int32(top_argument_mask), \
                                    top_antecedent_scores, top_antecedents, tf.to_int32(top_antecedents_mask),
                                    top_antecedent_pred_scores, k, a)
                                top_antecedent_scores -= self.config[
                                                             "arg_constraint_loss_weight"] * arg_violation_scores

                            # top_antecedent_scores -= self.config["constraint_loss_weight"] * self.get_coref_violation_scores_full(top_antecedents, top_span_subtype_scores, top_span_realis_scores)
                            # subtype mismatch
                            if self.config["add_subtype_loss"]:
                                subtype_mismatch_score = self.get_coref_violation_scores_2(top_antecedents,
                                                                                           top_span_subtype_scores)
                                top_antecedent_scores -= self.config[
                                                             "subtype_constraint_loss_weight"] * subtype_mismatch_score

                            if self.config["add_type_loss"]:
                                type_mismatch_score = self.get_coref_violation_scores_2(top_antecedents,
                                                                                        top_span_type_scores)
                                top_antecedent_scores -= self.config[
                                                             "type_constraint_loss_weight"] * type_mismatch_score

                            # realis_mismatch_score = tf.constant(0.0)
                            if self.config["add_realis_loss"]:
                                # realis mismatch
                                realis_mismatch_score = self.get_coref_violation_scores_2(top_antecedents,
                                                                                          top_span_realis_scores)
                                top_antecedent_scores -= self.config[
                                                             "realis_constraint_loss_weight"] * realis_mismatch_score

                            # if self.config["add_anaphoricity_violation_scores"]:
                            #   # if dummy score > 0 => predicted as not anaphoric, ant(i, j) < dummy score. If (ant(i-j) - dummy > 0), violated
                            #   # if dummy score < 0 => predicted as anaphoric, ant(i, j) > dummy score. if -1 * (ant(i-j) - dummy) > 0, violated
                            #   pred_non_anaphoricity = tf.where(anaphoric_scores > 0, tf.ones_like(anaphoric_scores), tf.zeros_like(anaphoric_scores)-1) # [k] # if anaphoric, 1, otherwise -1
                            #   top_antecedent_pred_scores = tf.reduce_max(top_antecedent_scores, axis = 1) #[k]

                            #   anaphoricity_violation_score = top_antecedent_scores - anaphoric_scores
                            #   anaphoricity_violation_score = tf.multiply(pred_non_anaphoricity, anaphoricity_violation_score)
                            #   anaphoricity_violation_score = tf.where(anaphoricity_violation_score>0, anaphoricity_violation_score, tf.zeros_like(anaphoricity_violation_score))
                            #   print(util.shape(anaphoricity_violation_score, 1))
                            #   top_antecedent_scores -= anaphoricity_violation_score
                            #   print(util.shape(top_antecedent_scores, 1))

                            # if self.config["add_anaphoricity_violation_scores"]:
                            #   pred_anaphoricity = tf.where(anaphoric_scores > 0, tf.ones_like(anaphoric_scores), tf.zeros_like(anaphoric_scores)-1) # [k] # if anaphoric, 1, otherwise -1
                            #   top_antecedent_pred_scores = tf.reduce_max(top_antecedent_scores, axis = 1) #[k]

                        top_antecedent_weights = tf.nn.softmax(
                            tf.concat([dummy_scores, top_antecedent_scores], 1))  # [k, c + 1]
                        top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb],
                                                       1)  # [k, c + 1, emb]
                        attended_span_emb = tf.reduce_sum(
                            tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1)  # [k, emb]
                        with tf.variable_scope("f"):
                            f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1),
                                                           util.shape(top_span_emb, -1)))  # [k, emb]
                            top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]
            else:
                top_antecedent_scores = top_fast_antecedent_scores

            top_antecedent_scores_full = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]
            top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
            top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c]
            same_cluster_indicator = tf.equal(top_antecedent_cluster_ids,
                                              tf.expand_dims(top_span_cluster_ids, 1))  # [k, c]
            non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
            pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]
            dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
            top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]

            if self.config['margin_loss']:
                if not self.config['anaphoricity_hard_constraint']:
                    latent_score, margin, loss = self.margin_loss(top_antecedent_scores_full, top_antecedent_labels,
                                                                  top_antecedents_mask)
                else:
                    # use anaphoricity prediction as hard constraint
                    pred_anaphoricity = anaphoric_scores > 0
                    anaphoric_top_antencedent_scores = tf.boolean_mask(top_antecedent_scores, pred_anaphoricity)
                    anaphoric_pairwise_labels = tf.boolean_mask(pairwise_labels, pred_anaphoricity)
                    _, _, loss = self.pairwise_margin_loss(anaphoric_top_antencedent_scores,
                                                           anaphoric_pairwise_labels)  # [num anaphoric mentions]


            else:
                loss = self.softmax_loss(top_antecedent_scores_full, top_antecedent_labels)  # [k]

            # if self.config["add_subtype_loss"]:
            #   entity_mask = top_span_subtypes > 21 #[k]
            #   self.entity_coref_loss = tf.boolean_mask(loss, entity_mask)
            #   self.entity_coref_loss = tf.reduce_sum(self.entity_coref_loss)

            #   self.coref_loss = tf.reduce_sum(loss)

            #   #loss = tf.cond(is_training_coref, self.get_coref_loss, self.get_empty_coref_loss)
            #   loss = tf.cond(is_training_coref, self.get_coref_loss, self.get_entity_coref_loss)

            # self.coref_loss = tf.reduce_sum(loss)
            self.coref_loss = loss
            coref_output = [top_antecedents, top_antecedent_scores_full]

        # entity_mask = top_span_subtypes > 21 #[k]
        # entity_weight = tf.where(entity_mask, tf.ones_like(top_span_subtypes, dtype=tf.float32)*self.config["entity_weight"], tf.ones_like(top_span_subtypes, dtype=tf.float32)*1.0)
        # self.subtype_loss = tf.multiply(tf.expand_dims(entity_weight, 1), tf.reduce_sum(self.subtype_loss, axis=1))

        def get_event_coref_loss():
            return self.config["event_weight"] * self.coref_loss

        def get_entity_coref_loss():
            return self.config["entity_weight"] * self.coref_loss

        def get_event_subtype_loss():
            return self.config["event_weight"] * self.subtype_loss

        def get_entity_subtype_loss():
            return self.config["entity_weight"] * self.subtype_loss

        event_mentions_mask = tf.logical_and(top_span_subtypes < 22, top_span_subtypes > 0)  # [k, 1]

        coref_loss = tf.where(event_mentions_mask, self.config["event_weight"] * self.coref_loss,
                              self.config["entity_weight"] * self.coref_loss)

        # self.subtype_loss = tf.reduce_sum(self.subtype_loss, axis=1)
        # subtype_loss = tf.where(event_mentions_mask, self.config["event_weight"] * self.subtype_loss, self.config["entity_weight"] * self.subtype_loss)

        E7394_mask = tf.logical_or(not_E7394, event_mentions_mask)  # [k]

        self.subtype_loss = tf.boolean_mask(self.subtype_loss, E7394_mask)
        self.subtype_loss = tf.reduce_sum(self.subtype_loss)

        self.coref_loss = tf.boolean_mask(coref_loss, E7394_mask)
        self.coref_loss = tf.reduce_sum(self.coref_loss)

        loss = tf.constant(0.0)
        loss += self.config['coref_loss_weight'] * self.coref_loss  # []
        loss += self.config["type_loss_weight"] * self.type_loss
        loss += self.config['subtype_loss_weight'] * self.subtype_loss
        loss += self.config['realis_loss_weight'] * self.realis_loss
        loss += self.config['anaphoricity_loss_weight'] * self.anaphoricity_loss
        loss += self.config['argument_loss_weight'] * self.arg_loss

        self.encoref_loss = tf.constant(0.0)
        self.en_subtype_loss = tf.constant(0.0)

        self.constraint_loss = tf.constant(0.0)

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_span_mention_scores, coref_output, \
                subtype_output, realis_output, anaphoricity_output, type_output], loss

    def get_coref_loss(self):
        return self.coref_loss

    def get_empty_coref_loss(self):
        return tf.constant(0.0)

    def get_entity_coref_loss(self):
        return self.entity_coref_loss

    def get_coref_violation_scores(self, top_antecedents, top_span_subtype_scores):
        top_span_subtypes_prediction = tf.argmax(top_span_subtype_scores, axis=1)  # [k]
        top_span_subtypes_prediction_scores = tf.reduce_max(top_span_subtype_scores, axis=1)  # [k]

        top_antecedents_subtypes = tf.gather(top_span_subtypes_prediction, top_antecedents)  # [k, c]
        top_antecedents_subtypes_scores = tf.gather(top_span_subtypes_prediction_scores, top_antecedents)  # [k, c]

        same_type = tf.equal(tf.expand_dims(top_span_subtypes_prediction, 1), top_antecedents_subtypes)  # [k, c]

        diff_type = tf.logical_not(same_type)  # difftype

        mention_pair_subtype_scores = tf.expand_dims(top_span_subtypes_prediction_scores,
                                                     1) + top_antecedents_subtypes_scores  # [k, c]

        violation_scores = tf.multiply(tf.to_float(diff_type), mention_pair_subtype_scores)  # [k, c]

        print("violation_scores shape:", violation_scores.shape)

        return violation_scores

    def get_coref_violation_scores_full(self, top_antecedents, top_span_subtype_scores, top_span_realis_scores):

        # subtype mismatch
        subtype_mismatch_score = self.get_coref_violation_scores_2(top_antecedents, top_span_subtype_scores)

        realis_mismatch_score = tf.constant(0.0)

        if self.config["add_realis_loss"]:
            # realis mismatch
            realis_mismatch_score = self.get_coref_violation_scores_2(top_antecedents, top_span_realis_scores)

        return subtype_mismatch_score + realis_mismatch_score

    def get_coref_violation_scores_2(self, top_antecedents, top_span_subtype_scores):

        # subtype
        # top_span_subtype_scores #[k, num_subtype]
        k = util.shape(top_span_subtype_scores, 0)
        c = util.shape(top_antecedents, 1)

        top_span_subtypes_prediction = tf.to_int32(tf.argmax(top_span_subtype_scores, axis=1))  # [k]
        top_span_subtypes_prediction_scores = tf.reduce_max(top_span_subtype_scores, axis=1)  # [k]

        top_antecedents_subtypes = tf.gather(top_span_subtypes_prediction, top_antecedents)  # [k, c]
        top_antecedents_subtypes_scores = tf.gather(top_span_subtypes_prediction_scores, top_antecedents)  # [k, c]

        same_type = tf.equal(tf.expand_dims(top_span_subtypes_prediction, 1), top_antecedents_subtypes)  # [k, c]
        diff_type = tf.logical_not(same_type)  # difftype

        # calculate the difference of subtype scores of two subtypes for anaphor (e1's subtype, e2's subtype) (measure the disagreement)
        row_indices = tf.tile(tf.expand_dims(tf.range(k), 1), [1, c])  # [k, c]
        row_indices = tf.reshape(row_indices, [k * c])

        col_indices = tf.reshape(tf.to_int32(top_antecedents_subtypes), [k * c])
        full_indices = tf.stack([row_indices, col_indices], axis=1)  # [k*c, 2]

        top_span_subtypes_prediction_scores_t2 = tf.gather_nd(top_span_subtype_scores, full_indices)  # [k*c]
        top_span_subtypes_prediction_scores_t2 = tf.reshape(top_span_subtypes_prediction_scores_t2, [k, c])

        top_span_subtype_scores_diff = tf.abs(
            tf.expand_dims(top_span_subtypes_prediction_scores, 1) - top_span_subtypes_prediction_scores_t2)  # [k,c]

        violation_scores = tf.multiply(tf.to_float(diff_type), top_span_subtype_scores_diff)  # [k, c]

        # calculate the difference of subtypes scores difference for antecedent
        row_indices = tf.reshape(top_antecedents, [k * c])  # [k*c]
        col_indices = tf.tile(tf.expand_dims(top_span_subtypes_prediction, 1), [1, c])  # [k, c]
        col_indices = tf.reshape(col_indices, [k * c])
        full_indices = tf.stack([row_indices, col_indices], axis=1)  # [k*c, 2]
        top_antecedents_t2 = tf.gather_nd(top_span_subtype_scores, full_indices)
        top_antecedents_t2 = tf.reshape(top_antecedents_t2, [k, c])

        top_ant_subtype_scores_diff = tf.abs(top_antecedents_subtypes_scores - top_antecedents_t2)

        violation_scores += tf.multiply(tf.to_float(diff_type), top_ant_subtype_scores_diff)  # [k, c]

        # calculate the difference of anaphor predicted subtype
        null_type_mask = top_span_subtypes_prediction < 1  # [k]
        null_type_mask = tf.tile(tf.expand_dims(tf.to_float(null_type_mask), 1), [1, c])

        top_span_subtypes_prediction_nonnull_scores = tf.reduce_max(
            tf.slice(top_span_subtype_scores, [0, 1], [k, util.shape(top_span_subtype_scores, 1) - 1]), axis=1)  # [k]
        top_span_subtypes_prediction_nonnull_scores = tf.tile(
            tf.expand_dims(top_span_subtypes_prediction_nonnull_scores, 1), [1, c])
        violation_scores += tf.multiply(null_type_mask, tf.tile(tf.expand_dims(top_span_subtypes_prediction_scores, 1),
                                                                [1, c]) - top_span_subtypes_prediction_nonnull_scores)
        # print("violation_scores shape:", violation_scores.shape)

        return violation_scores

    def get_gold_subtypes(self):
        return self.top_span_gold_subtypes

    def get_pred_subtypes(self):
        return self.top_span_pred_subtypes

    def get_argument_labels(self, span_starts, span_ends, arg_starts, arg_ends, arg_mask, \
                            gold_arg_mstart, gold_arg_mend, gold_arg_start, gold_arg_end, gold_arg_roles):
        num_mentions = util.shape(arg_starts, 0)
        num_arg = util.shape(arg_starts, 1)

        top_arg_mstart = tf.tile(tf.expand_dims(span_starts, 1), [1, num_arg])  # [num_mentions, num_args]
        top_arg_mend = tf.tile(tf.expand_dims(span_ends, 1), [1, num_arg])  # [num_mentions, num_args]

        top_arg_mstart = tf.reshape(top_arg_mstart,
                                    [num_mentions * num_arg])  # [num_mentions*num_args] [num_arg_candidates]
        top_arg_mend = tf.reshape(top_arg_mend, [num_mentions * num_arg])

        top_arg_starts = tf.reshape(arg_starts, [num_mentions * num_arg])
        top_arg_ends = tf.reshape(arg_ends, [num_mentions * num_arg])

        same_mstart = tf.equal(tf.expand_dims(gold_arg_mstart, 1),
                               tf.expand_dims(top_arg_mstart, 0))  # [num_labeled, num_candidates]
        same_mend = tf.equal(tf.expand_dims(gold_arg_mend, 1), tf.expand_dims(top_arg_mend, 0))
        same_astart = tf.equal(tf.expand_dims(gold_arg_start, 1), tf.expand_dims(top_arg_starts, 0))
        same_aend = tf.equal(tf.expand_dims(gold_arg_end, 1), tf.expand_dims(top_arg_ends, 0))

        same_arg = tf.logical_and(same_mstart, same_mend)
        same_arg = tf.logical_and(same_arg, same_astart)
        same_arg = tf.to_int32(tf.logical_and(same_arg, same_aend))

        candidate_labels = tf.matmul(tf.expand_dims(gold_arg_roles, 0), same_arg)
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_mentions*num_args]

        flatten_arg_mask = tf.reshape(arg_mask, [num_mentions * num_arg])
        flatten_candidate_labels = tf.boolean_mask(candidate_labels, flatten_arg_mask)  # [num_arg_candidates]
        flatten_top_arg_mstarts = tf.boolean_mask(top_arg_mstart, flatten_arg_mask)
        flatten_top_arg_mends = tf.boolean_mask(top_arg_mend, flatten_arg_mask)
        flatten_top_arg_starts = tf.boolean_mask(top_arg_starts, flatten_arg_mask)
        flatten_top_arg_ends = tf.boolean_mask(top_arg_ends, flatten_arg_mask)

        candidate_labels = tf.reshape(candidate_labels, [num_mentions, num_arg])  # [num_mentions, num_args]

        return candidate_labels, flatten_candidate_labels, flatten_top_arg_mstarts, flatten_top_arg_mends, \
               flatten_top_arg_starts, flatten_top_arg_ends

    def coarse_to_fine_span_argument_pruning(self, top_span_emb, top_span_mention_scores, top_span_sentence_indices,
                                             top_span_pruning_subtypes, a):
        # we only consider span pairs in the same sentence
        same_sent_ids = tf.equal(tf.expand_dims(top_span_sentence_indices, 1),
                                 tf.expand_dims(top_span_sentence_indices, 0))  # [k,k]

        # no self pair
        # filter out spans that are not in the same sentence
        k = util.shape(top_span_emb, 0)
        top_span_range = tf.range(k)  # [k]
        argument_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)  # [k, k]
        argument_mask = tf.not_equal(argument_offsets, tf.zeros_like(argument_offsets))  # [k, k]

        argument_mask = tf.logical_and(argument_mask, same_sent_ids)  # [k, k]

        # make sure there are only event - entity pairs (no event-event pairs, no entity-entity pairs, no entity-event pairs)
        event_only = tf.expand_dims(top_span_pruning_subtypes, 1)
        event_only_mask = event_only < 22  # event_only < 22 #tf.logical_and(event_only > 0, event_only < 22)
        entity_only = tf.expand_dims(top_span_pruning_subtypes, 0)
        event_entity_pairs = tf.logical_and(event_only_mask, entity_only > 21)

        argument_mask = tf.logical_and(argument_mask, event_entity_pairs)

        scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0)  # [k, k]
        scores += tf.log(tf.to_float(argument_mask))  # [k, k]
        scores += self.get_fast_argument_scores(top_span_emb)  # [k, k]

        _, top_argument = tf.nn.top_k(scores, a, sorted=False)  # [k, a]
        top_argument_mask = util.batch_gather(argument_mask, top_argument)  # [k, a]
        top_argument_offsets = util.batch_gather(argument_offsets, top_argument)  # [k, a]
        top_fast_scores = util.batch_gather(scores, top_argument)  # [k, a]

        return top_argument, top_argument_mask, top_argument_offsets, top_fast_scores

    def get_fast_argument_scores(self, top_span_emb):
        with tf.variable_scope("arg_src_projection"):
            source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)),
                                                self.dropout)  # [k, emb]
        target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout)  # [k, emb]
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True)  # [k, k]

    def binary_margin_loss(self, ana_scores, ana_labels, w_false_null, w_false_ana):
        ana_scores = tf.squeeze(ana_scores, 1)
        float_ana_labels = tf.to_float(ana_labels)
        delta = tf.where(float_ana_labels > 0.0, w_false_null * tf.ones_like(float_ana_labels),
                         w_false_ana * tf.ones_like(float_ana_labels))

        # make labels from 0 to -1，keep 1 as it is
        ana_labels = tf.where(ana_labels < 1, tf.zeros_like(ana_labels) - 1, ana_labels)

        # loss = max(0, 1 - t*y) t:label, y:score
        origin_margin = 1 - tf.multiply(tf.to_float(ana_labels), ana_scores)  # [k]
        print("origin_margin:", util.shape(origin_margin, -1))

        origin_margin = tf.where(origin_margin > 0, origin_margin, tf.zeros_like(origin_margin))  # [k]
        print("origin_margin:", util.shape(origin_margin, -1))

        origin_margin = tf.multiply(delta, origin_margin)

        return origin_margin

    def multilabel_margin_loss(self, scores, labels, num_labels, w_false_null, w_false_subtype, w_false_nonnull):
        # subtype_scores [num_mentions, num_subtypes]
        one_hot_labels = tf.one_hot(labels, num_labels)
        gold_scores = scores + tf.log(one_hot_labels)  # [num_mentions, num_subtypes]
        label_score = tf.reduce_max(gold_scores, [1], keepdims=True)  # [num_mentions]

        label_mask = one_hot_labels < 1  # [k, num_subtypes]

        origin_margin = 1 + scores - label_score
        # origin_margin = tf.multiply(self.error_delta(subtype_labels, num_labels, w_false_null, w_false_subtype), origin_margin)
        origin_margin = tf.multiply(
            util.error_delta_v2(labels, num_labels, w_false_null, w_false_subtype, w_false_nonnull), origin_margin)

        margin = tf.boolean_mask(origin_margin, label_mask)

        greater_mask = margin > 0

        margin_loss = tf.boolean_mask(margin, greater_mask)

        # margin_loss = tf.reduce_sum(margin)

        return margin_loss

    def error_delta(self, subtype_labels, num_labels, w_false_null, w_false_subtype):
        one_hot_subtype_labels = tf.one_hot(subtype_labels, num_labels)
        gold_index = one_hot_subtype_labels > 0

        gold_index = tf.to_float(tf.logical_not(gold_index))

        delta_0 = tf.zeros([util.shape(subtype_labels, 0), 1]) + w_false_null  # [num_mentions, 1]
        delta_1 = tf.zeros([util.shape(subtype_labels, 0), num_labels - 1]) + w_false_subtype  # [num_mentions, max_ant]

        delta = tf.concat([delta_0, delta_1], 1)  # [num_mentions, num_subtypes]
        delta = tf.multiply(delta, gold_index)  # [num_mentions, num_subtypes]
        return delta

    def margin_loss(self, antecedent_scores, antecedent_labels, antecedent_mask):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [num_mentions, max_ant + 1]
        latent_score = tf.reduce_max(gold_scores, axis=1, keepdims=True)  # [num_mentions, 1]

        margin = 1 + antecedent_scores - latent_score  # [num_mentions, max_ant + 1]
        # margin = tf.multiply(util.delta(antecedent_labels, antecedent_mask), margin) # [num_mentions, max_ant + 1] cost sensitive
        margin = tf.multiply(
            util.delta_v2(antecedent_labels, antecedent_mask, self.config["false_new"], self.config["false_link"],
                          self.config["wrong_link"]), margin)

        margin_loss = tf.reduce_max(margin, axis=1)  # [num_mentions]

        return latent_score, margin, margin_loss

    def pairwise_margin_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [num_mentions, max_ant]
        latent_score = tf.reduce_max(gold_scores, axis=1, keepdims=True)  # [num_mentions, 1]

        margin = 1 + antecedent_scores - latent_score  # [num_mentions, max_ant]
        # margin = tf.multiply(self.delta(antecedent_labels, antecedent_mask), margin) # [num_mentions, max_ant + 1] cost sensitive
        margin_loss = tf.reduce_max(margin, axis=1)  # [num_mentions]

        return latent_score, margin, margin_loss

    def delta(self, antecedent_labels, antecedent_mask):
        # antecedent_labels: [k, c+1]
        # antecedent_mask: [k, c]

        gold_index = tf.to_float(tf.logical_not(antecedent_labels))  # [num_mentions, max_ant+1]
        delta_0 = tf.zeros([util.shape(antecedent_labels, 0), 1]) + self.config["false_new"]  # [num_mentions, 1]
        delta_1 = tf.zeros([util.shape(antecedent_labels, 0), util.shape(antecedent_labels, 1) - 1]) + self.config[
            "false_link"]  # [num_mentions, max_ant]

        delta_1 = tf.multiply(delta_1, tf.to_float(antecedent_mask))  # [num_mentions, max_ant]

        delta = tf.concat([delta_0, delta_1], 1)  # [num_mentions, max_ant+1]
        delta = tf.multiply(delta, gold_index)  # [num_mentions, max_ant+1]
        return delta

    def compute_constraint_loss(self, top_span_subtype_scores, top_antecedents, top_antecedent_scores,
                                top_span_realis_scores):
        k = util.shape(top_antecedents, 0)
        c = util.shape(top_antecedents, 1)

        # subtype predictions
        # top_span_subtype_prob =  tf.nn.softmax(top_span_subtype_scores) #[k, num_subtypes]
        top_span_subtypes = tf.argmax(top_span_subtype_scores, axis=1)  # [k]

        # antecedent predictions
        # top_antecedent_prob = tf.nn.softmax(top_antecedent_scores) # [k, c + 1]
        top_ant_preds_index = tf.to_int32(tf.argmax(top_antecedent_scores, axis=1) - 1)  # [k]

        predicted_cluster_indices = coref_ops.get_predicted_cluster(top_antecedents, top_antecedent_scores, k,
                                                                    c)  # [k, 1]
        predicted_cluster_indices = tf.squeeze(predicted_cluster_indices, 1)  # [k]

        # RULE1: if not coref(e, dummy) -> not subtype(e, NULL)
        # if not coref(e, dummy)  and subtype(e, NULL) => violate
        rule1_violation = tf.logical_and(top_span_subtypes == 0, top_ant_preds_index > -1)
        rule1_violation_count = tf.reduce_sum(tf.to_int32(rule1_violation))

        # RULE2: If coref(e1, e2), subtype(e1, t1), subtype(e2, t2), t1 != t2 -> violation
        # if ant = -1, same ant = False
        top_span_same_cluster = tf.equal(tf.expand_dims(predicted_cluster_indices, 1),
                                         tf.expand_dims(predicted_cluster_indices, 0))  # [k, k]
        same_type = tf.equal(tf.expand_dims(top_span_subtypes, 1), tf.expand_dims(top_span_subtypes, 0))  # [k, k]

        rule2_violation = tf.logical_and(top_span_same_cluster, tf.logical_xor(top_span_same_cluster, same_type))
        rule2_violation_count = tf.reduce_sum(tf.to_int32(rule2_violation))

        if self.config["add_realis_loss"]:
            # realis predictions
            top_span_realis = tf.argmax(top_span_realis_scores, axis=1)  # [k]
            # RULE 3,  if realis(e, NULL) -> subtype(e, NULL)
            rule3_violation_1 = tf.logical_and(top_span_realis == 0, top_span_subtypes > 0)
            rule3_violation_2 = tf.logical_and(top_span_subtypes == 0, top_span_realis > 0)
            rule3_violation_count = tf.reduce_sum(tf.to_int32(rule3_violation_1)) + tf.reduce_sum(
                tf.to_int32(rule3_violation_2))

            # RULE4 coref(e1, e2), realis(e1, t1), realis(e2, t2), t1 != t2 -> violation
            same_realis = tf.equal(tf.expand_dims(top_span_realis, 1), tf.expand_dims(top_span_realis, 0))  # [k, k]
            rule4_violation = tf.logical_and(top_span_same_cluster, tf.logical_xor(top_span_same_cluster, same_realis))
            rule4_violation_count = tf.reduce_sum(tf.to_int32(rule4_violation))

            # RULE5 if not coref(e, dummy) -> not realis(e, NULL)
            rule5_violation = tf.logical_and(top_span_realis == 0, top_ant_preds_index > -1)
            rule5_violation_count = tf.reduce_sum(tf.to_int32(rule5_violation))
        else:

            rule3_violation_count = tf.constant(0.0)
            rule4_violation_count = tf.constant(0.0)
            rule5_violation_count = tf.constant(0.0)

        rule1_loss = self.config['rule1_weight'] * tf.to_float(rule1_violation_count)  # subtype
        rule2_loss = self.config['rule2_weight'] * tf.to_float(rule2_violation_count)  # subtype
        rule3_loss = self.config['rule3_weight'] * tf.to_float(rule3_violation_count)  # realis
        rule4_loss = self.config['rule4_weight'] * tf.to_float(rule4_violation_count)  # realis
        rule5_loss = self.config['rule5_weight'] * tf.to_float(rule5_violation_count)  # realis

        loss = (rule1_loss + rule2_loss + rule3_loss + rule4_loss + rule5_loss)

        constraint_output = [rule1_violation_count, rule2_violation_count]

        return loss, constraint_output

    def compute_constraint_loss_new(self, top_span_subtype_scores, top_antecedents, top_antecedent_scores,
                                    top_span_realis_scores):
        # top_span_subtypes_labels, top_span_realis_labels, top_antecedent_labels):
        k = util.shape(top_antecedents, 0)
        c = util.shape(top_antecedents, 1)

        # subtype predictions
        top_span_subtype_prob = tf.nn.softmax(top_span_subtype_scores)  # [k, num_subtypes]
        top_span_subtypes_prediction = tf.argmax(top_span_subtype_prob, axis=1)  # [k]
        top_span_subtypes_prediction_logprob = tf.log(tf.reduce_max(top_span_subtype_prob, axis=1))  # [k]

        # antecedent predictions
        top_antecedent_prob = tf.nn.softmax(top_antecedent_scores)  # [k, c + 1]
        top_ant_preds_index = tf.to_int32(tf.argmax(top_antecedent_scores, axis=1) - 1)  # [k]
        top_ant_preds_logprob = tf.log(tf.reduce_max(top_antecedent_prob, axis=1))  # [k]

        predicted_cluster_indices = coref_ops.get_predicted_cluster(top_antecedents, top_antecedent_scores, k,
                                                                    c)  # [k, 1]
        predicted_cluster_indices = tf.squeeze(predicted_cluster_indices, 1)  # [k]

        # RULE1: if not coref(e, dummy) -> not subtype(e, NULL)
        # if not coref(e, dummy)  and subtype(e, NULL) => violate
        rule1_violation = tf.logical_and(top_span_subtypes_prediction == 0, top_ant_preds_index > -1)  # [k]
        rule1_loss = tf.multiply(tf.to_float(rule1_violation),
                                 (top_span_subtypes_prediction_logprob + top_ant_preds_logprob))
        rule1_loss = tf.reduce_sum(rule1_loss)

        # RULE2: If coref(e1, e2), subtype(e1, t1), subtype(e2, t2), t1 != t2 -> violation
        # if ant = -1, same ant = False
        top_span_same_cluster = tf.equal(tf.expand_dims(predicted_cluster_indices, 1),
                                         tf.expand_dims(predicted_cluster_indices, 0))  # [k, k]
        same_type = tf.equal(tf.expand_dims(top_span_subtypes_prediction, 1),
                             tf.expand_dims(top_span_subtypes_prediction, 0))  # [k, k]

        rule2_violation = tf.logical_and(top_span_same_cluster,
                                         tf.logical_xor(top_span_same_cluster, same_type))  # [k, k]
        rule2_subtype_log_probs = tf.expand_dims(top_span_subtypes_prediction_logprob, 1) + tf.expand_dims(
            top_span_subtypes_prediction_logprob, 0)  # [k,k]
        rule2_ant_log_probs = tf.expand_dims(top_ant_preds_logprob, 1) + tf.expand_dims(top_ant_preds_logprob,
                                                                                        0)  # [k,k]
        rule2_loss = tf.multiply(tf.to_float(rule2_violation), rule2_subtype_log_probs + rule2_ant_log_probs)
        rule2_loss = tf.reduce_sum(rule2_loss)

        if self.config["add_realis_loss"]:
            # realis predictions
            top_span_realis = tf.argmax(top_span_realis_scores, axis=1)  # [k]
            top_span_realis_prob = tf.nn.softmax(top_span_realis_scores)  # [k, num_subtypes]
            top_span_realis_prediction_logprob = tf.log(tf.reduce_max(top_span_realis_prob, axis=1))  # [k]

            # RULE 3,  if realis(e, NULL) -> subtype(e, NULL)
            rule3_violation_1 = tf.logical_and(top_span_realis == 0, top_span_subtypes_prediction > 0)
            rule3_violation_2 = tf.logical_and(top_span_subtypes_prediction == 0, top_span_realis > 0)
            rule3_violation = tf.logical_or(rule3_violation_1, rule3_violation_2)

            rule3_loss = tf.multiply(tf.to_float(rule3_violation),
                                     top_span_subtypes_prediction_logprob + top_span_realis_prediction_logprob)  # [k]
            rule3_loss = tf.reduce_sum(rule3_loss)

            # RULE4 coref(e1, e2), realis(e1, t1), realis(e2, t2), t1 != t2 -> violation
            same_realis = tf.equal(tf.expand_dims(top_span_realis, 1), tf.expand_dims(top_span_realis, 0))  # [k, k]
            rule4_violation = tf.logical_and(top_span_same_cluster, tf.logical_xor(top_span_same_cluster, same_realis))
            rule4_realis_log_probs = tf.expand_dims(top_span_realis_prediction_logprob, 1) + tf.expand_dims(
                top_span_realis_prediction_logprob, 0)  # [k,k]
            rule4_loss = tf.multiply(tf.to_float(rule4_violation), rule4_realis_log_probs + rule2_ant_log_probs)
            rule4_loss = tf.reduce_sum(rule4_loss)

            # RULE5 if not coref(e, dummy) -> not realis(e, NULL)
            rule5_violation = tf.logical_and(top_span_realis == 0, top_ant_preds_index > -1)
            rule5_loss = tf.multiply(tf.to_float(rule5_violation),
                                     (top_span_realis_prediction_logprob + top_ant_preds_logprob))
            rule5_loss = tf.reduce_sum(rule5_loss)
        else:

            rule3_violation_count = tf.constant(0.0)
            rule4_violation_count = tf.constant(0.0)
            rule5_violation_count = tf.constant(0.0)

        loss = (rule1_loss + rule2_loss + rule3_loss + rule4_loss + rule5_loss)
        loss = tf.multiply(tf.constant(-1.0), loss)

        constraint_output = [rule1_loss, rule2_loss, rule3_loss, rule4_loss, rule5_loss]

        return loss, constraint_output

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                span_width_emb = tf.gather(
                    tf.get_variable("span_width_embeddings",
                                    [self.config["max_span_width"], self.config["feature_size"]],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02)),
                    span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
            head_attn_reps = tf.matmul(mention_word_scores, context_outputs)  # [K, T]
            span_emb_list.append(head_attn_reps)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]

        # if self.config["efficient"]:
        #  span_emb = tf.contrib.layers.recompute_grad(span_emb)
        return span_emb  # [k, emb]

    def get_mention_scores(self, span_emb, span_starts, span_ends):
        with tf.variable_scope("mention_scores"):
            span_scores = util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                    self.dropout)  # [k, 1]
        if self.config['use_prior']:
            span_width_emb = tf.get_variable("span_width_prior_embeddings",
                                             [self.config["max_span_width"], self.config["feature_size"]],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))  # [W, emb]
            span_width_index = span_ends - span_starts  # [NC]
            with tf.variable_scope("width_scores"):
                width_scores = util.ffnn(span_width_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                         self.dropout)  # [W, 1]
            width_scores = tf.gather(width_scores, span_width_index)
            span_scores += width_scores
        return span_scores

    def get_width_scores(self, doc, starts, ends):
        distance = ends - starts
        span_start_emb = tf.gather(doc, starts)
        hidden = util.shape(doc, 1)
        with tf.variable_scope('span_width'):
            span_width_emb = tf.gather(
                tf.get_variable("start_width_embeddings", [self.config["max_span_width"], hidden],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)), distance)  # [W, emb]
        scores = tf.reduce_sum(span_start_emb * span_width_emb, axis=1)
        return scores

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = util.shape(encoded_doc, 0)  # T
        num_c = util.shape(span_starts, 0)  # NC
        doc_range = tf.tile(tf.expand_dims(tf.range(0, num_words), 0), [num_c, 1])  # [K, T]
        mention_mask = tf.logical_and(doc_range >= tf.expand_dims(span_starts, 1),
                                      doc_range <= tf.expand_dims(span_ends, 1))  # [K, T]
        with tf.variable_scope("mention_word_attn", reuse=tf.AUTO_REUSE):
            word_attn = tf.squeeze(
                util.projection(encoded_doc, 1, initializer=tf.truncated_normal_initializer(stddev=0.02)), 1)
        mention_word_attn = tf.nn.softmax(tf.log(tf.to_float(mention_mask)) + tf.expand_dims(word_attn, 0))
        return mention_word_attn

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_subtypes=None, top_span_attributes=None, segment_distance=None):
        k = util.shape(top_span_emb, 0)
        c = util.shape(top_antecedents, 1)

        feature_emb_list = []

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)),
                antecedent_distance_buckets)  # [k, c]
            feature_emb_list.append(antecedent_distance_emb)
        if segment_distance is not None:
            with tf.variable_scope('segment_distance', reuse=tf.AUTO_REUSE):
                segment_distance_emb = tf.gather(tf.get_variable("segment_distance_embeddings",
                                                                 [self.config['max_training_sentences'],
                                                                  self.config["feature_size"]],
                                                                 initializer=tf.truncated_normal_initializer(
                                                                     stddev=0.02)), segment_distance)  # [k, emb]
            feature_emb_list.append(segment_distance_emb)

        if self.config["use_same_subtype_fea"] and top_span_subtypes is not None:
            top_antecedent_subtypes = tf.gather(top_span_subtypes, top_antecedents)  # [k, c]
            same_subtypes = tf.equal(tf.expand_dims(top_span_subtypes, 1), top_antecedent_subtypes)  # [k, c]
            same_subtypes_fea = tf.expand_dims(tf.to_float(same_subtypes), 2)
            feature_emb_list.append(same_subtypes_fea)

        if self.config["use_same_realis_fea"] and top_span_subtypes is not None:
            top_antecedent_attributes = tf.gather(top_span_attributes, top_antecedents)  # [k, c]
            same_attributes = tf.equal(tf.expand_dims(top_span_attributes, 1), top_antecedent_attributes)  # [k, c]
            same_attributes_fea = tf.expand_dims(tf.to_float(same_attributes), 2)
            feature_emb_list.append(same_attributes_fea)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.expand_dims(top_span_emb, 1)  # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [k, c, emb]
        target_emb = tf.tile(target_emb, [1, c, 1])  # [k, c, emb]

        pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout, efficient=self.config["efficient"])  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]

        # if self.config["efficient"]:
        #  pair_emb = tf.contrib.layers.recompute_grad(pair_emb)
        return slow_antecedent_scores  # [k, c]

    def get_fast_antecedent_scores(self, top_span_emb):
        with tf.variable_scope("src_projection"):
            source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)),
                                                self.dropout)  # [k, emb]
        target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout)  # [k, emb]
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True)  # [k, k]

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_subtypes, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index, (i, predicted_index)

            # if predicted_subtypes:
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]),
                                    predicted_subtypes[predicted_index])
            # else:
            #  predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]), "event")

            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            # if predicted_subtypes:
            mention = (int(top_span_starts[i]), int(top_span_ends[i]), predicted_subtypes[i])
            # else:
            #  mention = (int(top_span_starts[i]), int(top_span_ends[i]), "event")

            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def get_predicted_antecedents_with_gold_subtypes(self, span_subtypes, antecedents, antecedent_scores):
        predicted_antecedents = []
        num_mentions = len(span_subtypes)

        for i in range(num_mentions):
            antecedent_score_i = antecedent_scores[i][1:]
            best_score = 0
            pred_ant_with_same_type = -1

            for idx, score in enumerate(antecedent_score_i):
                # print("mention id:", i, idx, score, "mtype:", span_subtypes[i], "anttype:", span_subtypes[antecedents[i, idx]])
                if span_subtypes[i] == span_subtypes[antecedents[i, idx]] and span_subtypes[
                    i] != "null" and score > best_score:
                    best_score = score
                    pred_ant_with_same_type = antecedents[i, idx]

            if pred_ant_with_same_type < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(pred_ant_with_same_type)

            # print("mention id:", i, "pred_ant:", pred_ant_with_same_type)

        return predicted_antecedents

    def get_predicted_antecedents_with_gold_ana(self, spans_ana, span_scores, antecedents, antecedent_scores):
        predicted_antecedents = []
        num_mentions = len(spans_ana)

        for i in range(num_mentions):
            antecedent_score_i = antecedent_scores[i][1:]
            best_score = float('-inf')
            pred_ant = -1

            if spans_ana[i] == 1:
                for idx, score in enumerate(antecedent_score_i):
                    # if antecedents[i, idx] < i and score > best_score and span_scores[antecedents[i, idx]] > 0:
                    if score > best_score:
                        best_score = score
                        pred_ant = antecedents[i, idx]

            print("mention id:", i, spans_ana[i], "mscore:", span_scores[i], best_score, pred_ant,
                  span_scores[pred_ant])

            predicted_antecedents.append(pred_ant)

        return predicted_antecedents

    def get_predicted_subtypes(self, mention_scores, subtype_dict):
        predicted_subtypes = []
        predicted_subtype_indices = []

        # assert len(subtype_dict) == self.config["num_subtypes"] - 1

        for i, index in enumerate(np.argmax(mention_scores, axis=1)):
            # print(mention_scores[i])
            # print(index)
            predicted_subtype_indices.append(index)
            if index == 0:
                predicted_subtypes.append("null")
            else:
                predicted_subtypes.append(subtype_dict[index])

        return predicted_subtypes, predicted_subtype_indices

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_subtypes, predicted_antecedents, event_evaluator,
                       entity_evaluator, gold_clusters, not_evaluate_singleton=False):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_subtypes,
                                                                               predicted_antecedents)

        # add singletons
        if not not_evaluate_singleton:
            for mstart, mend, mtype in zip(top_span_starts, top_span_ends, predicted_subtypes):
                if mtype != "null" and (int(mstart), int(mend)) not in mention_to_predicted:
                    predicted_clusters.append([(int(mstart), int(mend), mtype)])

        # entity_predicted_clusters = []
        # event_predicted_clusters = []

        # entity_gold_clusters = []
        # event_gold_clusters = []

        # for gc in gold_clusters:
        #   if len(gc) == 0:
        #     continue

        #   if "_" in gc[0][2]:
        #     #print("event gc:", gc)
        #     event_gold_clusters.append(gc)
        #   else:
        #     #print("entity gc:", gc)
        #     entity_gold_clusters.append(gc)

        # for pc in predicted_clusters:
        #   if len(pc) == 0:
        #     continue

        #   if "_" in pc[0][2]:
        #     #print("event pc:", pc)
        #     event_predicted_clusters.append(pc)
        #   else:
        #     #print("entity pc:", pc)
        #     event_predicted_clusters.append(pc)

        # event_evaluator.update(event_predicted_clusters, event_gold_clusters, mention_to_predicted, mention_to_gold)
        # entity_evaluator.update(entity_predicted_clusters, entity_gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self, dataset='dev'):
        if (self.dev_data is None and dataset == 'dev') or (self.test_data is None and dataset != 'dev'):
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_example(example, is_training=False, is_training_coref=False), example

            if dataset == 'dev':
                with open(self.config["eval_path"]) as f:
                    self.dev_data = [load_line(l) for l in f.readlines()]
                    # num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
                    print("Loaded {} eval examples.".format(len(self.dev_data)))
            else:
                with open(self.config["test_path"]) as f:
                    self.test_data = [load_line(l) for l in f.readlines()]
                    print("Loaded {} eval examples.".format(len(self.test_data)))
        if dataset == 'dev':
            self.eval_data = self.dev_data
        else:
            self.eval_data = self.test_data

    def evaluate(self, session, out_path, dataset='dev', eval_mode=False):
        self.load_eval_data(dataset=dataset)

        coref_predictions = {}
        event_coref_evaluator = metrics.CorefEvaluator()
        entity_coref_evaluator = metrics.CorefEvaluator()
        subtype_evaluator = metrics.SubtypeEvaluator()
        losses = []
        doc_keys = []
        num_evaluated = 0

        if not self.config["end_to_end"]:
            out_path = out_path + "-goldmentions-"

        if self.config["eval_with_gold_subtype"]:
            out_path += "-eval_with_gold_subtype"

        if self.config["eval_with_gold_ana"]:
            out_path += "-eval_with_gold_ana"

        if self.config["eval_with_gold_realis"]:
            out_path += "-eval_with_gold_realis"

        if self.config["eval_with_gold_argument"]:
            out_path += "-eval_with_gold_argument"

        out_path += f'-{dataset}'

        json_filename = out_path + "-predictions.json";

        example_map = {}
        with open(json_filename, "w") as f:
            for example_num, (tensorized_example, example) in enumerate(self.eval_data):
                input_ids, input_mask, text_len, is_training, is_training_coref, gold_starts, gold_ends, cluster_ids, \
                sentence_map, gold_subtypes, gold_types, gold_realis, gold_arg_mstart, gold_arg_mend, gold_arg_start, \
                gold_arg_end, gold_arg_roles, gold_anaphoricity, not_E7394, _ = tensorized_example

                # print(example["doc_key"], "not_E7394:", not_E7394)

                feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
                # if tensorized_example[0].shape[0] <= 9:
                # if keys is not None and example['doc_key'] not in keys:
                #   # print('Skipping...', example['doc_key'], tensorized_example[0].shape)
                #   continue

                doc_keys.append(example['doc_key'])

                loss, (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, \
                       top_span_mention_scores, coref_output, \
                       subtype_output, realis_output, anaphoricity_output, type_output), argument_output = session.run(
                    [self.loss, self.predictions, self.argument_output], feed_dict=feed_dict)
                # losses.append(session.run(self.loss, feed_dict=feed_dict))
                losses.append(loss)

                predicted_subtypes = []
                if subtype_output:
                    top_span_subtype_scores, top_span_subtypes, _ = subtype_output
                    inv_subtype_dict = {v: k for k, v in self.subtype_dict.items()}
                    predicted_subtypes, predicted_subtype_indices = self.get_predicted_subtypes(top_span_subtype_scores,
                                                                                                inv_subtype_dict)
                    subtype_evaluator.update(predicted_subtype_indices, top_span_subtypes)

                    example['top_spans'] = []
                    for start, end in zip(top_span_starts, top_span_ends):
                        example['top_spans'].append((int(start), int(end)))

                    example['top_spans_subtypes'] = []
                    if predicted_subtypes:
                        example['top_spans_subtypes'] = predicted_subtypes

                if type_output:
                    top_span_type_scores, _ = type_output
                    inv_type_dict = {v: k for k, v in self.type_dict.items()}
                    predicted_types, predicted_type_indices = self.get_predicted_subtypes(top_span_type_scores,
                                                                                          inv_type_dict)

                    if predicted_types:
                        example['top_spans_types'] = predicted_types
                    # print("doc_key:", example["doc_key"])

                # if not self.config["end_to_end"]:
                #     example['top_spans'] = []
                #     for start, end in zip(gold_starts, gold_ends):
                #       example['top_spans'].append((int(start), int(end)))

                #     # inv_subtype_dict = {v:k for k, v in self.subtype_dict.items()}
                #     # example['top_spans_subtypes'] = []
                #     # for mtype in gold_subtypes:
                #     #   if mtype == 0:
                #     #     example['top_spans_subtypes'].append("null")
                #     #   else:
                #     #     example['top_spans_subtypes'].append(inv_subtype_dict[mtype])

                #     predicted_subtypes = example['top_spans_subtypes']

                # predicted_subtypes = example['top_spans_subtypes']

                subtype_map = {}
                for span, subtype in zip(example['top_spans'], example['top_spans_subtypes']):
                    subtype_map[tuple(span)] = subtype

                # output anaphoricity
                if anaphoricity_output:
                    anaphoric_scores = anaphoricity_output[0]
                    example["predicted_anaphoricity"] = []
                    for score in anaphoric_scores:
                        if score > 0:
                            example["predicted_anaphoricity"].append(1)
                        else:
                            example["predicted_anaphoricity"].append(-1)

                if realis_output:
                    inv_realis_dict = {}
                    for key, value in self.realis_dict.items():
                        inv_realis_dict[value] = key

                    predicted_realis, _ = self.get_predicted_subtypes(realis_output[0], inv_realis_dict)
                    example['top_spans_realis'] = predicted_realis

                if coref_output:
                    top_antecedents, top_antecedent_scores = coref_output

                    if self.config["eval_with_gold_subtype"]:
                        print("eval with gold subtypes:")
                        gold_subtype_map = {}
                        for c in example["gold_clusters"]:
                            for m in c:
                                gold_subtype_map[(m[0], m[1])] = m[2]
                        top_spans_gold_subtypes = []
                        for start, end in zip(top_span_starts, top_span_ends):
                            if (start, end) in gold_subtype_map:
                                top_spans_gold_subtypes.append(gold_subtype_map[(start, end)])
                            else:
                                top_spans_gold_subtypes.append("null")

                        predicted_antecedents = self.get_predicted_antecedents_with_gold_subtypes(
                            top_spans_gold_subtypes, top_antecedents, top_antecedent_scores)
                    elif self.config["eval_with_gold_realis"]:
                        print("eval_with_gold_realis")
                        gold_realis_map = {}
                        for mention in example["gold_event_mentions"]:
                            mstart, mend, _, mrealis = mention
                            gold_realis_map[(mstart, mend)] = mrealis

                        top_spans_realis = []
                        for start, end in zip(top_span_starts, top_span_ends):
                            if (start, end) in gold_realis_map:
                                top_spans_realis.append(gold_realis_map[(start, end)])
                            else:
                                top_spans_realis.append("null")
                        # print(top_spans_realis)
                        # input(" ")

                        predicted_antecedents = self.get_predicted_antecedents_with_gold_subtypes(top_spans_realis,
                                                                                                  top_antecedents,
                                                                                                  top_antecedent_scores)

                    elif self.config["eval_with_gold_ana"]:
                        print("eval with gold ana:")
                        gold_ana_map = {}
                        for cid, c in enumerate(example["gold_clusters"]):
                            for mid, m in enumerate(sorted(c)):
                                if mid == 0:
                                    gold_ana_map[(m[0], m[1])] = -1
                                else:
                                    gold_ana_map[(m[0], m[1])] = 1
                        # print(gold_ana_map)

                        # print(cid, mid, m, gold_ana_map[(m[0], m[1])])

                        top_spans_ana = []
                        for start, end in zip(top_span_starts, top_span_ends):
                            if (start, end) in gold_ana_map:
                                top_spans_ana.append(gold_ana_map[(start, end)])
                            else:
                                top_spans_ana.append(-1)
                        # print(top_spans_ana)
                        # print("predicted_antecedents:", predicted_antecedents)
                        predicted_antecedents = self.get_predicted_antecedents_with_gold_ana(top_spans_ana,
                                                                                             top_span_mention_scores,
                                                                                             top_antecedents,
                                                                                             top_antecedent_scores)
                        # print("predicted_antecedents:", predicted_antecedents)
                        # input(" ")
                    else:
                        predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)

                    predicted_clusters = self.evaluate_coref(top_span_starts, top_span_ends, predicted_subtypes,
                                                             predicted_antecedents, event_coref_evaluator,
                                                             entity_coref_evaluator, example["gold_clusters"], True)
                    coref_predictions[example["doc_key"]] = predicted_clusters
                    if predicted_clusters and not predicted_subtypes and not self.config["pipelined_subtypes"]:
                        cluster_mentions = set()
                        for c in predicted_clusters:
                            for m in c:
                                cluster_mentions.add((m[0], m[1]))

                        example['top_spans_subtypes'] = []
                        for m in example['top_spans']:
                            if m in cluster_mentions:
                                example['top_spans_subtypes'].append("event")
                            else:
                                example['top_spans_subtypes'].append("null")

                    example["predicted_antecedents"] = []
                    for i, predicted_index in enumerate(predicted_antecedents):
                        if predicted_index < 0:
                            example["predicted_antecedents"].append((-1, -1))
                        else:
                            example["predicted_antecedents"].append(
                                (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index])))

                        assert i > predicted_index, (i, predicted_index)

                    example["predicted_clusters"] = predicted_clusters

                # output predicted argument
                if self.config["add_span_argument_loss"]:

                    top_arg_mstart, top_arg_mend, top_arg_starts, top_arg_ends, top_arg_scores, top_arg_labels, _ = argument_output

                    # argument_instances.extend(top_arg_labels)

                    inv_role_dict = {}
                    for key, value in self.arg_role_dict.items():
                        inv_role_dict[value] = key
                    predicted_arguments = []

                    for i in range(0, len(top_arg_labels)):
                        mid = str(top_arg_mstart[i]) + "," + str(top_arg_mend[i])
                        event_type = subtype_map[(top_arg_mstart[i], top_arg_mend[i])] if (top_arg_mstart[i],
                                                                                           top_arg_mend[
                                                                                               i]) in subtype_map else "null"
                        arg_start = top_arg_starts[i]
                        arg_end = top_arg_ends[i]
                        arg_role_id = np.argmax(top_arg_scores[i])
                        arg_role = inv_role_dict[arg_role_id]

                        pred_argument = [int(top_arg_mstart[i]), int(top_arg_mend[i]), event_type, int(arg_start),
                                         int(arg_end), arg_role.lower(), ]

                        if arg_role == "NULL":
                            continue

                        predicted_arguments.append(pred_argument)

                    example["predicted_arguments"] = list(predicted_arguments)

                if example_num % 10 == 0:
                    print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

                example_map[example["doc_key"]] = example
                f.write(json.dumps(example))
                f.write("\n")

        offset_map = {}
        offset_name = self.config["offset_name"]
        if offset_name != "None":
            if dataset == 'dev': offset_name = self.config["offset_name_dev"]
            with open(offset_name, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    doc = json.loads(line)
                    offset_map[doc['doc_key']] = doc

        eval_coref = True
        if not coref_output:
            eval_coref = False
        gold_path = self.config['gold_path'] if dataset != 'dev' else self.config['gold_dev_path']
        entity_gold_path = self.config['entity_gold_path'] if dataset != 'dev' else self.config['entity_gold_dev_path']
        if eval_mode:
            all_mention_subtype_map = evaluate_scorer.output_scorer_format(example_map, eval_coref, out_path + "-final",
                                                                           "top_spans", offset_map, self.subtoken_maps,
                                                                           self.config["quote_file"], False)
            scores = coref_scorer.evaluate_file(gold_path, out_path + "-final",
                                                out_path + "-final-score", out_path + "-final-corefscore", None,
                                                self.config['type_white_list'])

            # scores = coref_scorer.evaluate_file(self.config['gold_path'], out_path + "-final", out_path + "-final-score", out_path + "-final-corefscore", None, self.config['type_white_list'])
            # evaluate entity
            entity_scores = coref_scorer.evaluate_file(entity_gold_path, out_path + "-final",
                                                       out_path + "-final-entity-score", \
                                                       out_path + "-final-entity-corefscore", None,
                                                       self.config["entity_type_white_list"])
            print("entity_scores:")
            for score in entity_scores:
                print(score)

            if self.config["add_span_argument_loss"]:
                arg_p, arg_r, arg_f, arg_wo_p, arg_wo_r, arg_wo_f = evaluate_scorer.evaluate_argument(example_map,
                                                                                                      all_mention_subtype_map,
                                                                                                      False, False)
                scores["argument"] = [arg_p, arg_r, arg_f, arg_wo_p, arg_wo_r, arg_wo_f]

            # anap, anar, anaf = self.prf(total_matched_anaphoricity, total_gold_anaphoricity, total_sys_anaphoricity)
            # scores["anaphoricity"] = [anap, anar, anaf]

            # anap, anar, anaf = self.prf(total_matched_pipeline_anaphoricity, total_gold_anaphoricity, total_pipeline_anaphoricity)
            # scores["pipeline_anaphoricity"] = [anap, anar, anaf]

            # with open(out_path + "-scores", "w") as outf:
            #   outf.write(json.dumps(scores))

            # print("extend argument annotations:")
            # arg_p, arg_r, arg_f, arg_wo_p, arg_wo_r, arg_wo_f = evaluate_scorer.evaluate_argument(example_map, all_mention_subtype_map, False, False, extend_arg=True)
            # scores["argument_extend"] = [arg_p, arg_r, arg_f, arg_wo_p, arg_wo_r, arg_wo_f]
        else:
            all_mention_subtype_map = evaluate_scorer.output_scorer_format(example_map, eval_coref, out_path + "-final",
                                                                           "top_spans", offset_map, self.subtoken_maps,
                                                                           self.config["quote_file"], False)
            scores = coref_scorer.evaluate_file(gold_path, out_path + "-final",
                                                out_path + "-final-score", out_path + "-final-corefscore", None,
                                                self.config['type_white_list'])
            entity_scores = coref_scorer.evaluate_file(entity_gold_path, out_path + "-final",
                                                       out_path + "-final-entity-score", \
                                                       out_path + "-final-entity-corefscore", None,
                                                       self.config["entity_type_white_list"])
        # summary_dict = {}
        # evp,evr,evf = event_coref_evaluator.get_prf()
        # summary_dict["Average F1 (py)"] = evf
        # print("Average Event Coref F1 (py): {:.2f}% on {} docs".format(evf * 100, len(doc_keys)))
        # summary_dict["Average precision (py)"] = evp
        # print("Average precision (py): {:.2f}%".format(evp * 100))
        # summary_dict["Average recall (py)"] = evr
        # print("Average recall (py): {:.2f}%".format(evr * 100))

        # enp,enr,enf = entity_coref_evaluator.get_prf()
        # print("Average Entity Coref F1 (py): {:.2f}% on {} docs".format(enf * 100, len(doc_keys)))
        # print("Average precision (py): {:.2f}%".format(enp * 100))
        # print("Average recall (py): {:.2f}%".format(enr * 100))

        # argument_instances_counter = collections.Counter(argument_instances)
        # print(len(argument_instances))
        # print(argument_instances_counter.most_common())
        return scores

    def debug_predictions(self, example, candidate_starts, candidate_ends, candidate_mention_scores,
                          candidate_cluster_ids, candidate_is_gold, candidate_subtypes, \
                          top_span_starts, top_span_ends, top_span_mention_scores, \
                          top_antecedents, top_antecedent_scores, top_antecedent_labels, top_span_label,
                          top_span_sentence_indices, top_fast_antecedent_scores, argument_output, predicted_antecedents, \
                          gold_starts, gold_ends, gold_subtypes, coref_loss):

        # tokens
        token_sid_map = {}
        tokens = []
        tid = 0
        for sid, sentence in enumerate(example["sentences"]):
            tokens.extend(sentence)

            for token in sentence:
                token_sid_map[tid] = sid
                tid += 1

                # gold annotation
        gold_subtype_map = {}
        for start, end, subtype in zip(gold_starts, gold_ends, gold_subtypes):
            gold_subtype_map[(start, end)] = subtype

        subtype_map = {}
        for span, subtype in zip(example['top_spans'], example['top_spans_subtypes']):
            subtype_map[tuple(span)] = subtype

        gold_span_map = {}  # whether a span is a gold span
        cluster_id_map = {}  # gold cluster id for each span
        cluster_multi_id_map = defaultdict(list)  # whether a span involves in more than one cluster
        subtype_id_map = {}  # gold subtype id for each span
        ant_dist = []  # sentence distance between a mention and its closest antecedent
        mention_anaphoricity = {}  # whether a mention is anaphoric or not
        for i in range(0, len(candidate_starts)):
            span = (candidate_starts[i], candidate_ends[i])
            gold_span_map[span] = candidate_is_gold[i]
            cluster_id_map[span] = candidate_cluster_ids[i]
            subtype_id_map[span] = candidate_subtypes[i]

        for cid, cluster in enumerate(example["gold_clusters"]):
            cluster = sorted([(m[0], m[1]) for m in cluster])
            if len(cluster) == 0:
                continue
            mention_anaphoricity[cluster[0]] = False
            for mid in range(1, len(cluster)):
                mention_anaphoricity[cluster[mid]] = True
                if "_" in m[2]:  # only check events
                    ant_dist.append(token_sid_map[cluster[mid][0]] - token_sid_map[cluster[mid - 1][0]])

            for m in cluster:
                cluster_multi_id_map[(m[0], m[1])].append(cid + 1)

        candidate_anaphoricity = {}
        for i in range(0, len(candidate_starts)):
            span = (candidate_starts[i], candidate_ends[i])
            if span in mention_anaphoricity:
                candidate_anaphoricity[span] = mention_anaphoricity[span]
            else:
                candidate_anaphoricity[span] = False
                if gold_span_map[span] != False:
                    print(span, " is gold but not found in gold_clusters")

        # print out candidate mention scores
        # for start, end, score in zip(candidate_starts, candidate_ends, candidate_mention_scores):
        #   print(start, end, score, " ".join(tokens[start:end+1]))
        # raw_input(" ")

        # print out antencedent predictions
        for i in range(0, len(top_span_starts)):
            span = (top_span_starts[i], top_span_ends[i])
            predsubtype = subtype_map[span]
            span_str = (" ".join(tokens[span[0]:span[1] + 1])).lower()

            print("\n\n")
            print("mention:", i, span, tokens[top_span_starts[i]:top_span_ends[i] + 1], \
                  top_span_mention_scores[i], \
                  "if in gold mention:", gold_span_map[span], \
                  "gold cluster id:", cluster_id_map[span], "multi:", cluster_multi_id_map[span], \
                  "pred_span_label:", subtype_map[span], candidate_anaphoricity[span], \
                  "gold_span_label:", subtype_id_map[span],
                  "in sentence :", top_span_sentence_indices[i]
                  )
            print("top_ant:", " size:", len(top_antecedents[i]), top_antecedents[i])
            print("pred_ant:", predicted_antecedents[i])
            print("coref_loss:", coref_loss[i])
            if np.isnan(coref_loss[i]):
                raw_input(" ")

            for a, scorea in zip(top_antecedents[i], top_antecedent_scores[i][1:]):
                spana = (top_span_starts[a], top_span_ends[a])
                spana_str = (" ".join(tokens[spana[0]:spana[1] + 1])).lower()

                iscoref = cluster_id_map[span] == cluster_id_map[spana] and a != i
                print(a, spana, tokens[spana[0]:spana[1] + 1], top_span_sentence_indices[a], iscoref,
                      cluster_id_map[spana], scorea)

            # raw_input(" ")
