import argparse
import heapq
import itertools
import logging
import math
import os
import re
import sys
import json
import io


import utils
from config import Config, MutableConfig, EvalState
from conll_coref import ConllEvaluator
import scorer_util

try: 
    import queue as Queue
except ImportError:
    import Queue
#import Queue

# from temporal import TemporalEval

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s : %(message)s'))
logger.addHandler(stream_handler)


def main():
    eval_data  = None
    eval_path = "english.test.best_gold.json"
    output_filename = "out.test.best_gold"
   
    with open(eval_path) as f:
        eval_data = map(lambda example: (example, example), (json.loads(jsonline) for jsonline in f.readlines()))

    scores = evaluate_type_and_coref(output_filename+"-score", output_filename+"-corefscore", output_filename+"-diff", eval_data, None)

    for scoretype, score in scores.items():
        print(scoretype, score)


def evaluate_type_and_coref(out_path, coref_out, diff_out_path, examples, type_white_list_path):
    # Read all documents.
    #read_all_doc(gf, sf, args.doc_id_to_eval)
    EvalState.clear()
    diff_out = None
    if diff_out_path is not None:
        utils.create_parent_dir(diff_out_path)
        diff_out = open(diff_out_path, 'w')


    if out_path is not None:
        utils.create_parent_dir(out_path)
        mention_eval_out = open(out_path, 'w')
        logger.info("Evaluation output will be saved at %s" % out_path)
    else:
        mention_eval_out = sys.stdout
        logger.info("Evaluation output at standard out.")

    if coref_out is not None:
        Config.conll_out = coref_out
        Config.conll_gold_file = coref_out + "_gold.conll"
        Config.conll_sys_file = coref_out + "_sys.conll"

    if type_white_list_path is not None:
        logger.info("Only the following types in the white list will be evaluated.")
        EvalState.white_listed_types = set()
        with open(type_white_list_path, "r") as f:
            type_white_list = f.readlines()
            for line in type_white_list:
                logger.info(line.strip())
                EvalState.white_listed_types.add(scorer_util.canonicalize_string(line))

    # Take all attribute combinations, which will be used to produce scores.
    attribute_comb = get_attr_combinations(Config.attribute_names)

    logger.info("Coreference mentions need to match %s before consideration" % Config.coref_criteria[0][1])

    for example_num, (tensorized_example, example) in enumerate(examples):
        if not evaluate(coref_out, attribute_comb, diff_out, example):
            break

    if coref_out is not None:
        # Run the CoNLL script on the combined files, which is concatenated from the best alignment of all documents.
        logger.debug("Running coreference script for the final scores.")
        ConllEvaluator.run_conll_script(Config.conll_gold_file, Config.conll_sys_file, Config.conll_out)
        # Get the CoNLL scores from output
        EvalState.overall_coref_scores = ConllEvaluator.get_conll_scores(Config.conll_out)

    scores = scorer_util.print_eval_results(mention_eval_out, attribute_comb)

    # Clean up, close files.
    scorer_util.close_if_not_none(diff_out)

    logger.info("Evaluation Done.")
    return scores


def get_attr_combinations(attr_names):
    """
    Generate all possible combination attributes.
    :param attr_names: List of attribute names
    :return:
    """
    attribute_names_with_id = list(enumerate(attr_names))
    comb = []
    for L in range(1, len(attribute_names_with_id) + 1):
        comb.extend(itertools.combinations(attribute_names_with_id, L))
    logger.debug("Will score on the following attribute combinations : ")
    logger.debug(", ".join([str(x) for x in comb]))
    return comb


def evaluate_file(gold_path, system_path, out_path, coref_out, diff_out_path, type_white_list_path):
    EvalState.clear()
    diff_out = None
    if diff_out_path is not None:
        utils.create_parent_dir(diff_out_path)
        diff_out = open(diff_out_path, 'w')


    if out_path is not None:
        utils.create_parent_dir(out_path)
        mention_eval_out = open(out_path, 'w')
        logger.info("Evaluation output will be saved at %s" % out_path)
    else:
        mention_eval_out = sys.stdout
        logger.info("Evaluation output at standard out.")

    if coref_out is not None:
        Config.conll_out = coref_out
        Config.conll_gold_file = coref_out + "_gold.conll"
        Config.conll_sys_file = coref_out + "_sys.conll"

    if type_white_list_path is not None:
        logger.info("Only the following types in the white list will be evaluated.")
        EvalState.white_listed_types = set()
        with open(type_white_list_path, "r") as f:
            type_white_list = f.readlines()
            for line in type_white_list:
                logger.info(line.strip())
                EvalState.white_listed_types.add(scorer_util.canonicalize_string(line))


    if os.path.isfile(gold_path):
        gf = open(gold_path)
    else:
        logger.error("Cannot find gold standard file at " + gold_path)
        sys.exit(1)

    if os.path.isfile(system_path):
        sf = open(system_path)
    else:
        logger.error("Cannot find system file at " + system_path)
        sys.exit(1)

  
    Config.coref_criteria = Config.possible_coref_mapping[1]

    # Read all documents.
    scorer_util.read_all_doc(gf, sf, None)

    # Take all attribute combinations, which will be used to produce scores.
    attribute_comb = get_attr_combinations(Config.attribute_names)

    logger.info("Coreference mentions need to match %s before consideration" % Config.coref_criteria[0][1])

    while True:
        token_dir = "."
        token_offset_fields = Config.default_token_offset_fields
        token_table_extension = Config.default_token_file_ext
        if not scorer_util.evaluate(token_dir, coref_out, attribute_comb,
                        token_offset_fields, token_table_extension,
                        diff_out):
            break

    # Run the CoNLL script on the combined files, which is concatenated from the best alignment of all documents.
    if coref_out is not None:
        logger.debug("Running coreference script for the final scores.")
        ConllEvaluator.run_conll_script(Config.conll_gold_file, Config.conll_sys_file, Config.conll_out)
        # Get the CoNLL scores from output
        EvalState.overall_coref_scores = ConllEvaluator.get_conll_scores(Config.conll_out)


    scores = scorer_util.print_eval_results(mention_eval_out, attribute_comb)

    # Clean up, close files.
    scorer_util.close_if_not_none(diff_out)

    logger.info("Evaluation Done.")
    return scores

def evaluate(coref_out, all_attribute_combinations, diff_out, example):
    """
    Conduct the main evaluation steps.
    :param coref_out:
    :param all_attribute_combinations:
    :param diff_out:
    :return:
    """

    # if EvalState.has_next_doc():
    #    res, (g_mention_lines, g_relation_lines), (
    #        s_mention_lines, s_relation_lines), doc_id, system_id = get_next_doc()
    #else:
    #    return False
    doc_id = example['doc_key']
    candidate_mentions = [tuple(m) for m in example["candidate_char"]]
    candidate_subtypes = example["subtypes"]
    candidate_clusters = []

    gold_mentions = [tuple(m) for m in example["gold_char"]]
    gold_subtypes = example["gold_subtypes"]
    gold_clusters = example["gold_clusters_char"]

    system_id = "s1"

    logger.info("Evaluating Document %s" % doc_id)

    if len(gold_mentions) == 0:
        logger.warn(
            "[%s] does not contain gold standard mentions. Document level F score will not be valid, but the micro "
            "score will be fine." % doc_id)


    # Parse the lines and save them as a table from id to content.
    system_mention_table = []
    gold_mention_table = []

    system_id_map = {}
    system_tkn_char_map = {}

    gold_id_map = {}

    # Save the raw text for visualization.
    sys_id_2_text = {}
    gold_id_2_text = {}

    logger.debug("Reading gold and response mentions.")

    remaining_sys_ids = set()
    num_system_mentions = 0
    for sid in range(0, len(candidate_mentions)):
        # print(candidate_mentions)
        spans = parse_characters(str(candidate_mentions[sid][2]) + "," + str(candidate_mentions[sid][3]))
        original_spans = spans
        event_id = "E" + str(sid)
        attributes = [candidate_subtypes[sid]]  # subtype
        text = " "

        if (candidate_mentions[sid][2],candidate_mentions[sid][3],candidate_subtypes[sid]) in system_id_map:
            system_id_map[(candidate_mentions[sid][2],candidate_mentions[sid][3],candidate_subtypes[sid])].put(event_id)
        else:
            system_id_map[(candidate_mentions[sid][2],candidate_mentions[sid][3],candidate_subtypes[sid])] = Queue.Queue()
            system_id_map[(candidate_mentions[sid][2],candidate_mentions[sid][3],candidate_subtypes[sid])].put(event_id)


        system_tkn_char_map[(candidate_mentions[sid][0],candidate_mentions[sid][1])] = (candidate_mentions[sid][2],candidate_mentions[sid][3], candidate_subtypes[sid])
        # key: token_id, value: chararcter_id

        parse_result = [spans, attributes, event_id, original_spans, text]

        # If parse result is rejected, we ignore this line.
        if not parse_result:
            continue

        num_system_mentions += 1

        sys_attributes = parse_result[1]
        sys_mention_id = parse_result[2]
        text = parse_result[4]

        system_mention_table.append(parse_result)
        EvalState.all_possible_types.add(sys_attributes[0])
        remaining_sys_ids.add(sys_mention_id)
        sys_id_2_text[sys_mention_id] = text

    if not num_system_mentions == len(remaining_sys_ids):
        logger.warn("Duplicated mention id for doc %s, one of them is randomly removed." % doc_id)

    remaining_gold_ids = set()
    for gid in range(0, len(gold_mentions)):
        # print(gold_mentions[gid])
        spans = parse_characters(str(gold_mentions[gid][0]) + "," + str(gold_mentions[gid][1]))
        original_spans = spans
        event_id = "E" + str(gid)
        attributes = [gold_subtypes[gid]]  # subtype
        text = " "
        if (gold_mentions[gid][0], gold_mentions[gid][1], gold_subtypes[gid]) in gold_id_map:
            gold_id_map[(gold_mentions[gid][0], gold_mentions[gid][1], gold_subtypes[gid])].put(event_id)
        else:
            gold_id_map[(gold_mentions[gid][0], gold_mentions[gid][1], gold_subtypes[gid])] = Queue.Queue()
            gold_id_map[(gold_mentions[gid][0], gold_mentions[gid][1], gold_subtypes[gid])].put(event_id)

        parse_result = [spans, attributes, event_id, original_spans, text]

        # If parse result is rejected, we ignore this line.
        if not parse_result:
            continue

        gold_attributes = parse_result[1]
        gold_mention_id = parse_result[2]
        text = parse_result[4]

        gold_mention_table.append(parse_result)
        EvalState.all_possible_types.add(gold_attributes[0])
        gold_id_2_text[gold_mention_id] = text
        remaining_gold_ids.add(gold_mention_id)

    num_system_predictions = len(system_mention_table)
    num_gold_predictions = len(gold_mention_table)

    # Store list of mappings with the score as a priority queue. Score is stored using negative for easy sorting.
    all_gold_system_mapping_scores = []

    # Debug purpose printing.
    print_score_matrix = False

    logger.debug("Computing overlap scores.")
    for system_index, (sys_spans, sys_attributes, sys_mention_id, _, _) in enumerate(system_mention_table):
        if print_score_matrix:
            print("%d %s" % (system_index, sys_mention_id))
        for index, (gold_spans, gold_attributes, gold_mention_id, _, _) in enumerate(gold_mention_table):
            if len(gold_spans) == 0:
                logger.warning("Found empty span gold standard at doc : %s, mention : %s" % (doc_id, gold_mention_id))
            if len(sys_spans) == 0:
                logger.warning("Found empty span system standard at doc : %s, mention : %s" % (doc_id, sys_mention_id))

            overlap = scorer_util.compute_overlap_score(gold_spans, sys_spans)

            if print_score_matrix:
                sys.stdout.write("%.1f " % overlap)

            if overlap > 0:
                # maintaining a max heap based on overlap score
                heapq.heappush(all_gold_system_mapping_scores, (-overlap, system_index, index))
        if print_score_matrix:
            print

    greedy_tp, greedy_attribute_tps, greedy_mention_only_mapping, greedy_all_attribute_mapping = scorer_util.get_tp_greedy(
        all_gold_system_mapping_scores, all_attribute_combinations, gold_mention_table,
        system_mention_table, doc_id)

    scorer_util.write_if_provided(diff_out, Config.bod_marker + " " + doc_id + "\n")
    if diff_out is not None:
        # Here if you change the mapping used, you will see what's wrong on different level!

        # write_gold_and_system_mappings(doc_id, system_id, greedy_all_attribute_mapping[0], gold_mention_table,
        #                                system_mention_table, diff_out)

        scorer_util.write_gold_and_system_mappings(system_id, greedy_mention_only_mapping, gold_mention_table, system_mention_table,
                                       diff_out)

    attribute_based_fps = [0.0] * len(all_attribute_combinations)
    for attribute_comb_index, abtp in enumerate(greedy_attribute_tps):
        attribute_based_fps[attribute_comb_index] = num_system_predictions - abtp

    # Unmapped system mentions and the partial scores are considered as false positive.
    fp = len(remaining_sys_ids) - greedy_tp

    EvalState.doc_mention_scores.append((greedy_tp, fp, zip(greedy_attribute_tps, attribute_based_fps),
                                         num_gold_predictions, num_system_predictions, doc_id))

    # Select a computed mapping, we currently select the mapping based on mention type. This means that in order to get
    # coreference right, your mention type should also be right. This can be changed by change Config.coref_criteria
    # settings.
    mention_mapping = None
    type_mapping = None
    for attribute_comb_index, attribute_comb in enumerate(all_attribute_combinations):
        if attribute_comb == Config.coref_criteria:
            mention_mapping = greedy_all_attribute_mapping[attribute_comb_index]
            logger.debug("Select mapping that matches criteria [%s]" % (Config.coref_criteria[0][1]))
        if attribute_comb[0][1] == "mention_type":
            type_mapping = greedy_all_attribute_mapping[attribute_comb_index]

    if Config.coref_criteria == "span_only":
        mention_mapping = greedy_mention_only_mapping

    if mention_mapping is None:
        # In case when we don't do attribute scoring.
        mention_mapping = greedy_mention_only_mapping

    # Evaluate how the performance of each type.
    scorer_util.per_type_eval(system_mention_table, gold_mention_table, type_mapping)

    ## Evaluate coreference links.
    if coref_out is not None:

        for cluster in example["predicted_clusters"]:
            cluster_char = []
            for event in cluster:
                # char_tuple = system_tkn_char_map[tuple(event)]
                cluster_char.append(system_tkn_char_map[tuple(event)])

            candidate_clusters.append(cluster_char)

        gold_corefs = get_coref_lines(gold_clusters, gold_id_map, remaining_gold_ids)
        sys_corefs = get_coref_lines(candidate_clusters, system_id_map, remaining_sys_ids)


    ## Evaluate coreference links.
    #if coref_out is not None:
        logger.debug("Start preparing coreference files.")

        # Prepare CoNLL style coreference input for this document.
        conll_converter = ConllEvaluator(doc_id, system_id, sys_id_2_text, gold_id_2_text)
        gold_conll_lines, sys_conll_lines = conll_converter.prepare_conll_lines(gold_corefs, sys_corefs,
                                                                                gold_mention_table,
                                                                                system_mention_table,
                                                                                mention_mapping,
                                                                                MutableConfig.coref_mention_threshold)

        # If we are selecting among multiple mappings, it is easy to write in our file.
        write_mode = 'w' if EvalState.claim_write_flag() else 'a'
        g_conll_out = open(Config.conll_gold_file, write_mode)
        s_conll_out = open(Config.conll_sys_file, write_mode)
        g_conll_out.writelines(gold_conll_lines)
        s_conll_out.writelines(sys_conll_lines)

        if diff_out is not None:
            scorer_util.write_gold_and_system_corefs(diff_out, gold_corefs, sys_corefs, gold_id_2_text, sys_id_2_text)

    scorer_util.write_if_provided(diff_out, Config.eod_marker + " " + "\n")

    return True

def get_coref_lines(coref_lines, event_id_map, remaining_event_ids):
    # Parse relations.
    clusters = []
    for cid, cluster in enumerate(coref_lines):
        relation = "Coref"
        rid = cid
        arguments = []
        for eid in range(0, len(cluster)):
            e = cluster[eid]
            arguments.append(event_id_map[(e[0], e[1], e[2])].get())

        if len(arguments) > 1:
            clusters.append([relation, rid, arguments])

    if EvalState.white_listed_types:
        clusters = utils.filter_relations(clusters, remaining_event_ids)

    return clusters

def parse_characters(s):
    """
    Method to parse the character based span
    :param s:
    """
    span_strs = s.split(Config.span_seperator)
    characters = []
    for span_strs in span_strs:
        span = list(map(int, span_strs.split(Config.span_joiner)))
        for c in range(span[0], span[1]):
            characters.append(c)

    return characters

def output_scorer_format(output_path, coref, examples):

    output = []


    for (tensorized_example, example) in examples:
        extra_mentions = {}
        mention_id_map = {}
        mention_subtype_map = {}

        filename = example['doc_key']
        filename = filename[filename.rfind('/')+1:]

        full_candidate_char = []
        full_subtype = []
        full_cluster = []

        tokens = []
        for sentence in example['sentences']:
            tokens.extend(sentence)
        #raw_input(len(tokens))

        output.append('#BeginOfDocument\t' + filename)

        for i in range(0, len(example['candidate_char'])):
            subtype = example['subtypes'][i]
            event_key = (example['candidate_char'][i][0], example['candidate_char'][i][1])
            mention_id_map[event_key] = "E"+str(i)
            mention_subtype_map[event_key] = subtype
            #print(tokens[event_key[0]:event_key[1]+1])
            trigger = (" ").join(tokens[event_key[0]:event_key[1]+1])

            if '#' not in subtype:
                eventline = ["UTD", filename, "E"+str(i), str(example['candidate_char'][i][2]) + "," + str(example['candidate_char'][i][3]+1), trigger, subtype,"actual"]
                output.append(("\t").join(eventline))
                full_candidate_char.append(example['candidate_char'][i])
                full_subtype.append(subtype) 
            else:

                subtypes = subtype.split("#")
                eventline1 = ["UTD", filename, "E"+str(i), str(example['candidate_char'][i][2]) + "," + str(example['candidate_char'][i][3]+1), trigger, subtypes[0],"actual"]
                eventline2 = ["UTD", filename, "E"+str(i + 1000), str(example['candidate_char'][i][2]) + "," + str(example['candidate_char'][i][3]+1), trigger, subtypes[1],"actual"]

                output.append(("\t").join(eventline1))
                output.append(("\t").join(eventline2))

                full_candidate_char.append(example['candidate_char'][i])
                full_candidate_char.append(example['candidate_char'][i])

                full_subtype.append(subtypes[0]) 
                full_subtype.append(subtypes[1]) 

                extra_mentions["E"+str(i)] = "E"+str(i + 1000)

        if coref:
            for i, cluster in enumerate(example['predicted_clusters']):
                tmp = "@Coreference\tC"+str(i)+"\t"
                cids = []
                subtype = ""
                extraflag = False


                for event in cluster:
                    eventkey = (event[0], event[1])
                    if eventkey in mention_id_map and "#" not in mention_subtype_map[eventkey]:
                        subtype = mention_subtype_map[eventkey]

                        if mention_id_map[eventkey] in extra_mentions:
                            extraflag = True

                if not extraflag or (extraflag and subtype == ""):
                    for event in cluster:
                        eventkey = (event[0], event[1])
                        if eventkey in mention_id_map:
                            cids.append(mention_id_map[eventkey])
                        else:
                            print(filename + "\t"+ str(event[0])+","+str(event[1]))
                            raw_input()
                elif extraflag and subtype != "":

                    for event in cluster:
                        eventkey = (event[0], event[1])
                        if eventkey in mention_id_map:
                            esubtype = mention_subtype_map[eventkey]

                            if "#" not in esubtype:
                                cids.append(mention_id_map[eventkey])
                            else:
                                if esubtype.split("#")[0]==subtype:
                                    cids.append(mention_id_map[eventkey])
                                else:
                                    cids.append(extra_mentions[mention_id_map[eventkey]])

                tmp = tmp +",".join(cids)
                output.append(tmp)

                if set(cids) <= extra_mentions.keys():
                    tmp = "@Coreference\tC"+str(i+1000)+"\t"
                    extracids = []
                    for event in cluster:
                        eventkey = (event[0], event[1])
                        extracids.append(extra_mentions[mention_id_map[eventkey]])

                    tmp = tmp +",".join(extracids)
                    output.append(tmp)

        output.append("#EndOfDocument")

        with open(output_path, "w") as f:
            for line in output:
                #print(line)
                f.write((line + "\n"))



if __name__ == "__main__":
    main()
