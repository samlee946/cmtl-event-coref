# -*- coding: utf-8 -*-
import sys
import json
import collections
import coref_scorer
#from corenlp_xml_reader import AnnotatedText as A
import os
from collections import defaultdict


def output_scorer_format_gold_nonsingleton(example_map, coref, output_path, mention_field, offset_map, subtoken_map, quote_path, propagation=False, eval_span=False):
  quote_map = defaultdict(list)

  print(quote_path)
  with open(quote_path) as f:
    lines = f.readlines()
    for line in lines:
      item = line.strip().split("\t")
      #print(item)
      quote_map[item[0]].extend(range(int(item[1]), int(item[2])))

  print(len(quote_map))

  all_mention_subtype_map = {}
  output = []
  for dockey, doc in example_map.items():
    filename = dockey[dockey.rindex("/")+1:] + ".xml"
    #print(filename)
    quote_region = set(quote_map[filename])
    #print(quote_region)
    #raw_input(" ")

    tokens = []
    for s in doc['raw_sentences']:
      tokens.extend(s)

    # fname = dockey[dockey.rindex("/")+1:]
    # xml = load_stanford_xml(doc, dockey_xmlkey_map[fname])
    # xml_tokens = []
    # for sentence in xml.sentences:
    #   xml_tokens.extend(sentence['tokens'])
    xml_tokens = offset_map[dockey]['offsets']

    filename = doc['doc_key']
    filename = filename[filename.rfind('/')+1:]
    output.append('#BeginOfDocument\t' + filename)
    # for key, value in doc.items():
    #   print(key, value)
    #   raw_input(" ")

    mentions = []
    subtypes = []
    realis = []
    
    for c in doc["gold_clusters"]:
      if len(c) == 1:
        continue
      for m in c:
        mentions.append((m[0], m[1]))
        subtype = m[2]

        subtypes.append(subtype)


    extra_mentions = {}
    mention_id_map = {}
    mention_subtype_map = {}

    trigger_output = []
    coref_output = []

    i = 0
    for idx, (mention, subtype) in enumerate(zip(mentions, subtypes)):
      mstart, mend = mention
      start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]

      if subtype == "null":
        continue

      if not ("_" in subtype or subtype == "event"):
        continue

      # if "_" in subtype or subtype == "event":
      #   subtype = "event"
      # else:
      #   subtype = "entity" 

      if xml_tokens[start][0] in quote_region and xml_tokens[end][1] in quote_region:
        #print("in quote skip:", mention, subtype)
        continue

      if (start, end) in mention_id_map:
        #print("duplication:", i, mstart, mend, start, end, mention_id_map[(start, end)])
        continue

      mention_id_map[(start,end)] = "E"+str(i)
      mention_subtype_map[(start,end)] = subtype

      # print(mstart, mend, start, end, mention_id_map[(start,end)])

      trigger = (" ").join(tokens[start:end+1])

      realis_value = 'actual'
      if realis:
        realis_value = realis[idx]

      if "#" in subtype:
        #for s in subtype.split("#"):
          #sys_mentions.append((mention[0], mention[1], s))

        if not eval_span:
          eventline = ["UTD", filename, "E"+str(i), str(xml_tokens[start][0]) + "," + str(xml_tokens[end][1]), trigger, subtype.split("#")[0],realis_value]
          trigger_output.append(("\t").join(eventline))
          eventline = ["UTD", filename, "E"+str(i + 1000), str(xml_tokens[start][0]) + "," + str(xml_tokens[end][1]), trigger, subtype.split("#")[1],realis_value]
          trigger_output.append(("\t").join(eventline))
        else:
          if "_" in subtype or subtype == "event":
            span_type = "event"
          else:
            span_type = "entity" 
           
          eventline = ["UTD", filename, "E"+str(i), str(xml_tokens[start][0]) + "," + str(xml_tokens[end][1]), trigger, span_type, realis_value]
          trigger_output.append(("\t").join(eventline))
          eventline = ["UTD", filename, "E"+str(i + 1000), str(xml_tokens[start][0]) + "," + str(xml_tokens[end][1]), trigger, span_type, realis_value]
          trigger_output.append(("\t").join(eventline))

        i += 1
          
      else:
        #sys_mentions.append((mention[0], mention[1], subtype)) 
        #print(str(xml_tokens[mention[0]]["character_offset_begin"]) + "," + str(xml_tokens[mention[1]]["character_offset_end"]+1))

        if not eval_span:
          eventline = ["UTD", filename, "E"+str(i), str(xml_tokens[start][0]) + "," + str(xml_tokens[end][1]), trigger, subtype, realis_value]
        else:
          if "_" in subtype or subtype == "event":
            span_type = "event"
          else:
            span_type = "entity" 
          eventline = ["UTD", filename, "E"+str(i), str(xml_tokens[start][0]) + "," + str(xml_tokens[end][1]), trigger, span_type, realis_value]


        outputstr = ("\t").join(eventline)
        trigger_output.append(outputstr)
        i += 1

    if coref:
      exist_mentions = set()
      for i, cluster in enumerate(doc['gold_clusters']):
        if len(cluster) == 1:
          continue
        tmp = "@Coreference\tC"+str(i)+"\t"
        cids = []
        subtype = ""
        extraflag = False

        for event in cluster:
          mstart, mend, _ = event
          start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]
          eventkey = (start, end)
          if eventkey in mention_id_map and "#" not in mention_subtype_map[eventkey]:
            subtype = mention_subtype_map[eventkey]

            if mention_id_map[eventkey] in extra_mentions:
                extraflag = True

        if not extraflag or (extraflag and subtype == ""):
          for event in cluster:
            mstart, mend, _ = event
            start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]

            eventkey = (start, end)

            if eventkey in exist_mentions:
              continue

            if eventkey in mention_id_map:
              cids.append(mention_id_map[eventkey])
              exist_mentions.add(eventkey)
            
        elif extraflag and subtype != "":

          for event in cluster:
            mstart, mend, _ = event
            start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]

            eventkey = (start, end)

            if eventkey in exist_mentions:
              continue

            if eventkey in mention_id_map:
              esubtype = mention_subtype_map[eventkey]

              if "#" not in esubtype:
                cids.append(mention_id_map[eventkey])
              else:
                if esubtype.split("#")[0]==subtype:
                  cids.append(mention_id_map[eventkey])
                else:
                  cids.append(extra_mentions[mention_id_map[eventkey]])

              exist_mentions.add(eventkey)

        if not cids:
          #p#rint(i, " all mentions has null type")
          continue

        if len(cids) == 1:
          #print("singleton cluster:", i, filename)
          continue
          #raw_input(" ")

        tmp = tmp +",".join(cids)
        coref_output.append(tmp)

        if set(cids) <= extra_mentions.keys():
          tmp = "@Coreference\tC"+str(i+1000)+"\t"
          extracids = []
          for event in cluster:
            mstart, mend, _ = event
            start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]
            eventkey = (start, end)

            if eventkey in mention_id_map:
              extracids.append(extra_mentions[mention_id_map[eventkey]])


          tmp = tmp +",".join(extracids)
          coref_output.append(tmp)

    output.extend(trigger_output)
    output.extend(coref_output)
    output.append("#EndOfDocument")

    all_mention_subtype_map[filename] = mention_subtype_map

  if propagation:
    output_path = output_path + "-prop"

  with open(output_path, "w") as f:
    for line in output:
      f.write(line)
      f.write("\n")

  return all_mention_subtype_map

def output_scorer_format(example_map, coref, output_path, mention_field, offset_map, subtoken_map, quote_path, propagation=False, eval_span=False, eval_nonsingleton=False):
  quote_map = defaultdict(list)
  if quote_path != "None":
    print(quote_path)
    with open(quote_path) as f:
      lines = f.readlines()
      for line in lines:
        item = line.strip().split("\t")
        #print(item)
        quote_map[item[0]].extend(range(int(item[1]), int(item[2])))

    print(len(quote_map))

  all_mention_subtype_map = {}
  output = []
  for dockey, doc in example_map.items():
    filename = dockey[dockey.rindex("/")+1:] + ".xml"
    #print(filename)
    quote_region = set(quote_map[filename])
    #print(quote_region)
    #raw_input(" ")

    tokens = []
    for s in doc['raw_sentences']:
      tokens.extend(s)

    # fname = dockey[dockey.rindex("/")+1:]
    # xml = load_stanford_xml(doc, dockey_xmlkey_map[fname])
    # xml_tokens = []
    # for sentence in xml.sentences:
    #   xml_tokens.extend(sentence['tokens'])
    xml_tokens = {}
    if offset_map:
      xml_tokens = offset_map[dockey]['offsets']

    filename = doc['doc_key']
    filename = filename[filename.rfind('/')+1:]
    output.append('#BeginOfDocument\t' + filename)
    # for key, value in doc.items():
    #   print(key, value)
    #   raw_input(" ")

    mentions = []
    subtypes = []
    realis = []
    if mention_field == 'candidates':
      mentions = doc['candidates']
      subtypes = doc['subtypes']
    elif mention_field == "top_spans":
      mentions = doc['top_spans']
      if 'top_spans_subtypes' in doc:
        subtypes = doc['top_spans_subtypes']
      else:
        subtypes = []
      if 'top_spans_realis' in doc:
        realis = doc['top_spans_realis']
    elif mention_field == "top_spans_event":
      mentions = doc['top_spans_event']
      subtypes = doc['top_spans_event_subtypes']
      if 'top_spans_realis' in doc:
        realis = doc['top_spans_realis']


    extra_mentions = {}
    mention_id_map = {}
    mention_subtype_map = {}

    trigger_output = []
    trigger_output_map = collections.defaultdict(list)
    coref_output = []


    i = 0
    for idx, (mention, subtype) in enumerate(zip(mentions, subtypes)):
      mstart, mend = mention
      start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]

      if subtype == "null":
        continue

      if xml_tokens and xml_tokens[start][0] in quote_region and xml_tokens[end][1] in quote_region:
        #print("in quote skip:", mention, subtype)
        continue

      if (start, end) in mention_id_map:
        #print("duplication:", i, mstart, mend, start, end, mention_id_map[(start, end)])
        continue

      mention_id_map[(start,end)] = "E"+str(i)
      mention_subtype_map[(start,end)] = subtype

      # print(mstart, mend, start, end, mention_id_map[(start,end)])

      trigger = (" ").join(tokens[start:end+1])

      realis_value = 'actual'
      if realis:
        realis_value = realis[idx]

      if xml_tokens:
        tstart = xml_tokens[start][0]
        tend = xml_tokens[end][1]
      else:
        tstart = start
        tend = end+1

      if "#" in subtype:

        if not eval_span:
          eventline = ["UTD", filename, "E"+str(i), str(tstart) + "," + str(tend), trigger, subtype.split("#")[0],realis_value]
          trigger_output_map[(start, end)].append(("\t").join(eventline))
          #trigger_output.append(("\t").join(eventline))
          eventline = ["UTD", filename, "E"+str(i + 1000), str(tstart) + "," + str(tend), trigger, subtype.split("#")[1],realis_value]
          #trigger_output.append(("\t").join(eventline))
          trigger_output_map[(start, end)].append(("\t").join(eventline))
        else:
          if "_" in subtype or subtype == "event":
            span_type = "event"
          else:
            span_type = "entity" 
           
          eventline = ["UTD", filename, "E"+str(i), str(tstart) + "," + str(tend), trigger, span_type, realis_value]
          #trigger_output.append(("\t").join(eventline))
          trigger_output_map[(start, end)].append(("\t").join(eventline))
          eventline = ["UTD", filename, "E"+str(i + 1000), str(tstart) + "," + str(tend), trigger, span_type, realis_value]
          #trigger_output.append(("\t").join(eventline))
          trigger_output_map[(start, end)].append(("\t").join(eventline))

        extra_mentions["E"+str(i)] = "E"+str(i + 1000)

        i += 1
          
      else:
        #sys_mentions.append((mention[0], mention[1], subtype)) 
        #print(str(xml_tokens[mention[0]]["character_offset_begin"]) + "," + str(xml_tokens[mention[1]]["character_offset_end"]+1))
        if not eval_span:
          eventline = ["UTD", filename, "E"+str(i), str(tstart) + "," + str(tend), trigger, subtype, realis_value]  
        else:
          if "_" in subtype or subtype == "event":
            span_type = "event"
          else:
            span_type = "entity" 
          eventline = ["UTD", filename, "E"+str(i), str(tstart) + "," + str(tend), trigger, span_type, realis_value]
          
        outputstr = ("\t").join(eventline)
        #trigger_output.append(outputstr)
        trigger_output_map[(start, end)].append(outputstr)

        i += 1

    if coref:
      exist_mentions = set()
      for i, cluster in enumerate(doc['predicted_clusters']):
        tmp = "@Coreference\tC"+str(i)+"\t"
        cids = []
        subtype = ""
        extraflag = False

        for event in cluster:
          mstart, mend, _ = event
          start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]
          eventkey = (start, end)
          if eventkey in mention_id_map and "#" not in mention_subtype_map[eventkey]:
            subtype = mention_subtype_map[eventkey]

            if mention_id_map[eventkey] in extra_mentions:
                extraflag = True

        if not extraflag or (extraflag and subtype == ""):
          for event in cluster:
            mstart, mend, _ = event
            start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]

            eventkey = (start, end)

            if eventkey in exist_mentions:
              continue

            if eventkey in mention_id_map:
              cids.append(mention_id_map[eventkey])
              exist_mentions.add(eventkey)
            
        elif extraflag and subtype != "":

          for event in cluster:
            mstart, mend, _ = event
            start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]

            eventkey = (start, end)

            if eventkey in exist_mentions:
              continue

            if eventkey in mention_id_map:
              esubtype = mention_subtype_map[eventkey]

              if "#" not in esubtype:
                cids.append(mention_id_map[eventkey])
              else:
                if esubtype.split("#")[0]==subtype:
                  cids.append(mention_id_map[eventkey])
                else:
                  cids.append(extra_mentions[mention_id_map[eventkey]])

              exist_mentions.add(eventkey)

        if not cids:
          #p#rint(i, " all mentions has null type")
          continue

        if len(cids) == 1:
          #print("singleton cluster:", i, filename)
          continue
          #raw_input(" ")

        tmp = tmp +",".join(cids)
        coref_output.append(tmp)

        if set(cids) <= extra_mentions.keys():
          tmp = "@Coreference\tC"+str(i+1000)+"\t"
          extracids = []
          for event in cluster:
            mstart, mend, _ = event
            start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]
            eventkey = (start, end)

            if eventkey in mention_id_map:
              extracids.append(extra_mentions[mention_id_map[eventkey]])


          tmp = tmp +",".join(extracids)
          coref_output.append(tmp)

    if eval_nonsingleton:
      for i, cluster in enumerate(doc['predicted_clusters']):
        if len(cluster) <= 1:
          continue

        for event in cluster:
          mstart, mend, _ = event
          start, end = subtoken_map[doc['doc_key']][mstart], subtoken_map[doc['doc_key']][mend]
          eventkey = (start, end)

          trigger_output.extend(trigger_output_map[eventkey])
    else:
      for value in trigger_output_map.values():
        trigger_output.extend(value)

    output.extend(trigger_output)
    output.extend(coref_output)
    output.append("#EndOfDocument")

    all_mention_subtype_map[filename] = mention_subtype_map

  if propagation:
    output_path = output_path + "-prop"

  with open(output_path, "w") as f:
    for line in output:
      f.write(line)
      f.write("\n")

  return all_mention_subtype_map


def get_gold_event_type_info(doc):
  event_type_map = {}

  gold_event_mentions = []
  for m in doc['gold_event_mentions']:
    gold_event_mentions.append([m[0],m[1],m[2]])

  for start, end, subtype in gold_event_mentions:
    if (start, end) in event_type_map and event_type_map[(start, end)] != subtype:
      #print("gold event:", doc["doc_key"], start, end, subtype, event_type_map[(start, end)])
      continue
    
    event_type_map[(start, end)] = subtype.replace("null", "NONE")

  return event_type_map

def prf(right_events, gold_events,sys_events):
  right_num = len(right_events)
  golden_num = len(gold_events)
  predict_num = len(sys_events)

  #print(golden_full)
  #input(" ")

  if predict_num == 0:
      precision = -1
  else:
      precision =  (right_num+0.0)/predict_num
  if golden_num == 0:
      recall = -1
  else:
      recall = (right_num+0.0)/golden_num
  if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
      f_measure = -1
  else:
      f_measure = 2*precision*recall/(precision+recall)

  return precision, recall, f_measure

def get_entity_type_info(doc, use_gold, use_entity_head):
  # get entity info
  entity_type_map = {}

  # if use_gold:
  #   entities = doc['gold_entity_mentions']
  # else:
  #   entities = doc['predicted_entity_mentions_crf']
  #   time_entities = doc['predicted_entity_mentions']
  entities = []
  for cid, cluster in enumerate(doc["gold_clusters"]):
    for mention in cluster:
      start, end, entype = mention
      if "_" not in entype:
        entities.append((start, end, entype))

  for entity in entities:
    #eid = entity[-2]
    entype = entity[-1]
    if entype == "null":
      entype = "NONE"

    en_start = entity[0] #start token index
    en_end = entity[1] #end token index

    #if use_entity_head:
    #  en_start = entity[2]
    #  en_end = entity[3]
    
    for tid in range(en_start, en_end + 1):
      entity_type_map[tid] = entype

  if not use_gold:
    # get time entity mentions from stanford coreNLP
    for entity in time_entities:
      #eid = entity[-2]
      entype = entity[-1]
      if entype != "time":
        continue

      #print(entity)

      en_start = entity[0] #start token index
      en_end = entity[1] #end token index

      #if use_entity_head:
      #  en_start = entity[2]
      #  en_end = entity[3]
      
      for tid in range(en_start, en_end + 1):
        entity_type_map[tid] = entype

  return entity_type_map


def get_combined_gold_arguments(doc, gold_event_type_map, use_entity_head, role_subtype_map, entity_event_map):
  gold_entity_type_map = get_entity_type_info(doc, True, use_entity_head)
  gold_argument_subtype_map = collections.defaultdict(list)
  for arg in doc["gold_arguments"]:
    estart = arg[0]
    eend = arg[1]

    etype = gold_event_type_map[(estart, eend)]

    arg_role = arg[6]
    arg_start = arg[2]
    arg_end = arg[3]

    if use_entity_head:
      arg_start = arg[4]
      arg_end = arg[5]

    gold_argument_subtype_map[(estart, eend, etype, arg_start, arg_end)].append(arg_role.lower())

  gold_args = []
  gold_args_wo_subtype = []
  gold_arg_roles = []
  for key, arg_roles in gold_argument_subtype_map.items():
    estart, eend, etype, arg_start, arg_end = key
    arg_roles = list(set(arg_roles))
    sorted(arg_roles)
    arg_role_str = "#".join(arg_roles)

    gold_arg_roles.append(arg_role_str)
    gold_args.append((estart, eend, etype, arg_start, arg_end, arg_role_str))
    gold_args_wo_subtype.append((estart, eend, arg_start, arg_end, arg_role_str))
    role_subtype_map[arg_role_str].add(etype)
    entity_type = gold_entity_type_map[arg_start]
    entity_event_map[entity_type].add(etype)

  return gold_args, gold_args_wo_subtype, gold_arg_roles

def extend_gold_arguments(example, gold_args):

  mention_set = set()
  span_cluster_map = {}
  for cluster_id, cluster in enumerate(example["gold_clusters"]):
    for mention in cluster:
      if (mention[0], mention[1]) in mention_set:
          continue
      mention_set.add((mention[0], mention[1]))
      span_cluster_map[(mention[0], mention[1])] = cluster_id

  exist_args = set(gold_args)
  extra_args = []
  # token to sentence id map
  token_sid_map = {}
  tokens = []
  tid = 0
  for sid, sentence in enumerate(example["sentences"]):
    tokens.extend(sentence)
    for token in sentence:
      token_sid_map[tid] = sid
      tid +=1 

  for (mstart, mend, mtype, start, end, role) in gold_args:
    #print("existing:", mstart, mend, start, end, role, " ".join(tokens[start: end+1]))
    if (start, end) not in span_cluster_map:
        #print("not in gold mention map")
        continue

    entity_id = span_cluster_map[(start, end)]
    sid = token_sid_map[start]
    corefed_entities = example["gold_clusters"][entity_id]
    for en in corefed_entities:
        estart = en[0]
        eend = en[1]
        esid = token_sid_map[estart]
        if sid != esid:
            continue

        if estart == start and eend == end:
            continue

        if (mstart, mend, mtype, estart, eend, role) not in exist_args:
          extra_arg = (mstart, mend, mtype, estart, eend, role)
          if extra_arg not in exist_args:
            extra_args.append(extra_arg)
          else:
            print("extra in exist:", mstart, mend, mtype, estart, eend, role)

  gold_args.extend(extra_args)
  gold_args_wo_subtype = []
  gold_arg_roles = []
  for (mstart, mend, mtype, start, end, role) in gold_args:
    gold_args_wo_subtype.append((mstart, mend, start, end, role))
    gold_arg_roles.append(role)

  return gold_args, gold_args_wo_subtype, gold_arg_roles

def get_ner_fmeasure_per_type(gold_list, predict_list, right_list):
    type_score_map = {}

    gold_type_map = defaultdict(list)
    for item in gold_list:
        label=item[-1]
        #span = item[0:item.index(']')+1]
        gold_type_map[label].append(item)
        #gold_type_map['span'].append(span)

    pred_type_map = defaultdict(list)
    for item in predict_list:
        label=item[-1]
        pred_type_map[label].append(item)

        #span = item[0:item.index(']')+1]
        #pred_type_map['span'].append(span)

    right_type_map = defaultdict(list)
    for item in right_list:
        label=item[-1]
        right_type_map[label].append(item)

    for key in gold_type_map.keys():
        golden_num = len(gold_type_map[key])
        predict_num = len(pred_type_map[key])
        right_num = len(right_type_map[key])

        if predict_num == 0:
            precision = -1
        else:
            precision =  (right_num+0.0)/predict_num
        if golden_num == 0:
            recall = -1
        else:
            recall = (right_num+0.0)/golden_num
        if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
            f_measure = -1
        else:
            f_measure = 2*precision*recall/(precision+recall)
        #accuracy = (right_tag+0.0)/all_tag

        type_score_map[key] = [precision, recall, f_measure, golden_num, predict_num, right_num]
        arg_results_str = "\t".join([str(a) for a in [key, precision, recall, f_measure, golden_num, predict_num, right_num]])
        print(arg_results_str)

    return type_score_map

def replace_argument_predictions_with_coref(doc, predicted_arguments):
  # this function only applies to 
  # get span coref map, key: span, value: cluster id
  # before_arg_role_counter: number of prediction per role
  # after_arg_role_counter:
  span_coref_map = {}
  cluster_map = {} #key: cid, value: cluster
  for cid, cluster in enumerate(doc["predicted_clusters"]):
    for mention in cluster:
      span_coref_map[tuple(mention)] = cid
    cluster_map[cid] = cluster

  new_pred_args = []
  for arg in predicted_arguments:
    ev_start, ev_end, etype, arg_start, arg_end, arg_role = arg

    if (arg_start, arg_end) not in span_coref_map: # arg is singleton
      new_pred_args.append(arg)
      continue

    cid = span_coref_map[(arg_start, arg_end)]
    cluster = cluster_map[cid]
    m0_start, m0_end = cluster[0]
    closest_mention = cluster[0]
    min_dist = min(min(abs(ev_start - m0_end), abs(ev_end - m0_end)), abs(m0_start - ev_end))

    for mention in cluster:
      arg_start, arg_end = mention
      dist = min(min(abs(ev_start - arg_end), abs(ev_end - arg_end)), abs(arg_start - ev_end))

      if dist < min_dist:
        closest_mention = mention
        min_dist = dist

    new_pred_args.append((ev_start, ev_end, etype, closest_mention[0], closest_mention[1], arg_role))
  return new_pred_args

def count_redundent_argument_predictions(doc, predicted_arguments):
  arg_map = defaultdict(list) # key: tuple(event_mention_start, event_mention_end, etype, arg_role), value: list of predicted arguments
  for arg in predicted_arguments:
    ev_start, ev_end, etype, arg_start, arg_end, arg_role = arg
    arg_map[(ev_start, ev_end, etype, arg_role)].append((arg_start, arg_end))

  tokens = []
  for sentence in doc["sentences"]:
    tokens.extend(sentence)

  count = 0
  for key, value in arg_map.items():
    ev_start, ev_end, etype, arg_role = key
    if len(value) >1:
      count += 1
  #     print(ev_start, ev_end, etype, arg_role, tokens[ev_start: ev_end+1])
  #     for arg in value:
  #       arg_start, arg_end = arg
  #       print(arg_start, arg_end, tokens[arg_start: arg_end+1])
  # raw_input(" ")

  return count

def remove_redundent_argument_predictions(doc, predicted_arguments, before_arg_role_counter, after_arg_role_counter):
  # this function only applies to 
  # get span coref map, key: span, value: cluster id
  # before_arg_role_counter: number of prediction per role
  # after_arg_role_counter:
  span_coref_map = {}
  for cid, cluster in enumerate(doc["predicted_clusters"]):
    for mention in cluster:
      span_coref_map[tuple(mention)] = cid

  arg_map = defaultdict(list) # key: tuple(event_mention_start, event_mention_end, etype, arg_role), value: list of predicted arguments
  for arg in predicted_arguments:
    ev_start, ev_end, etype, arg_start, arg_end, arg_role = arg
    arg_map[(ev_start, ev_end, etype, arg_role)].append((arg_start, arg_end))

  reduced_arg_map = defaultdict(list)
  for key, value in arg_map.items():
    ev_start, ev_end, etype, arg_role = key
    before_arg_role_counter[(arg_role, len(value))] += 1

    if len(value) == 1:
      reduced_arg_map[key].extend(value)
      continue

    closest_coref_arg_map = {} #key: cid, value: closet arg, dist
    for arg in value:
      arg_start, arg_end = arg
      if arg not in span_coref_map: # arg mention is singleton
        reduced_arg_map[key].append(arg)
      else:
        cid = span_coref_map[arg]
        if cid not in closest_coref_arg_map:
          dist = min(min(abs(ev_start - arg_end), abs(ev_end - arg_end)), abs(arg_start - ev_end))
          closest_coref_arg_map[cid] = (arg, dist)
        else:
          curr_arg = closest_coref_arg_map[cid]
          dist = min(min(abs(ev_start - arg_end), abs(ev_end - arg_end)), abs(arg_start - ev_end))

          #print(cid, "curr:", curr_arg, "new:", arg, dist)
          #raw_input(" ")

          if curr_arg[1] > dist:
            closest_coref_arg_map[cid] = (arg, dist)

    #print("closest_coref_arg_map:")
    #print(closest_coref_arg_map)

    for cid, (arg, dist) in closest_coref_arg_map.items():
      reduced_arg_map[key].append(arg)

  for key, value in reduced_arg_map.items():
    ev_start, ev_end, etype, arg_role = key
    after_arg_role_counter[(arg_role, len(value))] += 1

  # print("before removing redundency")
  # for key, value in arg_map.items():
  #   print(key, value)

  print("after removing redundency")
  for key, value in reduced_arg_map.items():
     print(key, value)
  reduced_arg = []
  for key, value in reduced_arg_map.items():
    ev_start, ev_end, etype, arg_role = key
    for mention in value:
      arg_start, arg_end = mention
      reduced_arg.append((ev_start, ev_end, etype, arg_start, arg_end, arg_role))

  #raw_input(" ")
  return reduced_arg

def evaluate_argument(example_map, all_mention_subtype_map, use_entity_head, postprocessing, extend_arg=False):
  gold_args_full = []
  gold_args_wo_full = []
  pred_args_full = []
  pred_args_wo_full = []
  right_args_full = []
  right_args_wo_full = []
  arg_roles_full = []
  role_subtype_map_2 = collections.defaultdict(set)
  entity_event_map = collections.defaultdict(set)

  before_arg_role_counter = collections.Counter()
  after_arg_role_counter = collections.Counter()
  total_redundent_count = 0

  role_subtype_map = {u'origin': set([u'movement_transportperson', u'movement_transportartifact']), \
  u'person#entity': set([u'movement_transportperson', u'personnel_endposition']), \
  u'origin#person#agent': set([u'movement_transportperson']), \
  u'money': set([u'transaction_transfermoney#transaction_transferownership', u'transaction_transfermoney', u'transaction_transferownership']), \
  u'giver#agent': set([u'movement_transportartifact']), \
  u'agent': set([u'conflict_attack#life_die', u'life_die', u'justice_arrestjail', u'personnel_elect', u'manufacture_artifact', u'conflict_attack#life_injure', u'movement_transportperson', u'life_injure', u'movement_transportartifact', u'conflict_attack']), \
  u'entity': set([u'contact_meet', u'contact_correspondence', u'conflict_demonstrate', u'contact_contact', u'personnel_startposition', u'contact_broadcast', u'personnel_endposition', u'movement_transportperson']), \
  u'instrument#agent': set([u'movement_transportperson']), \
  u'place#entity': set([u'contact_broadcast']), \
  u'victim': set([u'conflict_attack#life_die', u'life_injure', u'life_die', u'conflict_attack#life_injure']), \
  u'destination#agent': set([u'movement_transportperson']), u'person#target': set([u'conflict_attack']), \
  u'attacker#agent': set([u'conflict_attack', u'movement_transportperson', u'life_die']), \
  u'origin#destination': set([u'movement_transportperson', u'movement_transportartifact']), \
  u'thing': set([u'transaction_transfermoney#transaction_transferownership', u'transaction_transfermoney', u'transaction_transferownership']), \
  u'origin#agent': set([u'movement_transportperson']), \
  u'destination': set([u'movement_transportperson', u'movement_transportartifact']), \
  u'crime': set([u'justice_arrestjail']), \
  u'beneficiary': set([u'transaction_transfermoney#transaction_transferownership', u'transaction_transfermoney', u'transaction_transferownership', u'transaction_transaction']), \
  u'instrument': set([u'conflict_attack#life_die', u'life_die', u'movement_transportartifact', u'manufacture_artifact', u'conflict_attack', u'life_injure', u'movement_transportperson']), \
  u'destination#place': set([u'movement_transportperson', u'movement_transportartifact']), \
  u'recipient#giver': set([u'transaction_transfermoney#transaction_transferownership', u'transaction_transfermoney', u'transaction_transferownership', u'transaction_transaction']), u'victim#agent': set([u'conflict_attack#life_die', u'life_die']), u'giver': set([u'transaction_transfermoney#transaction_transferownership', u'transaction_transfermoney', u'transaction_transferownership', u'transaction_transaction']), u'origin#destination#place': set([u'conflict_demonstrate']), u'attacker#target': set([u'conflict_attack#life_die', u'conflict_attack']), u'attacker#victim#agent#target': set([u'life_die']), u'target#giver': set([u'conflict_attack']), u'agent#entity': set([u'conflict_demonstrate']), u'artifact': set([u'movement_transportperson', u'manufacture_artifact', u'movement_transportartifact']), u'target#victim': set([u'conflict_attack#life_die', u'conflict_attack', u'life_die']), u'attacker': set([u'conflict_attack#life_die', u'conflict_attack', u'life_die', u'conflict_attack#life_injure']), u'origin#person': set([u'movement_transportperson']), u'attacker#recipient': set([u'conflict_attack']), u'recipient': set([u'transaction_transfermoney#transaction_transferownership', u'transaction_transfermoney', u'transaction_transferownership', u'movement_transportartifact', u'transaction_transaction']), u'person#agent': set([u'movement_transportperson']), u'person#recipient': set([u'transaction_transfermoney']), u'target': set([u'conflict_attack#life_die', u'conflict_attack', u'conflict_attack#life_injure']), u'destination#target': set([u'conflict_attack']), u'person#victim': set([u'life_die']), u'person#position': set([u'personnel_endposition', u'personnel_startposition', u'personnel_elect']), u'person': set([u'movement_transportperson', u'personnel_startposition', u'personnel_endposition', u'personnel_elect', u'justice_arrestjail']), u'audience': set([u'contact_broadcast']), u'place': set([u'transaction_transfermoney#transaction_transferownership', u'contact_meet', u'personnel_startposition', u'conflict_demonstrate', u'contact_correspondence', u'conflict_attack#life_die', u'contact_contact', u'transaction_transfermoney', u'transaction_transferownership', u'life_die', u'justice_arrestjail', u'personnel_elect', u'manufacture_artifact', u'transaction_transaction', u'conflict_attack#life_injure', u'contact_broadcast', u'conflict_attack', u'life_injure', u'personnel_endposition']), u'time': set([u'transaction_transfermoney#transaction_transferownership', u'contact_meet', u'contact_correspondence', u'conflict_demonstrate', u'conflict_attack#life_die', u'contact_contact', u'personnel_startposition', u'transaction_transferownership', u'life_die', u'personnel_elect', u'justice_arrestjail', u'personnel_endposition', u'movement_transportartifact', u'transaction_transaction', u'conflict_attack#life_injure', u'transaction_transfermoney', u'contact_broadcast', u'conflict_attack', u'life_injure', u'manufacture_artifact', u'movement_transportperson']), u'position': set([u'transaction_transfermoney', u'personnel_endposition', u'personnel_startposition', u'life_die', u'personnel_elect'])}

  for doc_key, doc in example_map.items():
    tokens = []
    for sentence in doc["sentences"]:
      tokens.extend(sentence)

    gold_args = []
    gold_args_wo_subtype = []

    gold_event_type_map = get_gold_event_type_info(doc)

    gold_args, gold_args_wo_subtype, arg_roles = get_combined_gold_arguments(doc, gold_event_type_map, use_entity_head, \
      role_subtype_map_2, entity_event_map)
    if extend_arg:
      gold_args, gold_args_wo_subtype, arg_roles = extend_gold_arguments(doc, gold_args)
    arg_roles_full.extend(arg_roles)

    pred_args_raw = [tuple(arg) for arg in doc["predicted_arguments"]]
    # pred_args_raw = []
    # for arg in doc["gold_arguments"]:
    #   estart = arg[0]
    #   eend = arg[1]
    #
    #   etype = gold_event_type_map[(estart, eend)]
    #
    #   arg_role = arg[6]
    #   arg_start = arg[2]
    #   arg_end = arg[3]
    #   pred_args_raw.append((estart, eend, etype, arg_start, arg_end, arg_role))

    pred_args = []
    pred_args_wo_subtype = []
    for arg in pred_args_raw:
      ev_start, ev_end, etype, arg_start, arg_end, arg_role = arg

      if not postprocessing:
        pred_args.append(arg)
        pred_args_wo_subtype.append((ev_start, ev_end, arg_start, arg_end, arg_role))
      else:
        # postprocessing is for error analysis
        compt_role = True

        # remove incompatible event type and argument role  
        if etype != "null":
          compatible_subtypes = role_subtype_map[arg_role]
          print(doc_key, arg, tokens[ev_start:ev_end+1], tokens[arg_start:arg_end+1], "compat:", etype in compatible_subtypes)
          if etype not in compatible_subtypes:
            compt_role = False

        if compt_role:
          pred_args.append(arg)   
          pred_args_wo_subtype.append((ev_start, ev_end, arg_start, arg_end, arg_role))

    redundent_count = count_redundent_argument_predictions(doc, pred_args)
    total_redundent_count += redundent_count

    if postprocessing:
      # remove redundent predictions
      # for each event mention, if there are multiple predictions of the same argument role, 
      # remove corefered mentions and keep the closest mention
      reduced_args = remove_redundent_argument_predictions(doc, pred_args, before_arg_role_counter, after_arg_role_counter)
      pred_args = []
      pred_args_wo_subtype = []
      for arg in reduced_args:
        ev_start, ev_end, etype, arg_start, arg_end, arg_role = arg
        pred_args.append(arg)
        pred_args_wo_subtype.append((ev_start, ev_end, arg_start, arg_end, arg_role))

      # remove wrong event mentions
      temp_args = pred_args
      pred_args = []
      pred_args_wo_subtype = []
      for arg in temp_args:
        ev_start, ev_end, etype, arg_start, arg_end, arg_role = arg
        if (ev_start, ev_end) not in gold_event_type_map:
          #print(ev_start, ev_end, etype, tokens[ev_start:ev_end+1])
          continue
        else:
          if etype != gold_event_type_map[(ev_start, ev_end)]:
            #print(ev_start, ev_end, etype, gold_event_type_map[(ev_start, ev_end)], tokens[ev_start:ev_end+1])
            continue
          else:
            pred_args.append(arg)
            pred_args_wo_subtype.append((ev_start, ev_end, arg_start, arg_end, arg_role))
      #raw_input(" ")

      # replace with the closest coref arg
      # for each event mention, if an argument is predicted, 
      # among all entity mentions of the corresponding entity, the predicted mention is not the closest.
      new_args = replace_argument_predictions_with_coref(doc, pred_args)
      pred_args = []
      pred_args_wo_subtype = []
      for arg in new_args:
        ev_start, ev_end, etype, arg_start, arg_end, arg_role = arg
        pred_args.append(arg)
        pred_args_wo_subtype.append((ev_start, ev_end, arg_start, arg_end, arg_role))

    # if len(gold_args) != len(set(gold_args)):
    #   print(doc_key)
    #   print("gold arg size:", len(gold_args), len(set(gold_args)))

    #   print("gold arg size:", len(gold_args), len(set(gold_args)))
    #   print("pred arg size:", len(pred_args), len(set(pred_args)))
    for arg in gold_args:
      print("gold:", arg)

    # raw_input(" ")

    for arg in pred_args:
      print("pred:", arg)
    #if "ENG_DF_001503_20140801_G00A0GL9N" in doc_key:
    #if "ENG_DF_001503_20110828_G00A0GCON" in doc_key:
    
    right_args = list(set(gold_args).intersection(set(pred_args)))
    right_args_wo_subtype = list(set(gold_args_wo_subtype).intersection(set(pred_args_wo_subtype)))

    # print("right arg size:", len(right_args), len(set(right_args)))
    for arg in right_args:
      print("right:", arg)
    # input(" ")
    # if len(gold_args) != len(right_args):
    #   for arg in sorted(gold_args):
    #     print("gold:", arg)

    #   for arg in sorted(pred_args):
    #     print("pred:", arg)
      # raw_input(" ")

    gold_args_full.extend(list(set(gold_args)))
    gold_args_wo_full.extend(list(set(gold_args_wo_subtype)))
    pred_args_full.extend(list(set(pred_args)))
    pred_args_wo_full.extend(list(set(pred_args_wo_subtype)))
    right_args_full.extend(list(set(right_args)))
    right_args_wo_full.extend(list(set(right_args_wo_subtype)))

  # print(role_subtype_map)
  # for arg in pred_args_full:
  #   compatible_subtypes = role_subtype_map[arg[5]]
  #   if arg[2] not in compatible_subtypes and arg[2] != "null":
  #     print(arg, compatible_subtypes)
  #     raw_input(" ")

  # for key, value in entity_event_map.items():
  #     print(key, len(value), value)

  # input(" ")

  arg_p, arg_r, arg_f = prf(right_args_full, gold_args_full, pred_args_full)
  arg_results = ["arg results:", arg_p, arg_r, arg_f, len(gold_args_full), len(pred_args_full), len(right_args_full)]
  arg_result_str = "\t".join([str(a) for a in arg_results])
  print(arg_result_str)

  per_type_result = get_ner_fmeasure_per_type(gold_args_full, pred_args_full, right_args_full)
  arg_wo_p, arg_wo_r, arg_wo_f = prf(right_args_wo_full, gold_args_wo_full, pred_args_wo_full)
  arg_wo_results = ["arg wo results:", arg_wo_p, arg_wo_r, arg_wo_f, len(gold_args_wo_full), len(pred_args_wo_full), len(right_args_wo_full)]
  arg_wo_str  = "\t".join([str(a) for a in arg_wo_results])
  print(arg_wo_str)

  # print(len(arg_roles_full))
  # arg_role_counter = collections.Counter(arg_roles_full)
  # print(arg_role_counter.most_common())
  # roles = list(arg_role_counter)

  # for role in sorted(roles):
  #   print(role.upper())
  # raw_input(" ")
  print("before_arg_role_counter:")
  print(before_arg_role_counter)

  print("after_arg_role_counter:")
  print(after_arg_role_counter)

  print("total_redundent_count:")
  print(total_redundent_count)

  return [arg_p, arg_r, arg_f, arg_wo_p, arg_wo_r, arg_wo_f]

def same_sentence_realis(mention_field, example_map, is_gold):
  gold_output = []
  for doc_key, doc in example_map.items():
    tokens = []
    sentence_id_map = {}
    tid = 0
    for sid, sentence in enumerate(doc['sentences']):
      for token in sentence:
        sentence_id_map[tid] = sid
        tokens.append(token)
        tid += 1

    gold_mentions = []
    if is_gold:
      for m in doc['gold_event_mentions']:
        realis = m[3]
        gold_mentions.append((m[0], m[1], realis))
    else:
      for m, realis in zip(doc['top_spans'], doc['top_spans_realis']):
        gold_mentions.append((m[0], m[1], realis))
          

    for i in range(0, len(gold_mentions)):
      start_i, end_i, realis_i = gold_mentions[i]
      sid_i = sentence_id_map[start_i]
      for j in range(i+1, len(gold_mentions)):
        start_j, end_j, realis_j = gold_mentions[j]
        if (start_i == start_j) and (end_i == end_j):
          continue

        sid_j = sentence_id_map[start_j]

        if sid_i != sid_j:
          continue

        gold_output.append(realis_i+"#"+realis_j)

  output_counter = collections.Counter(gold_output)
  for item in output_counter.most_common():
    print(item)
  raw_input("")


def same_sentence_error(mention_field, example_map):
  output = []

  for doc_key, doc in example_map.items():
    tokens = []
    sentence_id_map = {}
    tid = 0
    for sid, sentence in enumerate(doc['sentences']):
      for token in sentence:
        sentence_id_map[tid] = sid
        tokens.append(token)
        tid += 1

    gold_mentions = []
    for m in doc['gold_event_mentions']:
      if not m[2].startswith("contact"):
        continue

      #m[2] = m[2][0:m[2].index("_")]

      if "#" in m[2]:
        for subtype in m[2].split("#"):
          gold_mentions.append((m[0], m[1], subtype))
      else:
        gold_mentions.append((m[0], m[1], m[2])) 
    gold_mentions = set(gold_mentions)
    #print(gold_mentions)

    sys_mentions = []
    mentions = []
    subtypes = []
    if mention_field == 'candidates':
      mentions = doc['candidates']
      subtypes = doc['subtypes']
    elif mention_field == "top_spans":
      mentions = doc['top_spans']
      subtypes = doc['top_spans_subtypes']

    for mention, subtype in zip(mentions, subtypes):
      if subtype == "null":
        continue

      if not subtype.startswith("contact"):
        continue

      #subtype = subtype[0:subtype.index("_")]

      if "#" in subtype:
        for s in subtype.split("#"):
          sys_mentions.append((mention[0], mention[1], s))
      else:
        sys_mentions.append((mention[0], mention[1], subtype)) 

    print(doc_key)
    for i in range(0, len(sys_mentions)):
      start_i, end_i, type_i = sys_mentions[i]
      sid_i = sentence_id_map[start_i]
      is_event_i = sys_mentions[i] in gold_mentions
      for j in range(i+1, len(sys_mentions)):
        start_j, end_j, type_j = sys_mentions[j]
        if (start_i == start_j) and (end_i == end_j):
          continue

        sid_j = sentence_id_map[start_j]
        is_event_j = sys_mentions[j] in gold_mentions

        if sid_i != sid_j:
          continue

        if type_i != type_j:
          continue

        print(i, sid_i, sys_mentions[i], tokens[start_i:end_i+1], is_event_i, \
          j, sid_j, sys_mentions[j], tokens[start_j:end_j+1], is_event_j)
        output.append(type_i + "#" + str(is_event_i) + "#" + str(is_event_j))
    
  output_counter = collections.Counter(output)
  #for key, count in dict(output_counter).items():
  #  print(key, count)
  for item in output_counter.most_common():
    print(item)
  raw_input("")

def evaluate_arg_entity_recall(example_map, use_entity_head):
  entity_type_list = []
  with open("entity_event_filler_type_list.txt", "r") as f:
    lines = f.readlines()

    for lid, line in enumerate(lines):
      if lid > 20:
        entity_type_list.append(line.strip())
  #print(entity_type_list)
  #raw_input(" ")

  gold_args_entities_full = []
  pred_entities_full = []
  right_entities_full = []
  role_subtype_map_2 = collections.defaultdict(set)
  entity_event_map = collections.defaultdict(set)

  for doc_key, doc in example_map.items():
    tokens = []
    for sentence in doc["sentences"]:
      tokens.extend(sentence)

    gold_event_type_map = get_gold_event_type_info(doc)

    gold_args, gold_args_wo_subtype, arg_roles = get_combined_gold_arguments(doc, gold_event_type_map, use_entity_head, \
                                                                             role_subtype_map_2, entity_event_map)
    gold_args_entities = [(a[3], a[4]) for a in gold_args]
    gold_args_entities = list(set(gold_args_entities))

    pred_entities = []
    # for mention, subtype in zip(doc["top_spans"], doc["top_spans_subtypes"]):
    #   if subtype not in entity_type_list:
    #     #print(mention, subtype)
    #     continue
    #   pred_entities.append(tuple(mention))

    for mention, subtype in zip(doc["top_spans"], doc["top_spans_subtypes"]):
      #mention = (m[0], m[1])
      #subtype = m[2]
      if subtype not in entity_type_list:
        #print(mention, subtype)
        continue
        pred_entities.append(tuple(mention))


    #raw_input(" ")

    right_entities = list(set(gold_args_entities).intersection(set(pred_entities)))
    # print(doc_key)
    # print("right entity:", len(right_entities))
    # for entity in right_entities:
    #   print(entity, tokens[entity[0]:entity[1]+1])

    # print("gold entity:", len(gold_args_entities))
    # for entity in gold_args_entities:
    #   print(entity, tokens[entity[0]:entity[1]+1])

    # print("pred entity:", len(pred_entities))
    # for entity in pred_entities:
    #   print(entity, tokens[entity[0]:entity[1]+1])
    # raw_input(" ")

    gold_args_entities_full.extend(gold_args_entities)
    pred_entities_full.extend(pred_entities)
    right_entities_full.extend(right_entities)

  arg_p, arg_r, arg_f = prf(right_entities_full, gold_args_entities_full, pred_entities_full)
  arg_results = ["arg entity coverage:", arg_p, arg_r, arg_f, len(gold_args_entities_full), len(pred_entities_full), \
                 len(right_entities_full)]
  arg_result_str = "\t".join([str(a) for a in arg_results])
  print(arg_result_str)

def evaluate_event_type(example_map, subtype_field):
  total_gold_types = []
  total_sys_types = []
  total_correct_types = []
  type_set = set()
  for doc in example_map.values():
    for cluster in doc["gold_clusters"]:
      for mid, m in enumerate(sorted(cluster)):
        if "_" in m[2]:              
          type_set.add(m[2].split("_")[0])

  print(type_set)
  for doc in example_map.values():
    gold_types = set()
    for cluster in doc["gold_clusters"]:
      for mid, m in enumerate(sorted(cluster)):
        if "_" in m[2]:              
          gold_types.add((m[0], m[1], m[2].split("_")[0]))

    sys_types = []
    match_types = []
    for m, mtype in zip(doc["top_spans"], doc[subtype_field]):
      if mtype == "null":
        continue

      mtype = mtype.split("_")[0]
      if mtype not in type_set:
        continue

      pred = (m[0], m[1], mtype)
      sys_types.append(pred)
      if pred in gold_types:
        match_types.append(pred)


    total_gold_types.extend(list(gold_types))
    total_sys_types.extend(list(sys_types))
    total_correct_types.extend(list(match_types))

  typep, typer, typef = prf(total_correct_types, total_gold_types, total_sys_types)
  return typep, typer, typef
