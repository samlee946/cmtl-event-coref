from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import os
import sys
import json
import tempfile
import subprocess
import collections

import util
from bert import tokenization

class DocumentState(object):
  def __init__(self, key):
    self.doc_key = key
    self.sentence_end = []
    self.token_end = []
    self.tokens = []
    self.subtokens = []
    self.info = []
    self.segments = []
    self.subtoken_map = []
    self.segment_subtoken_map = []
    self.sentence_map = []
    self.pronouns = []
    self.coref_stacks = collections.defaultdict(list)
    self.segment_info = []
    self.sentences = []

    self.clusters = [] #collections.defaultdict(list)
    self.clusters_tokens = []
    
    self.gold_event_mentions = []
    self.gold_event_mentions_tokens = []

    self.gold_arguments = []
    self.gold_arguments_tokens = []

  def finalize(self):
    # finalized: segments, segment_subtoken_map
      
    sentence_map =  get_sentence_map(self.segments, self.sentence_end)
    subtoken_map = util.flatten(self.segment_subtoken_map)

    # populate clusters
    subtoken_token_map = {}
    for subtid, tid in enumerate(subtoken_map):
      if tid not in subtoken_token_map:
        subtoken_token_map[tid] = []
      subtoken_token_map[tid].append(subtid)

    flatten_sentences = util.flatten(self.segments)

    #print(self.doc_key)
    for cluster in self.clusters_tokens:
      subtoken_cluster = []
      for mention in cluster:
        mstart, mend, mtype = mention
        sstart = subtoken_token_map[mstart][0]
        send = subtoken_token_map[mend][-1]

        #print(mstart, mend, sstart, send, subtoken_map[sstart:send+1], flatten_sentences[sstart:send+1])
        subtoken_cluster.append((sstart, send, mtype))
      
      self.clusters.append(subtoken_cluster)
      #input(" ")

    for emention in self.gold_event_mentions_tokens:
      print(emention)
      mstart, mend, mtype, mrealis = emention
      sstart = subtoken_token_map[mstart][0]
      send = subtoken_token_map[mend][-1]

      self.gold_event_mentions.append((sstart, send, mtype, mrealis))

    for arg in self.gold_arguments_tokens:
      mstart, mend, pstart, pend, hstart, hend, role, cid = arg

      smstart = subtoken_token_map[mstart][0]
      smend = subtoken_token_map[mend][-1]

      spstart = subtoken_token_map[pstart][0]
      spend = subtoken_token_map[pend][-1]

      shstart = subtoken_token_map[hstart][0]
      shend = subtoken_token_map[hend][-1]

      self.gold_arguments.append((smstart, smend, spstart, spend, shstart, shend, role, cid))

    #assert len(all_mentions) == len(set(all_mentions))
    num_words =  len(util.flatten(self.segments))
    assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
    assert num_words == len(sentence_map), (num_words, len(sentence_map))
    return {
      #"doc_id": self.doc_key,
      "raw_sentences": self.sentences,
      "doc_key": self.doc_key, 
      "sentences": self.segments,
      "gold_clusters": self.clusters,
      'sentence_map':sentence_map,
      "subtoken_map": subtoken_map,
      "gold_event_mentions": self.gold_event_mentions,
      "gold_arguments": self.gold_arguments, 
    }


def normalize_word(word, language):
  if language == "arabic":
    word = word[:word.find("#")]
  if word == "/." or word == "/?":
    return word[1:]
  else:
    return word

# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
  current = 0
  previous_token = 0
  while current < len(document_state.subtokens):
    end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
    while end >= current and not constraints1[end]:
      end -= 1
    if end < current:
        end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
        while end >= current and not constraints2[end]:
            end -= 1
        if end < current:
            raise Exception("Can't find valid segment")
    document_state.segments.append(['[CLS]'] + document_state.subtokens[current:end + 1] + ['[SEP]'])
    subtoken_map = document_state.subtoken_map[current : end + 1]
    document_state.segment_subtoken_map.append([previous_token] + subtoken_map + [subtoken_map[-1]])
    info = document_state.info[current : end + 1]
    document_state.segment_info.append([None] + info + [None])
    current = end + 1
    previous_token = subtoken_map[-1]

def get_sentence_map(segments, sentence_end):
  current = 0
  sent_map = []
  sent_end_idx = 0
  assert len(sentence_end) == sum([len(s) -2 for s in segments])
  for segment in segments:
    sent_map.append(current)
    for i in range(len(segment) - 2):
      sent_map.append(current)
      current += int(sentence_end[sent_end_idx])
      sent_end_idx += 1
    sent_map.append(current)
  return sent_map

def get_document(document_line, tokenizer, language, segment_len):
  document_state = DocumentState(document_line['doc_key'])

  doc_key = document_line['doc_key']

  word_idx = -1
  for sentence in document_line['sentences']:
    for word in sentence:
      word_idx += 1
      word = normalize_word(word, language)
      subtokens = tokenizer.tokenize(word)
      document_state.tokens.append(word)
      document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
      for sidx, subtoken in enumerate(subtokens):
        document_state.subtokens.append(subtoken)
        info = None if sidx != 0 else ([word, word_idx, '-'] + [len(subtokens)])
        document_state.info.append(info)
        document_state.sentence_end.append(False)
        document_state.subtoken_map.append(word_idx)
    document_state.sentence_end[-1] = True

  document_state.sentences = document_line['sentences']
  document_state.clusters_tokens = document_line['gold_clusters']
  document_state.gold_event_mentions_tokens = document_line["gold_event_mentions"]
  document_state.gold_arguments_tokens = document_line["gold_arguments"]

  # split_into_segments(document_state, segment_len, document_state.token_end)
  # split_into_segments(document_state, segment_len, document_state.sentence_end)
  constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
  split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
  stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]), stats["max_sent_len_{}".format(language)])
  document = document_state.finalize()
  return document

def skip(doc_key):
  # if doc_key in ['nw/xinhua/00/chtb_0078_0', 'wb/eng/00/eng_0004_1']: #, 'nw/xinhua/01/chtb_0194_0', 'nw/xinhua/01/chtb_0157_0']:
    # return True
  return False

def minimize_partition(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir):
  #input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
  input_path = "{}/{}{}".format(input_dir, name, extension)
  output_path = "{}/{}.{}.{}{}".format(output_dir, name, language, seg_len, extension)
  count = 0
  print("Minimizing {}".format(input_path))
  documents = []
  with open(input_path, "r") as input_file:
    for line in input_file.readlines():
      documents.append(json.loads(line))
    
  with open(output_path, "w") as output_file:
    for document_line in documents:
      document = get_document(document_line, tokenizer, language, seg_len)
      output_file.write(json.dumps(document))
      output_file.write("\n")
      count += 1
  print("Wrote {} documents to {}".format(count, output_path))

def minimize_language(language, labels, extension, stats, vocab_file, seg_len, input_dir, output_dir, do_lower_case):
  # do_lower_case = True if 'chinese' in vocab_file else False
  tokenizer = tokenization.FullTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)
  #minimize_partition("Richere_dev_E2968739464_triggermatched_rmquote_138", language, "json", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition("english.E2968739464.train.jsonlines.jointencoref-c2f-predictions.json.e7394en.realis", language, "", labels, stats, tokenizer, seg_len, input_dir, output_dir)
  minimize_partition("english.E51.test.jsonlines.jointencoref", language, "", labels, stats, tokenizer, seg_len, input_dir, output_dir)

if __name__ == "__main__":
  vocab_file = sys.argv[1]
  input_dir = sys.argv[2]
  output_dir = sys.argv[3]
  do_lower_case = sys.argv[4].lower() == 'true'
  extension = ""
  print(do_lower_case)
  labels = collections.defaultdict(set)
  stats = collections.defaultdict(int)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  for seg_len in [128, 256, 384, 512]:
    minimize_language("english", labels, extension, stats, vocab_file, seg_len, input_dir, output_dir, do_lower_case)
    # minimize_language("chinese", labels, stats, vocab_file, seg_len)
    # minimize_language("es", labels, stats, vocab_file, seg_len)
    # minimize_language("arabic", labels, stats, vocab_file, seg_len)
  for k, v in labels.items():
    print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
  for k, v in stats.items():
    print("{} = {}".format(k, v))
