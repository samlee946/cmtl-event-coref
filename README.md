# Constrained Multi-Task Learning for Event Coreference Resolution

This is the GitHub repository for Jing Lu's code used in her NAACL'21 paper Constrained Multi-Task Learning for Event Coreference Resolution. I do not own anything in this repository. If you use this code, please consider citing her paper:
```
@inproceedings{lu-ng-2021-constrained,
    title = "Constrained Multi-Task Learning for Event Coreference Resolution",
    author = "Lu, Jing  and
      Ng, Vincent",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.356",
    doi = "10.18653/v1/2021.naacl-main.356",
    pages = "4504--4514",
    abstract = "We propose a neural event coreference model in which event coreference is jointly trained with five tasks: trigger detection, entity coreference, anaphoricity determination, realis detection, and argument extraction. To guide the learning of this complex model, we incorporate cross-task consistency constraints into the learning process as soft constraints via designing penalty functions. In addition, we propose the novel idea of viewing entity coreference and event coreference as a single coreference task, which we believe is a step towards a unified model of coreference resolution. The resulting model achieves state-of-the-art results on the KBP 2017 event coreference dataset.",
}
```

# Usage
## Setup
1. Install requirements. We used TF 1.15 provided by NVIDIA. PyTorch is needed for loading PyTorch checkpoints. 
```
pip install -r requirements.txt
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

2. Compile the coref kernel
```
bash setup_all.sh
```
You might encounter the following error, depending on your g++ version. 
```
tensorflow.python.framework.errors_impl.NotFoundError: ./coref_kernels.so: undefined symbol: _ZN10tensorflow12OpDefBuilder4AttrESs
```
You might try setting `-D_GLIBCXX_USE_CXX11_ABI=1`.

3. Data setup
   ALl data paths need to be set in experiments.conf
   * training set:
     * train_path
   * test set:
     * test_path: The path to the jsonlines file of the test set.
     * gold_path: The path to the tbf file that contains gold event mentions and gold event coreference clusters.
     * entity_gold_path: The path to the tbf file that contains gold entity mentions and gold entity coreference clusters.
     * offset_name: The path to the offset file for the test set.
   * dev set:
     * eval_path: similar to the test set
     * gold_dev_path: similar to the test set
     * entity_gold_dev_path: similar to the test set
     * offset_name_dev: similar to the test set
   
   File structure:
   * train_path/test_path/eval_path
    Each line contains a json document:
     * doc_key: a string containing the key of the document
     * gold_clusters: a list of clusters, where each cluster is a list of event mentions, where each event mentions is a triplet of (event_start, event_end, event_type)
     * gold_event_mentions: a list of 4-tuples, where each 4-tuple is (trigger_st, trigger_ed, event_type, realis_status)
     * gold_arguments: a list of 8-tuples, where each 8-tuple is (trigger_st, trigger_ed, arg_st, arg_ed, arg_head_st, arg_head_ed, arg_type, idk)
     * subtoken_map: a mapping that maps subtoken indices to token indices
     * sentence_map: a mapping that maps subtoken indices to sentence indicies
     * sentences: a list of tokenized sentences 
     * raw_sentences: a list of raw sentences from the document
   * offset file:
    Each line contains a json document:
     * doc_key: a string containing the key of the document
     * offsets: a list of tuples of (st, ed), where the i-th tuple means: the position of the i-th token in raw_sentences is (st, ed) in the original document from. This is used to generate output for the scorer.
   * Other files are in the original TBF format.

## Training & Evaluation
* Train and evaluate Joint Model
```
export data_dir=data/       # this is where the model checkpoints are going to be saved
python train.py spanbert_large_joint2_naacl21 0 experiments.conf # config_name GPU_id config_file
```

* Inference only 
```
python3.8 evaluate.py spanbert_large_joint2_naacl21 0 experiments.conf
```