---
dataset_info:
  features:
  - name: topic
    dtype: string
  - name: source
    dtype: string
  - name: bias
    dtype:
      class_label:
        names:
          '0': '0'
          '1': '1'
          '2': '2'
  - name: url
    dtype: string
  - name: title
    dtype: string
  - name: date
    dtype: string
  - name: authors
    dtype: string
  - name: content
    dtype: string
  - name: content_original
    dtype: string
  - name: source_url
    dtype: string
  - name: bias_text
    dtype:
      class_label:
        names:
          '0': left
          '1': right
          '2': center
  - name: ID
    dtype: string
  splits:
  - name: train
    num_bytes: 337630550
    num_examples: 26590
  - name: test
    num_bytes: 15232421
    num_examples: 1300
  - name: valid
    num_bytes: 22318031
    num_examples: 2356
  download_size: 222644131
  dataset_size: 375181002
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: valid
    path: data/valid-*
license: apache-2.0
task_categories:
- text-classification
language:
- en
pretty_name: News Articles with Political Bias Annotations (Media Source Split)
---

# News Articles with Political Bias Annotations (Media Source Split)

## Source

Derived from Baly et al.'s work:
[We Can Detect Your Bias: Predicting the Political Ideology of News Articles](https://aclanthology.org/2020.emnlp-main.404/) (Baly et al., EMNLP 2020)

## Information

This dataset contains **34,737 news articles** manually annotated for political ideology, either "left", "center", or "right".
This version contains **media source** test/training/validation splits, where the articles in each split are 
from different media sources than the articles in the others.  These are identical to the media splits used by Baly 
(according to their [git repository](https://github.com/ramybaly/Article-Bias-Prediction)).