---
dataset_info:
  features:
  - name: image_file
    dtype: string
  - name: id
    dtype: string
  - name: caption
    dtype: string
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: first_mention
    dtype: string
  - name: response
    dtype: string
  - name: title
    dtype: string
  - name: abstract
    dtype: string
  - name: q_a_pairs
    sequence:
      sequence: string
  splits:
  - name: 1_percent_as_validation
    num_bytes: 23679772
    num_examples: 3520
  download_size: 10644163
  dataset_size: 23679772
---
# Dataset Card for "SciCapInstructed-350K"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)