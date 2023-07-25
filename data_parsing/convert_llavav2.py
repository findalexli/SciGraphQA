"""
Script used to convert the SciCapQA dataset to the format used by the LLAMA model with adding deplot information
"""

from datasets import load_dataset
import json
deplot_test = load_dataset('alexshengzhili/SciCapQA-test-with-deplot', split='1_percent_as_validation')
data = deplot_test.filter(lambda example: len(example['q_a_pairs']) > 0)


def get_input(example):
    question = example['q_a_pairs'][0][0]
    deplot = example['deplot']
    prompt = 'The underlying data table of the figure below is:' + deplot + question
    # answer = example['q_a_pairs'][0][1]
    categroy = "conv"
    image = example['image_file']
    return dict(text = question, category=categroy, image=image)

output_list = []

for i, example in enumerate(data):
    example_dict = get_input(example)
    example_dict['question_id'] = i
    output_list.append(example_dict)

import json
with open('vjuly23_without_deplot_llava_eval.jsonl', 'w') as f:
    for item in output_list:
        f.write(json.dumps(item) + '\n')