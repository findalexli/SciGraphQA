from datasets import load_dataset
import json
data = load_dataset('alexshengzhili/SciCapInstructed-graph-only-qa', split = '1_percent_as_validation')


def get_input(example):
    question = example['q_a_pairs'][0][0]
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
with open('vali.jsonl', 'w') as f:
    for item in output_list:
        f.write(json.dumps(item) + '\n')
