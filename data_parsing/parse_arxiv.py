"""
The following script adds title and abstract to to a processed huggingface dataset
March using ariv id (without version)
"""

import json
from datasets import load_dataset
from tqdm import tqdm

# Assumes you have downloaded arxiv from Kaggle Arxiv and unzip to the following location
with open('/home/ubuntu/SciCapPlus/arxiv_kaggle/arxiv-metadata-oai-snapshot.json', 'r') as f:
    arxive_str = f.read()
    arxive_list = arxive_str.split('\n')
    arxive = []
    for item in arxive_list:
        if item:
            arxive.append(json.loads(item))
arxive_dict = {}
for paper in arxive:
    arxive_dict[paper['id']] = paper


# def builds_results(id_list):
#     results = []
#     search = arxiv.Search(id_list=id_list)
#     for i in tqdm(range(len(id_list))):
#         paper = next(search.results())
#         result = dict(id = id_list[i], title = paper.title, summary = paper.summary)
#     return results

def find_corresponding_entry(id):
    if id not in arxive_dict:
        return '', ''
    result = arxive_dict[id]
    title = result.get('title', '').strip()
    abstract = result.get('abstract', '').strip()

    return title, abstract

def proc(example):
    id = example['id'].split('v')[0]
    title, abstract = find_corresponding_entry(id)
    example['title'] = title
    example['abstract'] = abstract

    return example

if __name__ = '__main__':
    data_with_ocr = load_dataset('alexshengzhili/SciCapInstructed410K')

    data_with_ocr_supplimented_fixed = data_with_ocr.map(proc, num_proc=80)
    data_with_ocr_supplimented_fixed.push_to_hub('alexshengzhili/SciCapAbstractsOCR0350K')
