from multiprocess import set_start_method
import torch
import os
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image
from datasets import load_dataset, load_from_disk
from typing import Dict, List
set_start_method("spawn")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(device)
# local_rank = int(os.environ.get("LOCAL_RANK"))
# model = torch.nn.DataParallel(model, device_ids=[local_rank], output_device=local_rank)
processor = Pix2StructProcessor.from_pretrained('google/deplot')

dataset = load_dataset('alexshengzhili/SciCap-instruct-multiturn-filtered', split='train[6%:8%]')
data = dataset.filter(lambda x: x['q_a_pairs'] is not None and len(x['q_a_pairs']) > 0)

image_path = '/home/ubuntu/imgs/train/'

def process_example(example):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())
    path = image_path + example['image_file']
    image = Image.open(path)
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt").to(device)
    predictions = model.generate(**inputs, max_new_tokens=512)
    output_string = processor.decode(predictions[0], skip_special_tokens=True)
    example['deplot'] = output_string

    return example

updated_dataset = data.map(process_example)
updated_dataset.save_to_disk('train_6-8%_deplot')