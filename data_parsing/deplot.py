from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image
from io import BytesIO
import torch
from PIL import Image
import pdb
from datasets import load_dataset, load_from_disk
from typing import Dict, List
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(device)
model = torch.compile(model, mode="reduce-overhead")
model.eval()
processor = Pix2StructProcessor.from_pretrained('google/deplot')

vali_dataset = load_dataset('alexshengzhili/SciCapInstructed-graph-only-qa', split='1_percent_as_validation')
data = vali_dataset.filter(lambda x: x['q_a_pairs'] is not None and len(x['q_a_pairs']) > 0)


from PIL import Image
image_path = '/home/ubuntu/imgs/train/'

def process_example(example):
    path = image_path + example['image_file']
    image = Image.open(path)
    inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt").to(device)
    predictions = model.generate(**inputs, max_new_tokens=512)
    output_string = processor.decode(predictions[0], skip_special_tokens=True)
    example['deplot'] = output_string

    return example

def process_examples(examples):
    file_images = examples['image_file']
    paths = [image_path + file_image for file_image in file_images]
    images = [Image.open(path) for path in paths]
    print(len(images))
    inputs = processor(images=images, text="Generate underlying data table of the figure below:", return_tensors="pt").to(device)
    predictions = model.generate(**inputs, max_new_tokens=512)
    output_strings = processor.batch_decode(predictions, skip_special_tokens=True)
    examples['deplot'] = output_strings
    return examples

validation_1k = data.map(process_example)
validation_1k.save_to_disk('validation_1k_deplot')