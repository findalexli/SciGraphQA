from datasets import load_dataset
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image

vali_dataset = load_dataset('alexshengzhili/SciCapInstructed-graph-only-qa', split='1_percent_as_validation')
data = vali_dataset.filter(lambda x: x['q_a_pairs'] is not None and len(x['q_a_pairs']) > 0)



processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

from tqdm import tqdm
def get_input(example):
    question = example['q_a_pairs'][0][0]

    prompt = f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    Human: {question}.
    AI: '''
    image_root_folder = '/home/ubuntu/imgs/train/'
    image_filepath = example['image_file']
    return prompt, image_root_folder + image_filepath

response_model = []
for example in tqdm(data):
    prompt, img_path = get_input(example)
    image_paper = Image.open(img_path)
    
    inputs = processor(image_paper, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
    response_model.append(generated_text)

new_data = data.add_column('response_BLIP2', response_model)
new_data.save_to_disk('1_percent_as_validation_blip')