from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=4
)

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch
from datasets import load_dataset
import torch
from PIL import Image

vali_dataset = load_dataset('alexshengzhili/SciCapInstructed-graph-only-qa', split='1_percent_as_validation')
data = vali_dataset.filter(lambda x: x['q_a_pairs'] is not None and len(x['q_a_pairs']) > 0)

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device, torch.float16)

from PIL import Image
import requests
from tqdm import tqdm
import random
import json

def get_input_example_for_contextual_lerning(context_data, num_examples):
    # Pick num_examples random examples after 100
    #example_index = random.randint(0, len(context_data), num_examples)
    example_indexes = random.sample(range(len(context_data)), num_examples)
    questions = []
    answers = []
    img_paths = []
    image_root_folder = '/home/ubuntu/imgs/train/'
    for example_idx in example_indexes:
        example = context_data[example_idx]
        question = example['q_a_pairs'][0][0]
        answer = example['q_a_pairs'][0][1]
        img_path = image_root_folder + example['image_file']
        questions.append(question)
        answers.append(answer)
        img_paths.append(img_path)
    return questions, answers, img_paths


def get_input(example):
    question = example['q_a_pairs'][0][0]
    image_root_folder = '/home/ubuntu/imgs/train/'
    image_filepath = example['image_file']
    return question, image_root_folder + image_filepath

tokenizer.padding_side = "left" # For generation padding tokens should be on the left

def generate_text(example, num_examples):
    """
    Step 0: pick num_examples random examples
    ""
    Step 1: Load images
    """
    questions, answers, img_paths = get_input_example_for_contextual_lerning(data, num_examples)
    demo_examples = [f"question: {q} answer: {a}" for q, a in zip(questions, answers)]
    demo_images = [Image.open(img_path) for img_path in img_paths]
    # Step 1: Load query image
    question, img_path = get_input(example)
    query_image = Image.open(img_path)
    # query = json.dumps({"question:": question, "answer:": ''})
    query = f"question: {question} answer: "
    """
    Step 2: Preprocess images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
    batch_size x num_media x num_frames x channels x height x width. 
    In this case batch_size = num_examples + 1, num_media = 1, num_frames = 1,
    channels = 3, height = 224, width = 224.
    """
    if num_examples > 0:
        vision_x = [image_processor(img).unsqueeze(0) for img in demo_images]
        vision_x.append(image_processor(query_image).unsqueeze(0))
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device, torch.float16)
    else:
        vision_x = image_processor(query_image).unsqueeze(0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device, torch.float16)


    """
    Step 3: Preprocess question
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """

    if num_examples == 0:
        lang_x = tokenizer(
            [f"<image>{query}"],
            return_tensors="pt",
        )
    else:
        lang_x = tokenizer(
            [f"<image>{'<|endofchunk|>'.join(demo_examples)}<|endofchunk|><image>{query}"],
            return_tensors="pt",
        )
    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"].to(device),
        attention_mask=lang_x["attention_mask"].to(device),
        max_new_tokens=100,
        num_beams=1,
    )

    output = tokenizer.decode(generated_text[0])
    print("Generated text: ", output)
    return output

# generate_text(first_100[3], 10)

with open("open_flaming_2shot.jsonl", "a") as f:
    for i in tqdm(range(len(data))):
        response = generate_text(data[i], 2)
        data[i]["of_response"] = response
        json.dump(data[i], f)
        f.write("\n")

response_data = data.add_column("of_response", response)
response_data.save_to_disk("open_flamingo_2shot.jsonl")