from datasets import load_dataset
from tqdm import tqdm
import json

dataset_name = 'alexshengzhili/SciCap-instruct-multiturn-filtered'
dataset = load_dataset(dataset_name, split='train')
json_filename = 'SciCap-instruct-multiturn-filtered-llava-13b-training.json'

def convert_dataset_into_json(data, json_filename):
    # Initialize a list to save processed data
    all_data = []

    # Iterate over the dataset
    for record in tqdm(data):
        # Apply some processing function to your data, if needed
        # processed_data = process_numpy(record)
        output_example = dict()
        output_example['image_file'] = record['image_file']
        output_example['conversations'] = record['conversations']
        output_example['id'] = record['id']
        # If no processing is needed, directly append the record

        # Append the data to the list
        all_data.append(output_example)

    # Save all the data to a line-separated JSON file
    with open(json_filename, "w") as output_file:
        json.dump(all_data, output_file, indent=4)
            
# convert_dataset_into_json(combined_list, 'data/eval.json')
convert_dataset_into_json(dataset, json_filename)