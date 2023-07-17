import vertexai

from datasets import load_dataset
import pandas as pd
from vertexai.preview.language_models import InputOutputTextPair

from vertexai.preview.language_models import (ChatModel, InputOutputTextPair,
                                              TextEmbeddingModel,
                                              TextGenerationModel)

from graph_only_prompts_examples import system_message, examples
PROJECT_ID = "rwe-200-survey-data"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location="us-central1")

example_1_textpair = InputOutputTextPair(input_text=examples[0], output_text=examples[1])
example_2_textpair = InputOutputTextPair(input_text=examples[2], output_text=examples[3])

def return_palm(example):
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    chat = chat_model.start_chat(
        context=system_message,
        examples=[example_1_textpair, example_2_textpair],
        temperature=0.3,
        max_output_tokens=1020,
        top_p=0.8,
        top_k=40,
    )
    prompt = str(example['conversations'])[1:-2]
    example['response'] = chat.send_message(prompt).text
    
    
    return example


dataset = load_dataset("alexshengzhili/SciCapAbstractsOCR0350K", num_proc  = 4, split = 'train[40%:70%]')
dataset_non_empty_mention = dataset.filter(lambda item: len(item['first_mention']) > 10, num_proc  = 4)

lastthirty_to_last_ten = dataset_non_empty_mention.map(return_palm, num_proc=12)
lastthirty_to_last_ten.save_to_disk('with_abstract_graph_derived_question_last_30percent_tolast10_train')
