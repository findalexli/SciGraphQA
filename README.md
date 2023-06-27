# 🌋 LLaVA-Graph: Context-prompted vision-language assistant for explaining scientific graphs
*TLDR: We re-formulate image-caption-generation problem as context-prompted 'explanation' generation problem, which alings models output given (graph, caption) to its first mentioned paragraph. Fine-tuned from LLaVa with 5X more tokens using SciCap *

[[Demo](https://llava.hliu.cc/)]  [[Data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)] [[Model](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0)]

## Main Contribution of LLaVA-Graph
- **Caption as context, not as prediction target**. 
    One-third of captions in SciCap are single-sentence (Hsu et al.2023) Many are incomplete sentences(i.e.  "Vary number of objects') or consist of single/multple Nouns ('IIPS', 'Average Travel Time'). Efforts to predict captions from image have not been successful, despite attempts to contexualize the model (Yang, 2023). We theroized that captions are instead excellent prompts as contextting the Large Vision-Language Models. 
- **OCR as additional context (optional)**. 
    ["On the Hidden Mystery of OCR in Large Multimodal Models"](https://arxiv.org/abs/2305.07895) shows that even most powerful large multimodal models cannot match expert OCR models by large margin. Zero-shot text recognition results ranges  37-61% compared to Supervised SOTA of 85%. This deficincy is particular detrimental to science graphs where text and their location are much more important for humans to read and understand. We added extracted text and bounding boxes when text is detected in figures as part of the contextual prompt. 
- **Paragraph as target**. 
    first-paragraph that mentioned the figure are significantly more informative and carry descriptive and logical explanations, which we hypothezied to aid as ground truth in instructing the LMM to assist users to understand graphs via Chat.
- **Dataset**
    Our training dataset is constructed using SciCap from 290K papers published on Arxiv on topic of CS and ML. Our training corpose is 5X larger than LLaVapre-training dataset (LiON-CC-590K) or 50X larger with OCR features.  


**Usage and License Notices**: The data, code and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.


## Contents
- [Data Download](#data-download)
- [Install](#install)
- [LLaVA Weights](#llava-weights)
- [Serving](#serving)
- [Evaluation](#evaluation)
- [Fine-tuning](#fine-tuning)

## Data

| Data file name | Size |
| --- | ---: |
| [llava_instruct_150k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/llava_instruct_150k.json) | 229 MB |


### Pretraining Dataset
The pretraining dataset used in this release is a subset of CC-3M dataset, filtered with a more balanced concept coverage distribution.  Please see [here](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) for a detailed description on the dataset structure and how to download the images.

If you already have CC-3M dataset on your disk, the image names follow this format: `GCC_train_000000000.jpg`.  You may edit the `image` field correspondingly if necessary.

| Data | Chat File | Meta Data | Size |
| --- |  --- |  --- | ---: |
| CC-3M Concept-balanced 595K | [chat.json](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/raw/main/chat.json) | [metadata.json](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/raw/main/metadata.json) | 211 MB
| LAION/CC/SBU BLIP-Caption Concept-balanced 558K | [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/raw/main/blip_laion_cc_sbu_558k.json) | [metadata.json](#) | 181 MB

**Important notice**: Upon the request from the community, as ~15% images of the original CC-3M dataset are no longer accessible, we upload [`images.zip`](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip) for better reproducing our work in research community. It must not be used for any other purposes. The use of these images must comply with the CC-3M license. This may be taken down at any time when requested by the original CC-3M dataset owner or owners of the referenced images.



## Install

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/findalexli/LLaVA-Graph
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install ninja
pip install flash-attn==1.0.2
```



## LLaVA Weights
We release [LLaVA](https://llava-vl.github.io/) weights as delta weights to comply with the LLaMA model license.
You can add our delta to the original LLaMA weights to obtain the LLaVA weights.

Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get LLaVA weights by applying our delta ([13b-v0](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0), [7b-v0](https://huggingface.co/liuhaotian/LLaVA-7b-delta-v0), [lightning-7B-v1-1](https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1)). It will automatically download delta weights from our Hugging Face account.

### LLaVA-13B
This conversion command needs around 60 GB of CPU RAM.
```bash
python3 -m llava.model.apply_delta \
    --base /path/to/llama-13b \
    --target /output/path/to/LLaVA-13B-v0 \
    --delta liuhaotian/LLaVA-13b-delta-v0
```

### LLaVA-7B
This conversion command needs around 30 GB of CPU RAM.
```bash
python3 -m llava.model.apply_delta \
    --base /path/to/llama-7b \
    --target /output/path/to/LLaVA-7B-v0 \
    --delta liuhaotian/LLaVA-7b-delta-v0
```


### LLaVA pretrained projector weights
The initial release is pretrained on [LLaVA-filtered CC3M 595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) with 1 epoch.  The pretrained weights are released [here](https://huggingface.co/liuhaotian/LLaVA-13b-pretrain-projector-v0).

You may perform instruction tuning on our pretrained checkpoints, by using our [visual instruction tuning](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) data following the instructions [here](https://github.com/haotian-liu/LLaVA#fine-tuning-with-local-gpus).

## Serving

### Web UI

#### Launch a controller
```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a model worker
```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/LLaVA-13B-v0 --multi-modal
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

#### Launch a model worker (Multiple GPUs, when GPU VRAM <= 24GB)

If your the VRAM of your GPU is less than 24GB (e.g., RTX 3090, RTX 4090, etc.), you may try running it with multiple GPUs.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/LLaVA-13B-v0 --multi-modal --num-gpus 2
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

#### Launch a gradio web server.
```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000
```
#### You can open your browser and chat with a model now.

### CLI Inference

A starting script for inference with LLaVA without the need of Gradio interface. The current implementation only supports for a single-turn Q-A session, and the interactive CLI is WIP.  This also serves as an example for users to build customized inference scripts.

```Shell
python -m llava.eval.run_llava \
    --model-name /path/to/LLaVA-13B-v0 \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --query "What are the things I should be cautious about when I visit here?"
```

Example output (varies in different runs):

> When visiting this picturesque location with a serene lake and a wooden pier extending over the water, one should be cautious about various safety aspects. Some important considerations include:
> 
> 1. Ensuring that the pier is structurally sound andstable, as old or weakened pier structures might not support the weight of visitors.
> 2. Being aware of the water depth around the pier and lake, as sudden drop-offs or strong currents may pose a risk to swimmers, boaters, or those who venture too close to the edge.
> 3. Staying vigilant about the presence of wildlife in the area, such as slippery, stealthy fish or other animals that might cause harm or inconvenience.
> 4. Maintaining a safe distance from the water's edge, particularly for children, elderly individuals, or those who are not strong swimmers.
> 5. Following any posted signs or guidelines related to safety and the use of the pier and surrounding areas.
> 
> By considering these safety precautions, visitors can enjoy the natural beauty of the location while minimizing risks and ensuring a safe and pleasant experience.


## Evaluation

### GPT-assisted Evaluation

Our GPT-assisted evaluation pipeline for multimodal modeling is provided for a comprehensive understanding of the capabilities of vision-language models.  Please see our paper for more details.

1. Generate LLaVA responses

```Shell
python model_vqa.py \
    --model-name ./checkpoints/LLaVA-13B-v0 \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /path/to/coco2014_val \
    --answers-file \
    /path/to/answer-file.jsonl
```

2. Evaluate the generated responses.  In our case, [`answer-file-1.jsonl`](./playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl) is the response generated by text-only GPT-4 (0314), with the context captions/boxes provided.

```Shell
OPENAI_API_KEY="sk-***********************************" python eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    /path/to/answer-file-1.jsonl \
    /path/to/answer-file-2.jsonl \
    --rule table/rule.json \
    --output /path/to/review.json
```

3. Summarize the evaluation results

```Shell
python summarize_gpt_review.py
```




### Code and Hyperparameters
We fine-tune the model using the code from [FastChat](https://github.com/lm-sys/FastChat). We use a similar set of hyperparameters as Vicuna in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-13B | 128 | 2e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-13B | 32 | 2e-5 | 3 | 2048 | 0 |

### Fine-tuning with Local GPUs
LLaVA is trained on 8 A100 GPUs with 80GB memory with the following code. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly to keep the global batch size the same.

1. Pretraining

<details>
<summary>Pretrain: LLaVA-13B, 8x A100 (80G).  Time: ~4 hours.</summary>

```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/llama-vicuna-13b \
    --data_path /path/to/cc3m_595k.json \
    --image_folder /path/to/cc3m_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```
</details>

You may run this with a single A100 GPU with the following code.  Please note that the `per_device_train_batch_size` * `gradient_accumulation_steps` should be equal to 128 to keep the global batch size the same.

<details>
<summary>Pretrain: LLaVA-13B, 1x A100 (80G).  Time: ~33 hours.</summary>

```Shell
python llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/llama-vicuna-13b \
    --data_path /path/to/cc3m_595k.json \
    --image_folder /path/to/cc3m_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```
</details>

<details>
<summary>Pretrain: LLaVA-7B, 1x A100 (80G/40G).  Time: ~19 hours.</summary>

```Shell
python llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/llama-vicuna-7b \
    --data_path /path/to/cc3m_595k.json \
    --image_folder /path/to/cc3m_595k \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```
</details>


### Train LLaVA Lightning
LLaVA-Lightning can be trained on 8x A100 GPUs in just 3 hours, including both pretraining and finetuning. When using spot instances, it costs just ~$40.

Please make sure to: (1) [install](#install) or [upgrade](#upgrade-to-latest-code-base) to the latest code base, and (2) pass the correct model version identifier `v0`/`v1` to ensure the correct conversation template is loaded.

```Shell
bash ./scripts/train_lightning.sh {v0,v1}
```

#### Hyperparameters

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-Lightning-7B | 128 | 2e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-Lightning-7B | 128 | 2e-5 | 1 | 2048 | 0 |




## Acknowledgement
- [LLaVA] which the codebase we built on
- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!

## Related Projects

- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)

For future project ideas, pleae check out:
- [SEEM: Segment Everything Everywhere All at Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to detect, segment, and generate anything by marrying [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).
