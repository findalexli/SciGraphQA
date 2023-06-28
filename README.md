# 🌋 LLaVA-Graph: Context-prompted vision-language assistant for explaining scientific graphs
*TLDR: We re-formulate the image-caption-generation problem as a context-prompted 'explanation' generation problem, which aligns the model's output given (graph, caption) to its first mentioned paragraph. Trained with SciCap-390K, 5X more tokens using SciCap compared to original LLaVa vision-text alignment pretraining corpse *

[[Demo](https://llava.alex.li/)]  [[Data]([https://github.com/findalexli/LLaVA-Graph](https://huggingface.co/datasets/alexshengzhili/LLAVA-graph-OCRCleaned))] [[Model](https://huggingface.co/alexshengzhili/Llava-Graph-ocr-ft-on-instruct150k)]

## Main Contribution of LLaVA-Graph from LLaVa
- **Caption as context, not as prediction target**. 
    One-third of captions in SciCap are single-sentence (Hsu et al.2023) Many are incomplete sentences(i.e.  "Vary number of objects') or consist of single/multiple Nouns ('IIPS', 'Average Travel Time'). Efforts to predict captions from images have not been successful, despite attempts to contextualize the model (Yang, 2023). We theorized that captions are instead excellent prompts to the Large Vision-Language Models. 
- **OCR as additional context (optional)**. 
    ["On the Hidden Mystery of OCR in Large Multimodal Models"](https://arxiv.org/abs/2305.07895) shows that even the most powerful large multimodal models cannot match expert OCR models by a large margin. Zero-shot text recognition results range  from 37-61% compared to Supervised SOTA of 85%. This deficiency is particularly detrimental to science graphs where text and their location are much more important for humans to read and understand. We added extracted text and bounding boxes when text is detected in figures as part of the contextual prompt. 
- **Paragraph as alignment given above context**. 
    First-paragraphs that mentioned the figure are significantly more informative and carry descriptive and logical explanations, which we hypothesized to aid as ground truth in instructing the LMM to assist users in understanding graphs via Chat. LLaVA-receipt has two steps: 1. (image, caption) feature alignment which is to train a vision token projector into the language token space, pretraining on filtered 590K images with short captions . 2 (image, ground truth bounding box, GPT-4 rewritten multi-turn conversation) for visual instruction tuning. We **switched the first step of feature alignment using the proposed stronger signal between (image, short academic caption, ocr extracted token) and (first mentioned paragraph)**. Our training dataset is constructed using SciCap from 290K papers published on Arxiv on topics of CS and ML. Our training corpse [[Data](https://github.com/findalexli/LLaVA-Graph)] is 5X larger than LLaVa pre-training dataset (LiON-CC-590K) 



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
The pretraining dataset used in this release is from the SciCap dataset. SCICAP is a large-scale image captioning dataset that contains real-world scientific figures and captions. SCICAP was constructed using more than two million images from over 290,000 papers collected and released by arXiv.Homepage. The papers are filtered as 2010-2020 compuer science and machine learning papers. Hsu, Giles, and Huang (2021) show that text normalizationand figure filtering do not improve model performance.
In all three splits, around 90% of the captions are less than 66 words.Out of 2.1 million figures, Only graph figures are considered. Ohther formats such as tables, euqaitions, flowcharts, scatter plots and bar charts are not considered.  Please see here for a visualidation dataset [vis](https://huggingface.co/datasets/alexshengzhili/llava-graph-caption2mentioned-vis)


The Scicap dataset is transformed into the following single turn conversation: Xc Xq Xv<STOP>nn Assistant : Xm<STOP>nn, 
where Xc are context tokens (from caption and ocr-extract text), Xq is randomly selected question prompt are , Xv is image tokens, and Xm is mentioned paragraph. 

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
```Shell

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("alexshengzhili/LLaVa-graph-caption-to-paragraph")
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
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path alexshengzhili/LLaVa-graph-caption-to-paragraph --multi-modal
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".

#### Launch a model worker (Multiple GPUs, when GPU VRAM <= 24GB)

If your the VRAM of your GPU is less than 24GB (e.g., RTX 3090, RTX 4090, etc.), you may try running it with multiple GPUs.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path alexshengzhili/LLaVa-graph-caption-to-paragraph --multi-modal --num-gpus 2
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
[TODO]




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
