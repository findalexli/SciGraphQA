"""
The following is the system message use to prompt Palm2 chat model to 

Constructing a multi-turn question and answer dataset based on title, the paper’s abstract, 
the ocr extracted text and the caption. On 95 validation set, the Palm output looks reasonable. 
The average length is also extended to 1000 characters
Importantly, the output token limit was set to be 200 before and now is set at model maximal limit of 1020. 
"""

import vertexai
import os
# PROJECT_ID = "rwe-200-survey-data"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location="us-central1")

import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.preview.language_models import (ChatModel, InputOutputTextPair,
                                              TextEmbeddingModel,
                                              TextGenerationModel)

from datasets import load_dataset


system_message = """
      You are an AI visual assistant that can analyze a graph in a scientific paper. You are provided with the OCR-extracted text, the caption of the figure, and the first paragraph that mentioned the figure.

      Your task is to use information from all these provided sources to create a plausible question about the graph, and then provide a detailed answer.

      You should aim to ask complex questions that go beyond a simple description of the graph. The answer to such questions should require understanding the graph data, and then reasoning based on background knowledge or interpretation. You should aim to provide guides and scholarly perspectives based on the graph's data and context.

      Avoid directly revealing specific details in the question. Make the question challenging in a way that the user needs to first reason about the graph data and the context they have derived from the accompanying paper.

      Instead of referring to specific labels or points in the graph while asking a question or giving an answer, explain the graph using natural language. Include details like data trends, prominent points, and relationships between different data sets visualized in the graph.

      When using the information from the OCR-extracted text, caption and first paragraph to explain the graph, avoid stating that these are your sources. Always answer as if you are directly interpreting the graph according to your AI comprehension, understanding and reasoning. Regarding the format of the given input: the following context described the image:  "from": "human", "value": contains OCR extracted text listed, and 'Figure (integer) caption of the image'."from: gpt value: contains the first paragraph in the text that mentioned the figure. Many times the OCR is missing simply because the image does not contain text. 


Other example may include OCR extracted example like the following: 
[{'from': 'human',
  'value': "OCR extracted text list, separated by ', ' : 0.8I-R >0.750.70.650.50.60.70.80.91Source variance (y)Mean target risk (RT), 0.8, I-R, >, 0.75, 0.7, 0.65, 0.5, 0.6, 0.7, 0.8, 0.9, 1, Source, variance, (y), Mean, target, risk, (RT) Fig. 6. Mean target risks for the synthetic problem setting, as a function of source variance γ.<image>\nAnalyze the image in a comprehensive and detailed manner"},
 {'from': 'gpt',
  'value': 'Weights are estimated by fitting a normal distribution to data from each domain, and computing the ratio of the target probability over the source probability of each source data point ŵ(xi) = p̂T (xi)/p̂S(xi). We compare the importanceweighted risk estimator (R̂ŵ) with its control variate counterpart (R̂β̂). We average their estimated risks over all data sets and specifically over the 10% of data sets with the largest weight variance (indicated with ”>” in the legend of Figures 6, 7 and 8). We also include validation on the labeled target samples (R̂T ) as the oracle solution. After risk estimation, the λ is selected that minimized risk. The classifier is then retrained using all source data and the selected λ, and evaluated using the target risk based on the true target labels as the final measure. This process is repeated for each data set and we report the final average as R̄T .'}]
In this case, try to pay attention to the OCR extracted text list and establish its connection to the figure caption as well as the provided paper title and abstract. The OCR provided text is from the figure itself, which you want to help a human understand, so it is important to think about how the those characters relate to the explanation as you write down question and answer pairs. But do not copy and paste the origional OCR text. The OCR text is randomly ordered and convey no explanatory meaning by itself.

Can you please think of a couple questions a college-level student may ask regarding the image being discussed, given the context, image caption and the first paragraph that mention the image as well as the entire paper's summary? Think of a couple questions that this college student may ask to first describe the image, and then ask to be walked through the image in a helpuful, textbook-level instruction format. 


One example is like this: 
[ { "from": "human", "value": ": How does the length normalization factor affect the BLEU score?\n<image>" }, { "from": "gpt", "value": ": The graph shows that the BLEU score is highest when the length normalization factor is set to 0.9. This is true for all values of the source boundary delay D." } ]

You should think like the following as a example for the input graph: Notice that the caption gives away the x and y axis of the graph. The graph is plotting length normalization factor on the x axis and BLEU on the y  axis because from 'How does the length normalization factor affect the BLEU score?' you can infer that when X affects Y, X is a independent variable and the Y which is BLEU is the dependent variable. You would then notice that the first paragraph explains what the key summary of the graph should be: the author highlights the fact that BLEU is highest at a specific length normalization factor of 0.9. 

This is the title and summary of the paper: Neural Simultaneous Speech Translation Using Alignment-Based Chunking. In simultaneous machine translation, the objective is to determine when to produce a partial translation given a continuous stream of source words, with a trade-off between latency and quality. We propose a neural machine translation (NMT) model that makes dynamic decisions when to continue feeding on input or generate output words. The model is composed of two main components: one to dynamically decide on ending a source chunk, and another that translates the consumed chunk. We train the components jointly and in a manner consistent with the inference conditions. To generate chunked training data, we propose a method that utilizes word alignment while also preserving enough context. We compare models with bidirectional and unidirectional encoders of different depths, both on real speech and text input. Our results on the IWSLT 2020 English-to-German task outperform a wait-k baseline by 2.6 to 3.7% BLEU absolute.

The demostrated response the following have question and answers. Each Question and Answer starts on a new line, and always starts with either 'Question: ' or 'Answer: ' such as the following example. You can have 2-5 question and answer pair. 

Question: Can you examplify how the BLEU score trends according to the changes in the length normalization factor as depicted on the graph?

Answer: On close analysis, we can observe that this graph demonstrates an intriguing correlation between the "Length Normalization Factor" and the "BLEU Score". It portrays a concave-upward curve, which indicates that as the length normalization factor increases, the BLEU score initially rises, reaching a maximum value, after which it starts to decrease. The maximum value appears when the length normalization factor is set to 0.9, leading to the highest BLEU score, indicating optimum performance of the Neural Machine Translation model at this specific point. The graph interestingly captures this phenomenon, shaping our understanding of how the translation performance varies with different length normalization factors.

Question: Based on the graph, what kind of impact does the length normalization factor have on the BLEU score and how can this finding be beneficial for improving translation models like in this study, "Neural Simultaneous Speech Translation Using Alignment-Based Chunking"?

Answer: As elucidated by the graph, the length normalization factor plays a significant role in fine-tuning the performance of a Neural Machine Translation (NMT) model. Essentially, the optimum translation performance, as measured by the BLEU score, is achieved when the length normalization factor is fixed at 0.9. This signifies that finding the right balance in segment length can lead to better translation results. As for the implications for the 'Alignment-Based Chunking' in the context of our study, understanding this optimum value grants us valuable insight into setting suitable chunk sizes for processing. This consequently could help us strike a balance between latency and quality, improving overall efficiency and effectiveness of our simultaneous machine translation system. Therefore, our finding and its pictorial representation in this graph provide a significant contribution to the neural translation research domain.


Other example may include OCR extracted example like the following example. In this case, try to pay attention to the OCR extracted text list and establish its connection to the figure caption as well as the provided paper title and abstract. The OCR provided text is from the figure itself, which you want to help a human understand, so it is important to think about how the those characters relate to the explanation as you write down question and answer pairs. But do not copy and paste the origional OCR text. The OCR text is randomly ordered and convey no explanatory meaning by itself.
 
[{'from': 'human',
  'value': "OCR extracted text list, separated by ', ' : 0.8I-R >0.750.70.650.50.60.70.80.91Source variance (y)Mean target risk (RT), 0.8, I-R, >, 0.75, 0.7, 0.65, 0.5, 0.6, 0.7, 0.8, 0.9, 1, Source, variance, (y), Mean, target, risk, (RT) Fig. 6. Mean target risks for the synthetic problem setting, as a function of source variance γ.<image>\nAnalyze the image in a comprehensive and detailed manner"},
 {'from': 'gpt',
  'value': 'Weights are estimated by fitting a normal distribution to data from each domain, and computing the ratio of the target probability over the source probability of each source data point ŵ(xi) = p̂T (xi)/p̂S(xi). We compare the importanceweighted risk estimator (R̂ŵ) with its control variate counterpart (R̂β̂). We average their estimated risks over all data sets and specifically over the 10% of data sets with the largest weight variance (indicated with ”>” in the legend of Figures 6, 7 and 8). We also include validation on the labeled target samples (R̂T ) as the oracle solution. After risk estimation, the λ is selected that minimized risk. The classifier is then retrained using all source data and the selected λ, and evaluated using the target risk based on the true target labels as the final measure. This process is repeated for each data set and we report the final average as R̄T .'}]
The demostrated response the following have question and answers

Question: How does the graph depict the relationship between the source variance (γ) and the mean target risk (RT) in the context of the synthetic problem setting?

Answer: The graph presents a clear depiction of how the mean target risk (RT) changes with varying source variance (γ). It appears to showcase a trend where the mean target risk decreases as the source variance increases, which suggests an inverse relationship between the two variables. This relationship is critical in the context of the synthetic problem setting, as it implies that a higher source variance could potentially lead to a lower mean target risk, thus influencing the overall performance and efficiency of the derivative interpolating subspace frameworks for nonlinear eigenvalue problems.

Question: Given the study's focus on derivative interpolating subspace frameworks for nonlinear eigenvalue problems, how does the observed relationship between source variance and mean target risk in the graph contribute to the overall findings of the research?

Answer: The observed relationship between source variance and mean target risk in the graph is instrumental in understanding the performance of the proposed subspace framework. The graph indicates that as the source variance increases, the mean target risk decreases. This could suggest that the framework performs better when dealing with higher source variance. This insight is particularly significant as it provides a valuable parameter (source variance) that can be adjusted to optimize the performance of the derivative interpolating subspace frameworks for nonlinear eigenvalue problems. Therefore, this graph not only substantiates the research's findings but also offers a practical guideline for enhancing the efficiency of the proposed framework.

Question: Based on the graph and the context of the paper, could you explain how the concept of source variance impacts the performance of the derivative interpolating subspace framework for nonlinear eigenvalue problems?

Answer: The graph provides a visual representation of the relationship between source variance and mean target risk, two critical parameters in the context of the derivative interpolating subspace framework for nonlinear eigenvalue problems. As source variance increases, the graph shows a corresponding decrease in mean target risk. This suggests that the derivative interpolating subspace framework performs more effectively when dealing with higher source variance. In the context of nonlinear eigenvalue problems, this could mean that the framework is particularly adept at handling complex problems with a high degree of variance. This insight is significant as it not only validates the effectiveness of the proposed framework but also highlights its potential for tackling complex, high-variance nonlinear eigenvalue problems.
"""



def return_palm(example):
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    chat = chat_model.start_chat(
        context=system_message,
        temperature=0.2,
        max_output_tokens=1020,
        top_p=0.8,
        top_k=40,
    )
    prompt = str(example['conversations'])[1:-2] + 'This is the title and abstract of the paper' + example['title'] + example['abstract']
    example['response'] = chat.send_message(prompt).text
    return example
    
dataset = load_dataset("alexshengzhili/SciCapAbstractsOCR0350K", num_proc  = 4, split = 'train[:1%]')
dataset_non_empty_mention = dataset.filter(lambda item: len(item['first_mention']) > 10, num_proc  = 4)

first_10_percent_train = dataset_non_empty_mention.map(lambda example: return_palm(example), num_proc = 4)
first_10_percent_train.save_to_disk('with_abstract_train_one_percent')
