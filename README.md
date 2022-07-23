# Word Recommendation System for Movie Reviews 

This project is co-worked with [Kuan-Hung Chen](https://github.com/kuanhungchen) & [Chia-Wei Wu](https://github.com/marcovwu) 
## Introduction 

We design a pipeline to help people easily give their feedback or reviews of a movie. By using this system, users can provide a “draft” of their opinion and the system will automatically compute the semantic meaning of that draft, then find some alternatives of words from the gallery dataset. Finally, the system display one or some better writing of that draft to the user. With this system, users can give feedback after watching a movie with less effort.

## Method 
Our system can be roughly divided into two parts: multi-label text classification and candidate sentences ranking. The details of each state are explained in [report](https://drive.google.com/file/d/1TfEDJJuOB2nqghRGNvNZyJrcI-Suc4eX/view?usp=sharing).

![](https://i.imgur.com/6HpN4vA.png)

For text classification, we apply a BERT-based model pre-trained by [here](https://huggingface.co/bert-base-uncased), and fine-tuned on the [IMDb dataset of movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Specially, we manually annotate multiple labels about the genre of movies.


For candidate sentences ranking,  we compare similarity of the original words and all possible alternatives with embeddings, which produced by the well-trained word2vec model. Then we take all combinations of the alternatives and fill them back into the input sentence. To analyze the disparity, we use a sentence similarity model, borrowed from [Hugging Face](https://huggingface.co/tasks/sentence-similarity), to compute the distance between original sentence and alternative sentences. 



## Preparation & Installation 

The trained-weights and data our system needs are stored [here](https://drive.google.com/drive/folders/1VJp29A73TzXNTAjESy8cvupRs4qRFRc3?usp=sharing), where also provides IMDb reviews dataset as the gallery set. For using, you need to unzip them and save in local directory. You can also apply your own dataset as customized gallery for our system instead. 
### Installation
``pip install nltk`` \
``pip install -r requirement.txt``  

## Implementation
The **POS tag** defined in NLTK function represents the part of speech in words, such as JJ (adjective), VB (verb), etc. You can set the target POS tag in ``run.py`` then run the script.  

``python run.py``

