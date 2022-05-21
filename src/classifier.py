import os
import nltk
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel


os.environ["TOKENIZERS_PARALLELISM"] = "False"


class TextClassifierModel(nn.Module):
    def __init__(self, n_classes, BERT_MODEL_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME,
                                              return_dict=True)

        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = dict()
        if labels is not None:
            loss["total"] = self.criterion(output, labels.float())
        return loss, output


def load_classifier_model(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL_NAME)
    model = TextClassifierModel(cfg.N_CLASSES, cfg.BERT_MODEL_NAME)
    if cfg.MODEL_STATE_DICT:
        model.load_state_dict(
            torch.load(cfg.MODEL_STATE_DICT,
                       map_location=torch.device("cpu"))["model_state_dict"])

    return model, tokenizer


def tokenize(sent, tokenizer, MAX_TOKEN_LENGTH):
    encoding = tokenizer(sent,
                         return_token_type_ids=False,
                         padding=True,
                         max_length=MAX_TOKEN_LENGTH,
                         truncation=True,
                         return_attention_mask=True,
                         return_tensors="pt")

    return encoding


def test(sent, model, stopwords, tokenizer, MAX_TOKEN_LENGTH, THRESHOLD, GPU):
    # encode sentence
    sent = [" ".join([w for w in nltk.word_tokenize(s)
        if w.lower() not in stopwords]).lower() for s in sent]
    encoding = tokenize(sent, tokenizer, MAX_TOKEN_LENGTH)

    inputs = dict()
    inputs["input_ids"] = encoding["input_ids"]
    inputs["attention_mask"] = encoding["attention_mask"]
    if GPU:
        model = model.cuda()
        inputs = {k: torch.LongTensor(v).cuda() for k, v in inputs.items()}

    # model inference
    with torch.no_grad():
        _, outputs = model(**inputs)

    return np.where(outputs.cpu().numpy() > THRESHOLD)


def inference(model, test_set, batch_size, tokenizer,
              MAX_TOKEN_LENGTH, THRESHOLD, GPU):
    test_loader = DataLoader(test_set, batch_size=batch_size)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    stopwords = set(nltk.corpus.stopwords.words("english"))

    # testing
    class_list = list()
    for i, batch in enumerate(test_loader):
        class_list.append(test([b for b in list(batch[0])],
                               model,
                               stopwords,
                               tokenizer,
                               MAX_TOKEN_LENGTH,
                               THRESHOLD,
                               GPU))
    final_list = list()
    num_list = [min(batch_size, len(test_set) - i)
        for i in range(0, len(test_set), batch_size)]

    for i, nums in enumerate(num_list):
        final_list.extend([class_list[i][1][class_list[i][0] == b]
                          for b in range(nums)])

    return final_list
