import torch
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    # avg pooling by taking attention mask into account
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)


def hugging_face_embedding(model, tokenizer, sentences, GPU=False):
    # tokenize input sentences
    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors="pt")

    # compute token embeddings
    if GPU:
        model = model.cuda()
        encoded_input = {k: torch.LongTensor(v).cuda()
                         for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    # mean pooling
    sentence_embeddings = mean_pooling(model_output,
                                       encoded_input["attention_mask"])

    # normalization
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    if not GPU:
        sentence_embeddings = sentence_embeddings.cpu()

    return sentence_embeddings
