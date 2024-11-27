import gc 
import torch
import nltk
import os
from deeppavlov import build_model
from transformers import AutoTokenizer, BertModel, AutoModelForSequenceClassification
from torch import nn
import pickle
import numpy as np
import random


class BertCLS(nn.Module):
    def __init__(self, model, n_classes):
        super(BertCLS, self).__init__()
        self.model = model
        self.fc = nn.Linear(768, n_classes)

    def forward(self, batch):
        return self.fc(self.model(**batch).pooler_output)

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init(flag):
    clear_cache()
    set_seed(42)
    nltk.download('stopwords')
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "blanchefort/rubert-base-cased-sentiment-rusentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open('path2model/bert_model.pkl', 'rb') as f:
        bert_cls = pickle.load(f)
    if flag:
        ner_model = build_model('ner_collection3_bert', download=True, install=True)
        sa1_model = build_model('sentiment_twitter', download=True, install=True)
        sa2_model = build_model('rusentiment_convers_bert', download=True, install=True)
        return ner_model, sa1_model, sa2_model, bert_cls, device
    else:
        return bert_cls, device

