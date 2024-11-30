import gc
import torch
import nltk
import os
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn


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


def init():
    clear_cache()
    set_seed(42)
    nltk.download('stopwords')
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_cls = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)
    return bert_cls, device

