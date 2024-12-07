import gc
import torch
import nltk
import os
import numpy as np
import random
from deeppavlov import build_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel
from torch import nn
import pickle


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
# Load model directly

def init(flag):
    clear_cache()
    set_seed(42)
    nltk.download('stopwords')
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "blanchefort/rubert-base-cased-sentiment-rusentiment"
    state_dict = torch.load('weights_adapter.bin')
    model = BertModel.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        num_labels=3
    )
    bert_cls = BertCLS(model, n_classes=3)
    bert_cls.fc.load_state_dict(
        state_dict
    )

    ner_model = build_model('ner_collection3_bert', download=True, install=True)
    sa1_model = AutoModelForSequenceClassification.from_pretrained("r1char9/rubert-base-cased-russian-sentiment", return_dict=True)
    return ner_model, sa1_model, bert_cls, device

