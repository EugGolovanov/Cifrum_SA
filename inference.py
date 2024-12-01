import torch
from torch import nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class BertCLS(nn.Module):
    def __init__(self, model, n_classes):
        super(BertCLS, self).__init__()
        self.model = model
        self.fc = nn.Linear(768, n_classes)

    def forward(self, batch):
        return self.fc(self.model(**batch).pooler_output)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "blanchefort/rubert-base-cased-sentiment-rusentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)


class ClassificationDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.text = data['reviews'].tolist()
        # self.target = data.target.to_list()

    def __getitem__(self, idx):
        text = self.text[idx]
        return text

    def __len__(self):
        return len(self.text)


def collate_fn(batch):
    model_input = [text for text in batch]
    tok = tokenizer(model_input, padding=True, max_length=300, truncation=True, return_tensors='pt')
    return {key: value.to(device) for key, value in tok.items()}  # Перемещение на устройство


def get_loader(dataset, shuffle, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    return loader


def test(model, loader, device):
    pred = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for batch_idx, data in enumerate(pbar):
            data = {key: value.to(device) for key, value in data.items()}
            outputs = model(**data)
            pred.extend(outputs.logits.argmax(-1).detach().cpu().numpy().tolist())
    return pred


def predict(text, bert_cls):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = bert_cls(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy().tolist()
    return predicted
