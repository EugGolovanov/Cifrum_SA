import re
from inference import ClassificationDataset, get_loader, test, predict

from fastapi import FastAPI
from pydantic import BaseModel

def preprocess(text):
    output = text.replace('\n', ' ').replace('\t', ' ').replace('\u200c', ' ')
    output = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", output)
    output = re.sub(r'^https?:\/\/.*[\r\n]*', '', output, flags=re.MULTILINE)
    for k in p_d:
        output = output.replace(k, ' ')
    output = output.replace('  ', ' ')
    return output.strip()


app = FastAPI()

class SentimentRequest(BaseModel):
    reviews: list[str]

@app.post("/generate_answer")
async def predict_sentiment(request: SentimentRequest):

    reviews = request.reviews
    if len(reviews) < 16: #
        preprocessed_reviews = [preprocess(review) for review in reviews]
        predictions = predict(preprocessed_reviews)
    else:
        dataset = ClassificationDataset(reviews)
        dataloader = get_loader(dataset, shuffle=False, batch_size=32)
        predictions = test(bert_cls, dataloader, device)
    return {"sentiments": predictions}
