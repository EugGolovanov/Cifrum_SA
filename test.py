import re
from inference import ClassificationDataset, get_loader, test, predict
from initial import init
from fastapi import FastAPI
from pydantic import BaseModel
name2int = {
    'neutral': 0,
    'positive': 1,
    'negative': -1
}
p_d = './,][-")(~!#@^%$;*?&№∙^:<:>=_+\|`1°234}{567890'
def preprocess(text):
    output = text.replace('\n', ' ').replace('\t', ' ').replace('\u200c', ' ')
    output = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", output)
    output = re.sub(r'^https?:\/\/.*[\r\n]*', '', output, flags=re.MULTILINE)
    for k in p_d:
        output = output.replace(k, ' ')
    output = output.replace('  ', ' ')
    return output.strip()

flag = False
if not flag:
    bert_cls, device = init(flag)
else:
    ner_model, sa1_model, sa2_model, bert_cls, device = init(flag)

app = FastAPI()

class SentimentRequest(BaseModel):
    reviews: list[str]

@app.post("/generate_answer")
async def predict_sentiment(request: SentimentRequest):

    reviews = request.reviews
    preprocessed_reviews = [preprocess(review) for review in reviews]
    if flag:
        if len(reviews) < 16:
            predictions = [predict(preprocessed_reviews[i]) for i in range(len(preprocessed_reviews))]
        else:
            dataset = ClassificationDataset(preprocessed_reviews)
            dataloader = get_loader(dataset, shuffle=False, batch_size=32)
            predictions = test(bert_cls, dataloader, device)
    else:
        if len(reviews) < 16: #
            predictions_sa0 = [predict(preprocessed_reviews[i]) for i in range(len(preprocessed_reviews))]
        else:
            dataset = ClassificationDataset(preprocessed_reviews)
            dataloader = get_loader(dataset, shuffle=False, batch_size=32)
            predictions_sa0 = test(bert_cls, dataloader, device)
        ner_model_preds, sa1_model_preds, sa2_model_preds = ner_model(preprocessed_reviews), sa1_model(preprocessed_reviews), sa2_model(preprocessed_reviews)
        sa1_model_preds, sa2_model_preds =  \
            [name2int[sa1_model_preds[i]] for i in range(len(sa1_model_preds))], \
                [name2int[sa2_model_preds[i]] for i in range(len(sa2_model_preds))]
        predictions = (sa1_model_preds + predictions_sa0-1) *0.33/2 + sa2_model_preds
    return {"sentiments": predictions}
