import re
import pandas as pd
import numpy as np
from inference import ClassificationDataset, get_loader, test, predict
from initial import init
from fastapi import FastAPI
from pydantic import BaseModel
from bertopic import BERTopic
from yandexgptlite import YandexGPTLite

p_d = './,][-")(~!#@^%$;*?&№∙^:<:>=_+\|`1°234}{567890'

name2int = {
    'Neutral': 0,
    'Positive': 1,
    'Negative': 2
}

def preprocess(text):
    output = text.replace('\n', ' ').replace('\t', ' ').replace('\u200c', ' ')
    output = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", output)
    output = re.sub(r'^https?:\/\/.*[\r\n]*', '', output, flags=re.MULTILINE)
    for k in p_d:
        output = output.replace(k, ' ')
    output = output.replace('  ', ' ')
    return output.strip()

flag = False
if flag:
    bert_cls, device = init(flag)
else:
    ner_model, sa1_model, bert_cls, device = init(flag)


# Подготовка шаблона для генерации текста
def generate_query(topics_pos, topics_neg):
    template = """Положительные отзывы: {topics_pos}\n\nОтрицательные отзывы: {topics_neg}"""
    return template.format(topics_pos=topics_pos, topics_neg=topics_neg)


app = FastAPI()


class SentimentRequest(BaseModel):
    reviews: list[str]


@app.post("/generate_answer")
async def predict_sentiment(request: SentimentRequest):
    reviews = request.reviews
    preprocessed_reviews = [preprocess(review) for review in reviews]
    df_reviews = pd.DataFrame({'reviews': preprocessed_reviews})

    if flag:
    # Предсказание тональности
        if len(reviews) < 16:
            predictions = [predict(preprocessed_reviews[i], bert_cls) for i in range(len(preprocessed_reviews))]
            # Разделение отзывов на положительные и отрицательные
            positive_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 1]
            negative_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 2]
        else:
            dataset = ClassificationDataset(df_reviews)
            dataloader = get_loader(dataset, shuffle=False, batch_size=32)
            predictions = test(bert_cls, dataloader, device)
            positive_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 1]
            negative_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 2]
    else:
        if len(reviews) < 16: 
            predictions = [predict(preprocessed_reviews[i], bert_cls) for i in range(len(preprocessed_reviews))]
            positive_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 1]
            negative_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 2]
        else:
            dataset = ClassificationDataset(df_reviews)
            dataloader = get_loader(dataset, shuffle=False, batch_size=32)
            predictions_sa0 = test(bert_cls, dataloader, device)
            sa1_model_preds = sa1_model(preprocessed_reviews)
            sa1_model_preds = [name2int[sa1_model_preds[i]] for i in range(len(sa1_model_preds))]
            sa1_model_preds = np.array(sa1_model_preds)
            predictions_sa0 = np.array(predictions_sa0)
            predictions = (sa1_model_preds + predictions_sa0-1) * 0.33/2
            predictions = predictions.tolist()
            positive_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 0.33]
            negative_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 0.495]
    
    topics_pos = None
    topics_neg = None
    response = None
    
    # Анализ топиков только для больших наборов отзывов
    if len(positive_reviews) > 15:
        topic_model_pos = BERTopic(language="multilingual")
        topic_model_pos.fit_transform(positive_reviews)
        topics_pos = topic_model_pos.get_topic_info()['Representative_Docs'][:7].values.tolist()

    if len(negative_reviews) > 15:
        topic_model_neg = BERTopic(language="multilingual")
        topic_model_neg.fit_transform(negative_reviews)
        topics_neg = topic_model_neg.get_topic_info()['Representative_Docs'][:7].values.tolist()

     # Интеграция с Yandex GPT
    if topics_pos is not None:
        topics_pos_str = "\n".join([", ".join(sublist) for sublist in topics_pos])
        topics_neg_str = "\n".join([", ".join(sublist) for sublist in topics_neg])
        prompt = """Ты сотрудник маркетингового отдела. На основании положительных и отрицательных отзывов \
            определи ключевые драйверы роста (что нравится пользователям) и ключевые \
            барьеры (что не нравится пользователям) развития продукта или услуг. \n
            Разработай рекомендации для маркетингового отдела. Старайся аргументировать свой ответ.
        """
        query = generate_query(topics_pos_str, topics_neg_str)
        account = YandexGPTLite("b1g***************", 'y0_A********************************')
        response = account.create_completion(query, '0.6', system_prompt = prompt)
        response = response.strip()

    return {
        "sentiments": predictions,
        "positive_topics": topics_pos,
        "negative_topics": topics_neg,
        "yandex_gpt_response": response
    }
