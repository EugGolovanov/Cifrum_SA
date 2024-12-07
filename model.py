import re
import pandas as pd
import numpy as np
from inference import ClassificationDataset, get_loader,predict,test, test_logits, predict_logits
from initial import init
from fastapi import FastAPI
from pydantic import BaseModel
from bertopic import BERTopic
from yandexgptlite import YandexGPTLite
from process import process_ner, preprocess, int2name
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

    # Sentiement Analysis
    if len(reviews) < 16:
            predictions_1 = np.array(
                [predict_logits(
                    preprocessed_reviews[i], bert_cls, False
                ) for i in range(len(preprocessed_reviews))]
            )
            predictions_2 = np.array(
                [predict_logits(
                    preprocessed_reviews[i], sa1_model, True
                ) for i in range(len(preprocessed_reviews))]
            )
            predictions = np.argmax(predictions_1 + predictions_2, axis=1).tolist()
            positive_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 2]
            negative_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 0]
    else:
            dataset = ClassificationDataset(df_reviews)
            dataloader = get_loader(dataset, shuffle=False, batch_size=32)
            predictions_sa0 = test_logits(bert_cls, dataloader, device)
            sa1_model_preds = test_logits(sa1_model, dataloader, device)
            predictions = np.argmax((predictions_sa0 + sa1_model_preds), axis=1).tolist()
            positive_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 2]
            negative_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 0]
    predictions = [int2name[predictions[i]] for i in range(len(predictions))]

    # NER
    NER_preds = ner_model(preprocessed_reviews)
    NER_preds = process_ner(NER_preds)

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
        "NER": NER_preds,
        "negative_topics": topics_neg,
        "yandex_gpt_response": response
    }
