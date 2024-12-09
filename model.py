import pandas as pd
import numpy as np
from inference import ClassificationDataset, get_loader, test_logits, predict_logits
from initial import init
from fastapi import FastAPI
from pydantic import BaseModel
from bertopic import BERTopic
from yandexgptlite import YandexGPTLite
from process import process_ner, preprocess, int2name
from nltk import word_tokenize, ngrams
from collections import Counter
from time import time
flag = False
if flag:
    bert_cls, device = init(flag)
else:
    ner_model, sa1_model, bert_cls, device = init(flag)


# Подготовка шаблона для генерации текста
def generate_query(topics_pos, topics_neg):
    template = """Положительные отзывы: {topics_pos}\n\nОтрицательные отзывы: {topics_neg}"""
    return template.format(topics_pos=topics_pos, topics_neg=topics_neg)

#  Извлекаем наиболее частые n-граммы из списка отзывов.
def extract_common_phrases(reviews, n=3, top_n=10):
    text = " ".join(reviews)
    tokens = word_tokenize(preprocess(text.lower()))
    n_grams = ngrams(tokens, n)
    n_grams_freq = Counter(n_grams)
    return n_grams_freq.most_common(top_n)

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
            predictions_1 = np.array( df_reviews['reviews'].apply(lambda x: predict_logits(x, bert_cls, False)))
            predictions_2 = np.array( df_reviews['reviews'].apply(lambda x: predict_logits(x, sa1_model, True)))
            combined_data = np.vstack(predictions_1 + predictions_2)
            predictions = np.argmax(combined_data, axis=1).tolist()
            positive_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 2]
            negative_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 0]
    else:
            dataset = ClassificationDataset(df_reviews)
            dataloader = get_loader(dataset, shuffle=False, batch_size=8)
            predictions_sa0 = test_logits(bert_cls, dataloader, device, False)
            sa1_model_preds = test_logits(sa1_model, dataloader, device, True)
            predictions = np.argmax((predictions_sa0 + sa1_model_preds), axis=1).tolist()
            positive_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 2]
            negative_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 0]
    predictions = [int2name[i] for i in predictions]

    # NER
    NER_preds = ner_model(preprocessed_reviews)
    NER_preds = process_ner(NER_preds)
    
    topics_pos = None
    topics_neg = None
    response = None
    positive_phrases = None
    negative_phrases = None
      
    # Анализ топиков только для больших наборов отзывов
    if len(positive_reviews) > 15:
        topic_model_pos = BERTopic(language="multilingual")
        topic_model_pos.fit_transform(positive_reviews)
        topics_pos = topic_model_pos.get_topic_info()['Representative_Docs'][:7].values.tolist()
        positive_common_phrases = extract_common_phrases(positive_reviews, n=4, top_n=4)
        positive_phrases = [{"phrase": " ".join(phrase), "frequency": freq} for phrase, freq in positive_common_phrases]
        
    if len(negative_reviews) > 15:
        topic_model_neg = BERTopic(language="multilingual")
        topic_model_neg.fit_transform(negative_reviews)
        topics_neg = topic_model_neg.get_topic_info()['Representative_Docs'][:7].values.tolist()
        negative_common_phrases = extract_common_phrases(negative_reviews, n=4, top_n=4)
        negative_phrases = [{"phrase": " ".join(phrase), "frequency": freq} for phrase, freq in negative_common_phrases]
 
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
        account = YandexGPTLite("b1ge****************", 'y0_AgAAAABU****************************')
        response = account.create_completion(query, '0.6', system_prompt = prompt)
        response = response.strip()
        
    return {
        "sentiments": predictions,
        "positive_topics": topics_pos,
        "negative_topics": topics_neg,
        "NER": NER_preds,
        "yandex_gpt_response": response,
        "positive_phrases": positive_phrases,
        "negative_phrases": negative_phrases
    }
