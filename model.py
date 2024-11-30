import re
import pandas as pd
from inference import ClassificationDataset, get_loader, test, predict
from initial import init
from fastapi import FastAPI
from pydantic import BaseModel
from bertopic import BERTopic
from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate
from create_iam_token import get_iam_token

p_d = './,][-")(~!#@^%$;*?&№∙^:<:>=_+\|`1°234}{567890'


def preprocess(text):
    output = text.replace('\n', ' ').replace('\t', ' ').replace('\u200c', ' ')
    output = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", output)
    output = re.sub(r'^https?:\/\/.*[\r\n]*', '', output, flags=re.MULTILINE)
    for k in p_d:
        output = output.replace(k, ' ')
    output = output.replace('  ', ' ')
    return output.strip()


bert_cls, device = init()
app = FastAPI()


class SentimentRequest(BaseModel):
    reviews: list[str]


@app.post("/generate_answer")
async def predict_sentiment(request: SentimentRequest):
    reviews = request.reviews
    preprocessed_reviews = [preprocess(review) for review in reviews]

    # Предсказание тональности
    if len(reviews) < 16:
        predictions = [predict(preprocessed_reviews[i], bert_cls) for i in range(len(preprocessed_reviews))]
    else:
        dataset = ClassificationDataset(pd.DataFrame(preprocessed_reviews, columns=['reviews']))
        dataloader = get_loader(dataset, shuffle=False, batch_size=32)
        predictions = test(bert_cls, dataloader, device)

    # Разделение отзывов на положительные и отрицательные
    positive_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 1]
    negative_reviews = [preprocessed_reviews[i] for i in range(len(predictions)) if predictions[i] == 2]

    topics_pos = None
    topics_neg = None
    response = None

    # Анализ топиков только для больших наборов отзывов
    if len(positive_reviews) > 16:
        topic_model_pos = BERTopic(language="multilingual")
        topic_model_pos.fit_transform(positive_reviews)
        topics_pos = topic_model_pos.get_topic_info()['Representative_Docs'][:7].values.tolist()

    if len(negative_reviews) > 16:
        topic_model_neg = BERTopic(language="multilingual")
        topic_model_neg.fit_transform(negative_reviews)
        topics_neg = topic_model_neg.get_topic_info()['Representative_Docs'][:7].values.tolist()

     # Интеграция с Yandex GPT
    if topics_pos is not None:
        topics_pos_str = "\n".join([", ".join(sublist) for sublist in topics_pos])
        topics_neg_str = "\n".join([", ".join(sublist) for sublist in topics_neg])
        topics_pos_str = "\n".join([", ".join(sublist) for sublist in topics_pos])
        topics_neg_str = "\n".join([", ".join(sublist) for sublist in topics_neg])

        inputs = {
            "topics_pos": topics_pos_str,
            "topics_neg": topics_neg_str
        }

        template = """Положительные отзывы: {topics_pos}\n\nОтрицательные отзывы: {topics_neg}\n
            На основании положительны и отрицательных отзывов определите ключевые драйверы роста (что нравится пользователям) и ключевые \
            барьеры (что не нравится пользователям) развития продукта или услуг. \n
            Разработай рекомендации для маркетингового отдела.\n
            """
        prompt = PromptTemplate.from_template(template)
        iam_token = get_iam_token()
        yandex_gpt = YandexGPT(iam_token=iam_token, folder_id="b1ge2***********", model_name="yandexgpt-lite", temperature=0.2)
        llm_sequence = prompt | yandex_gpt
        response = llm_sequence.invoke(inputs).strip()

    return {
        "sentiments": predictions,
        "positive_topics": topics_pos,
        "negative_topics": topics_neg,
        "yandex_gpt_response": response
    }
