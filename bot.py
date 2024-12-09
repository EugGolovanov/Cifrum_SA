import asyncio
import tempfile
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.dispatcher.filters import Command
import aiofiles
import aiohttp
import math
import pandas as pd
import os

# Токен бота
token = "74924******************"
bot = Bot(token)
dp = Dispatcher(bot)

API_URL = "http://127.0.0.1:8975/generate_answer"

@dp.message_handler(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот для обработки отзывов. Пришли текст отзыва или CSV файл.")

async def send_reviews_to_api(reviews: list[str]) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json={"reviews": reviews}) as response:
            response_data = await response.json()
            return {
                "sentiments": response_data.get("sentiments", []),
                "positive_topics": response_data.get("positive_topics", []),
                "negative_topics": response_data.get("negative_topics", []),
                "ner": response_data.get("NER", []),
                "yandex_gpt_response": response_data.get("yandex_gpt_response", []),
                "positive_phrases": response_data.get("positive_phrases", []),
                "negative_phrases": response_data.get("negative_phrases", [])
            }


async def safe_edit_text(message: types.Message, new_text: str):
    try:
        if message.text != new_text:
            await message.edit_text(new_text)
    except Exception:
        pass

@dp.message_handler(content_types=['document'])
async def handle_file(message: Message):
    document = message.document
    if document.file_name.endswith('.csv'):
        progress_message = await message.answer("🔄 Обрабатываю файл...")

        file = await bot.download_file_by_id(document.file_id)
        file_data = file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name

        try:
            reviews = []
            total_rows = 0

            async with aiofiles.open(temp_file_path, mode='r') as temp_file:
                reader = pd.read_csv(temp_file_path, chunksize=1000)
                
                for idx, chunk in enumerate(reader):
                    chunk_reviews = chunk['reviews'].tolist()
                    reviews.extend(chunk_reviews)
                    total_rows += len(chunk_reviews)

                    progress = math.ceil((idx * 1000) / total_rows * 100)
                    await safe_edit_text(progress_message, f"🔄 Считываю данные: {progress}%")

            # Отправляем данные на бэкенд
            await safe_edit_text(progress_message, "✅ Анализирую данные...")
            api_response = await send_reviews_to_api(reviews)

            # Обрабатываем ответ
            sentiments = api_response["sentiments"]
            positive_topics = api_response["positive_topics"]
            negative_topics = api_response["negative_topics"]
            yandex_gpt_response = api_response["yandex_gpt_response"]
            ner_response = api_response['ner']
            positive_phrases = api_response["positive_phrases"]
            negative_phrases = api_response["negative_phrases"]

            # Форматируем вывод
            response_message = (
                f"Sentiment scores:\n{sentiments[:10]}\n\n"
                #f"Positive Topics:\n{positive_topics}\n\n"
                #f"Negative Topics:\n{negative_topics}\n\n"
                #f"Named Entities:\n{ner_response}\n\n"
                f"Positive phrases:\n" + "\n".join([f"{item['phrase']}: {item['frequency']}" for item in positive_phrases]) + "\n\n"
                f"Negative phrases:\n" + "\n".join([f"{item['phrase']}: {item['frequency']}" for item in negative_phrases]) + "\n\n"
                f"LLM_response:\n{yandex_gpt_response}"
            )
            await message.answer(response_message)
            
            response_message = (
                f"Sentiment scores:\n{sentiments}\n\n"
                f"{'-' * 50}\n"
                f"Positive Topics:\n{positive_topics}\n\n"
                f"{'-' * 50}\n"
                f"Negative Topics:\n{negative_topics}\n\n"
                f"{'-' * 50}\n"
                f"Named Entities:\n{ner_response}\n\n"
                f"{'-' * 50}\n"
                f"Positive phrases:\n" + "\n".join([f"{item['phrase']}: {item['frequency']}" for item in positive_phrases]) + "\n\n"
                f"{'-' * 50}\n"
                f"Negative phrases:\n" + "\n".join([f"{item['phrase']}: {item['frequency']}" for item in negative_phrases]) + "\n\n"
                f"{'-' * 50}\n"
                f"LLM_response:\n{yandex_gpt_response}"
            )
            
            temp_dir = tempfile.gettempdir()
            result_file_path = os.path.join(temp_dir, "результаты_анализа.txt")
            with open(result_file_path, mode="w") as result_file:
                result_file.write(response_message)

            await bot.send_document(chat_id=message.chat.id, document=open(result_file_path, "rb"))
        
        except Exception as e:
            await safe_edit_text(progress_message, f"❌ Ошибка обработки файла: {e}")
    else:
        await message.answer("Пожалуйста, отправьте файл в формате CSV.")



@dp.message_handler(content_types=['text'])
async def handle_text(message: Message):
    reviews = [message.text]
    api_response = await send_reviews_to_api(reviews)
    sentiments = api_response["sentiments"]
    #positive_topics = api_response["positive_topics"]
    #negative_topics = api_response["negative_topics"]
    #yandex_gpt_response = api_response["yandex_gpt_response"]
    ner_response = api_response['ner']

    # Форматируем вывод
    response_message = (
        f"Sentiment scores:\n{sentiments}\n\n"
        #f"Positive Topics:\n{positive_topics}\n\n"
        #f"Negative Topics:\n{negative_topics}\n\n"
        #f"Named Entities:\n{ner_response}\n\n"
        #f"LLM_response:\n{yandex_gpt_response}"
    )
    await message.answer(response_message)


async def main():
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())
