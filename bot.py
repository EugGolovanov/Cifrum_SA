import asyncio
import tempfile
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.dispatcher.filters import Command
import aiofiles
import aiohttp
import csv

# Токен бота
token = "YOUR_BOT_TOKEN"
bot = Bot(token)
dp = Dispatcher(bot)

API_URL = "http://127.0.0.1:8975/generate_answer"

@dp.message_handler(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот для обработки отзывов. Пришли текст отзыва или CSV файл.")

async def send_reviews_to_api(reviews: list[str]) -> list[float]:
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json={"reviews": reviews}) as response:
            response_data = await response.json()
            return response_data.get("sentiments", [])

@dp.message_handler(content_types=['document'])
async def handle_file(message: Message):
    document = message.document
    if document.file_name.endswith('.csv'):
        # Скачиваем содержимое файла
        file = await bot.download_file_by_id(document.file_id)
        file_data = file.read()  # Здесь убираем 'await', так как метод 'read()' возвращает байты синхронно

        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name

        reviews = []
        async with aiofiles.open(temp_file_path, mode='r') as file:
            async for line in file:
                reviews.append(line.strip().split(",")[0])  # Считываем первую колонку

        if reviews:
            sentiments = await send_reviews_to_api(reviews)
            await message.answer(f"Sentiment scores: {sentiments}")
        else:
            await message.answer("CSV файл пуст или содержит некорректные данные.")
    else:
        await message.answer("Пожалуйста, отправьте файл в формате CSV.")

@dp.message_handler(content_types=['text'])
async def handle_text(message: Message):
    reviews = [message.text]
    sentiments = await send_reviews_to_api(reviews)
    await message.answer(f"Sentiment scores: {sentiments}")

async def main():
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())