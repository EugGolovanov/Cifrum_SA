import asyncio

async def start_model():
    # Запуск модели на порту 8975
    process_db = await asyncio.create_subprocess_exec(
        "uvicorn", "test:app", "--host", "localhost", "--port", "8975"
    )
    await process_db.wait()

async def start_bot():
    # Запуск телеграм бота
    process_db = await asyncio.create_subprocess_exec("python", "bot.py")
    await process_db.wait()

async def main():
    # Параллельный запуск обоих процессов
    await asyncio.gather(start_model(), start_bot())

asyncio.run(main())