import asyncio

from sqlalchemy import text

from backend.app.database import engine


async def test_database():
    try:
        async with engine.connect() as connection:
            result = await connection.execute(text("SELECT NOW()"))
            print("DATABASE CONNECTION: SUCCESS")
            print("Server time:", result.scalar())

    except Exception as e:
        print("DATABASE CONNECTION: FAILED")
        print(e)

    finally:
        await engine.dispose()


asyncio.run(test_database())
