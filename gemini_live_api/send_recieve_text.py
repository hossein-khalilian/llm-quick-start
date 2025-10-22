import asyncio

from dotenv import load_dotenv
from google import genai

load_dotenv()

import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

client = genai.Client()


model = "gemini-2.0-flash-live-001"

config = {"response_modalities": ["TEXT"]}


async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        message = "Hello, how are you?"
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": message}]}, turn_complete=True
        )

        async for response in session.receive():
            if response.text is not None:
                print(response.text, end="")


if __name__ == "__main__":
    asyncio.run(main())
