import asyncio

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()
model = "gemini-2.0-flash-live-001"

tools = [{"google_search": {}}]
config = {"response_modalities": ["TEXT"], "tools": tools}


async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        prompt = "When did the last Brazil vs. Argentina soccer match happen?"
        await session.send_client_content(turns={"parts": [{"text": prompt}]})

        async for chunk in session.receive():
            if chunk.server_content:
                if chunk.text is not None:
                    print(chunk.text)

                # The model might generate and execute Python code to use Search
                model_turn = chunk.server_content.model_turn
                if model_turn:
                    for part in model_turn.parts:
                        if part.executable_code is not None:
                            print(part.executable_code.code)

                        if part.code_execution_result is not None:
                            print(part.code_execution_result.output)


if __name__ == "__main__":
    asyncio.run(main())
