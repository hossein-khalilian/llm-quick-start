import asyncio
import wave

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()
model = "gemini-2.0-flash-live-001"

config = {"response_modalities": ["AUDIO"], "output_audio_transcription": {}}


async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        wf = wave.open("./result.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        message = "Hello? Gemini are you there? tell me about ai"

        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": message}]}, turn_complete=True
        )

        async for response in session.receive():
            if response.server_content.model_turn:
                # wf.writeframes(response.server_content.model_turn)
                # print("Model turn:", response.server_content.model_turn)
                print()

            if response.server_content.output_transcription:
                print("Transcript:", response.server_content.output_transcription.text)


if __name__ == "__main__":
    asyncio.run(main())
