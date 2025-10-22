import asyncio
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()
model = "gemini-2.0-flash-live-001"

config = {
    "response_modalities": ["TEXT"],
    "input_audio_transcription": {},
}


async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        audio_data = Path("/home/dev/projects/data/sample_01.wav").read_bytes()

        await session.send_realtime_input(
            audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
        )

        async for msg in session.receive():
            if msg.server_content.input_transcription:
                print("Transcript:", msg.server_content.input_transcription.text)


if __name__ == "__main__":
    asyncio.run(main())
