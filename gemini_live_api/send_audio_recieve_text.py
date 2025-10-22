# Test file: https://storage.googleapis.com/generativeai-downloads/data/16000.wav
# Install helpers for converting files: pip install librosa soundfile
import asyncio
import getpass
import io
import os
from pathlib import Path

import librosa
import soundfile as sf
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


client = genai.Client()

model = "gemini-2.0-flash-live-001"

config = {"response_modalities": ["TEXT"]}


async def main():
    async with client.aio.live.connect(model=model, config=config) as session:

        buffer = io.BytesIO()
        y, sr = librosa.load("/home/dev/projects/data/output.wav", sr=16000)
        sf.write(buffer, y, sr, format="RAW", subtype="PCM_16")
        buffer.seek(0)
        audio_bytes = buffer.read()

        # If already in correct format, you can use this:
        # audio_bytes = Path("sample.pcm").read_bytes()

        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )

        async for response in session.receive():
            if response.usage_metadata:
                usage = response.usage_metadata
                print(
                    f"Used {usage.total_token_count} tokens in total. Response token breakdown:"
                )
                for detail in usage.response_tokens_details:
                    match detail:
                        case types.ModalityTokenCount(
                            modality=modality, token_count=count
                        ):
                            print(f"{modality}: {count}")
            if response.text is not None:
                print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
