import os
import requests
from pipecat.services.tts_service import TTSService

HF_API_KEY = os.getenv("HF_API_KEY")

class HFKokoroTTSService(TTSService):
    def __init__(self, model="hexgrad/Kokoro-82M"):
        super().__init__()
        self.model = model

    async def tts(self, text: str) -> bytes:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"inputs": text}
        )

        if response.status_code != 200:
            return b""

        return response.content
