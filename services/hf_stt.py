import os
import requests
from pipecat.services.stt_service import STTService

HF_API_KEY = os.getenv("HF_API_KEY")

class HFWhisperSTTService(STTService):
    def __init__(self, model="openai/whisper-small"):
        super().__init__()
        self.model = model

    async def stt(self, audio_bytes: bytes) -> str:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            data=audio_bytes
        )
        
        if response.status_code != 200:
            return ""
        
        return response.json().get("text", "")
