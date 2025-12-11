import os
import requests
from pipecat.services.llm_service import LLMService

HF_API_KEY = os.getenv("HF_API_KEY")

class HFQwenLLMService(LLMService):
    def __init__(self, model="Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__()
        self.model = model

    async def llm(self, messages):
        prompt = ""
        for m in messages:
            prompt += f"{m['role']}: {m['content']}\n"

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"inputs": prompt}
        )

        if response.status_code != 200:
            return "Error generating response"

        output = response.json()[0]["generated_text"]
        return output
