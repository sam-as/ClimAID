# llm_client.py
import requests

class LocalOllamaLLM:
    """
    Local LLM client using Ollama API (offline & free).
    Compatible with DiseaseReporter.
    """

    def __init__(self, model="mistral", host="http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # more scientific
                        "num_predict": 3000,  # allow long reports
                    }
                },
                timeout=600  # <-- CRITICAL FIX (was 120)
            )
            response.raise_for_status()
            return response.json()["response"]

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Local LLM timed out. The model is too slow or the prompt is very large."
            )

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Ollama is not running. Start it using: 'ollama serve'"
            )


