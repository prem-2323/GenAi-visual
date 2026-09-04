import base64
import httpx


class GemmaModel:

    def __init__(self):
        self.model_name = "gemma3:4b"
        self.ollama_url = "http://localhost:11434/api/chat"

        print(f"Using Ollama model: {self.model_name}")

    async def analyze_image(self, image, prompt: str) -> str:

        # Convert PIL image to JPEG bytes
        import io

        image_buffer = io.BytesIO()
        image.save(image_buffer, format="JPEG")

        image_bytes = image_buffer.getvalue()

        # Convert image to Base64
        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        # Send request to Ollama
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        image_base64
                    ]
                }
            ],
            "stream": False
        }

        async with httpx.AsyncClient(timeout=120.0) as client:

            response = await client.post(
                self.ollama_url,
                json=payload
            )

        response.raise_for_status()

        result = response.json()

        return result["message"]["content"]