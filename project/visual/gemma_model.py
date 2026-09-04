import base64
import io
import json
import httpx


class GemmaModel:

    system_prompt = """
You are a strict image analysis assistant.

You MUST follow the user's requested task.

TASK RULES:

1. If the user asks to EXTRACT TEXT:
    - Look only for visible written text.
    - Return ONLY the text that is clearly readable.
    - Do NOT describe the image.
    - Do NOT mention objects, people, colors, or scenery.
    - Do NOT guess missing or unclear text.
    - If no readable text exists, respond exactly:
      No clearly visible text.

2. If the user asks to IDENTIFY OBJECTS:
    - List only objects clearly visible.
    - Do not guess.

3. If the user asks for a DESCRIPTION:
    - Describe only clearly visible information.
    - Do not infer relationships, age, location, emotions, or intentions.

4. If the user asks for a SUMMARY:
    - Summarize only clearly visible information.

Always follow the requested task exactly.
Never answer a different task.

For every image request, return ONLY valid JSON with exactly these fields:
{
    "description": "A concise description using only visible information.",
    "objects": ["clearly visible object"],
    "visible_text": ["clearly readable text"],
    "important_details": ["clearly visible detail"]
}
Use empty arrays when no items are clearly visible. Do not use Markdown or code fences.
"""

    def __init__(self):
        self.model_name = "gemma3:4b"
        self.ollama_url = "http://localhost:11434/api/chat"

        print(f"Using Ollama model: {self.model_name}")

    async def analyze_image(self, image, prompt: str) -> dict:

        # Convert PIL image to JPEG bytes
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
                    "role": "system",
                    "content": self.system_prompt
                },
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

        content = result["message"]["content"].strip()

        # Accept JSON wrapped in a code fence, but return only structured data.
        if content.startswith("```"):
            content = content.removeprefix("```").removeprefix("json").removesuffix("```").strip()

        try:
            structured_result = json.loads(content)
        except json.JSONDecodeError as error:
            raise RuntimeError("Ollama returned invalid structured JSON.") from error

        if not isinstance(structured_result, dict):
            raise RuntimeError("Ollama returned structured data in an invalid format.")

        return structured_result