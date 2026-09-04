from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from .image_processor import (
    validate_image,
    load_image
)

from .gemma_model import GemmaModel

from .schemas import VisualResponse


router = APIRouter(
    prefix="/visual",
    tags=["Visual"]
)


gemma = GemmaModel()


@router.post(
    "/analyze",
    response_model=VisualResponse
)
async def analyze_image(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):

    try:

        # 1. Validate image
        validate_image(image.content_type)

        # 2. Read image
        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty image file."
            )

        # 3. Convert to PIL image
        pil_image = load_image(image_bytes)

        # 4. Send image + prompt to Gemma through Ollama
        result = await gemma.analyze_image(
            pil_image,
            prompt
        )

        # 5. Return response
        return VisualResponse(
            status="success",
            message="Image analyzed successfully.",
            result=result
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except HTTPException:

        raise

    except Exception as e:

        print(f"Error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )