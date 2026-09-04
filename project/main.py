from fastapi import FastAPI

from visual.routes import router as visual_router


app = FastAPI(
    title="GenAI Visual API",
    description="FastAPI service for image understanding using Gemma 3 4B",
    version="1.0.0"
)


# Register visual routes
app.include_router(visual_router)


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "GenAI Visual API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }