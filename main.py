from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Creole Platform SDK", version="0.1.0")

class TranslationRequest(BaseModel):
    text: str
    from_lang: str
    to_lang: str

class TranslationResponse(BaseModel):
    translated_text: str

@app.get("/")
async def root():
    return {"message": "Creole Platform SDK API"}

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    # Placeholder implementation
    return TranslationResponse(
        translated_text=f"Translated: {request.text} from {request.from_lang} to {request.to_lang}"
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}