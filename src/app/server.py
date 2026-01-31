from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn


from src.models.predict import load_model

app = FastAPI(
    title="ESC50 Audio Classifier API",
    description="API for environmental sound classification",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    global model
    model_path="models/saved/model30012026_74valacc.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)

@app.get("/")
async def root():
    return {
        "status": "running"
    }

@app.get("/predict")
async def predict():
    return 

uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    log_level="info"
)