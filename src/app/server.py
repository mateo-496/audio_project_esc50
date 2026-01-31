import uvicorn
import torch
import tempfile
import os 

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.models.predict import load_model, predict_file
from src.config.config import esc50_labels

app = FastAPI(
    title="ESC50 Audio Classifier API",
    description="API for environmental sound classification",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    global model
    global device
    model_path="models/saved/model30012026_74valacc.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)

@app.get("/")
async def root():
    return FileResponse("src/app/index.html")

@app.get("/api/status")
async def status():
    return {
        "status": "running"
    }

@app.post("/predict-top-k")
async def predict_top_k(file: UploadFile = File(...), k: int = 5):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        predicted_class, top_probs, top_indices = predict_file(
            model, tmp_path, device=device, top_k=k
        )        

        os.unlink(tmp_path)
        
        return {
            'top_predictions': [
                {'class': esc50_labels[idx], 'confidence': float(prob)}
                for prob, idx in zip(top_probs, top_indices)
            ],
            'predicted_class': esc50_labels[predicted_class],
            'confidence': float(top_probs[0])
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    log_level="info"
)