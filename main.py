from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import gdown
import io
import os

app = FastAPI(title="Validacion Fotos API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/best.pt"

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    print("Descargando modelo desde Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1zWqG-Ydo8n37PWxePVESvuHxO5pUyxuk",
        MODEL_PATH,
        quiet=False
    )
    print("Modelo descargado.")

model = YOLO(MODEL_PATH)
print("Modelo listo.")

@app.get("/")
def health_check():
    return {"status": "ok", "modelo": "cargado"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model(image)

    detecciones = []
    for r in results:
        for box in r.boxes:
            detecciones.append({
                "clase": model.names[int(box.cls)],
                "confianza": round(float(box.conf), 3),
                "confianza_pct": f"{round(float(box.conf) * 100)}%"
            })

    mejor = None
    if detecciones:
        mejor = max(detecciones, key=lambda x: x["confianza"])

    return {
        "valido": mejor is not None and mejor["confianza"] >= 0.75,
        "total_detecciones": len(detecciones),
        "mejor_deteccion": mejor,
        "todas_detecciones": detecciones
    }
