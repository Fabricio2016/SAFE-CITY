from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import gdown
import io
import os

app = FastAPI(title="Validacion Fotos API - Safe City Nodo")

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
model.overrides['imgsz'] = 640
model.overrides['conf'] = 0.35
model.overrides['device'] = 'cpu'
print("Modelo listo.")

# Mapeo de clases
CLASS_NAMES = {
    0: "ETIQUETA",
    1: "GABINET"
}

@app.get("/")
def health_check():
    return {"status": "ok", "modelo": "cargado", "clases": CLASS_NAMES}

@app.post("/predict/nodo-cerrado")
async def predict_nodo_cerrado(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    # Reducir tamaño para ahorrar RAM en plan Free
    image = image.resize((640, 640))
    results = model(image, imgsz=640, verbose=False)

    # Variables de detección
    gabinete_valido   = False
    etiqueta_valida   = False
    region_gabinete   = None
    region_etiqueta   = None
    coords_gabinete   = None
    coords_etiqueta   = None
    conf_gabinete     = 0.0
    conf_etiqueta     = 0.0
    todas_detecciones = []

    for r in results:
        for box in r.boxes:
            class_id   = int(box.cls)
            confianza  = round(float(box.conf), 3)
            bbox       = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            clase_nombre = CLASS_NAMES.get(class_id, f"clase_{class_id}")

            # Calcular región (cuadrante)
            img_w, img_h = image.size
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            h_pos = "izquierda" if cx < img_w / 2 else "derecha"
            v_pos = "superior" if cy < img_h / 2 else "inferior"
            region = f"{v_pos}_{h_pos}"

            deteccion = {
                "class_id":  class_id,
                "clase":     clase_nombre,
                "confianza": confianza,
                "confianza_pct": f"{round(confianza * 100)}%",
                "region":    region,
                "bbox":      [round(c, 1) for c in bbox]
            }
            todas_detecciones.append(deteccion)

            # GABINET (class_id == 1)
            if class_id == 1 and confianza > conf_gabinete:
                gabinete_valido = confianza >= 0.60
                conf_gabinete   = confianza
                region_gabinete = region
                coords_gabinete = [round(c, 1) for c in bbox]

            # ETIQUETA (class_id == 0)
            if class_id == 0 and confianza > conf_etiqueta:
                etiqueta_valida = confianza >= 0.55
                conf_etiqueta   = confianza
                region_etiqueta = region
                coords_etiqueta = [round(c, 1) for c in bbox]

    # Resultado final
    nodo_valido = gabinete_valido and etiqueta_valida

    motivo = ""
    if not gabinete_valido and not etiqueta_valida:
        motivo = "No se detectó el gabinete ni la etiqueta del Nodo Safe City"
    elif not gabinete_valido:
        motivo = "No se detectó el gabinete del Nodo Safe City"
    elif not etiqueta_valida:
        motivo = "No se detectó la etiqueta en el Nodo Safe City"
    else:
        motivo = "Gabinete y etiqueta detectados correctamente"

    return {
        "nodo_valido":      nodo_valido,
        "gabinete_valido":  gabinete_valido,
        "etiqueta_valida":  etiqueta_valida,
        "conf_gabinete":    round(conf_gabinete * 100),
        "conf_etiqueta":    round(conf_etiqueta * 100),
        "region_gabinete":  region_gabinete,
        "region_etiqueta":  region_etiqueta,
        "coords_gabinete":  coords_gabinete,
        "coords_etiqueta":  coords_etiqueta,
        "motivo":           motivo,
        "total_detecciones": len(todas_detecciones),
        "todas_detecciones": todas_detecciones
    }
