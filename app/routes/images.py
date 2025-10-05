from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.utils.detector import detect_ppe
import os
import json

router = APIRouter()
UPLOAD_FOLDER = "static/overlays"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HISTORY_FILE = "static/history.json"
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

router.mount("/static", StaticFiles(directory="static"), name="static")

@router.post("/")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = detect_ppe(file_path)
    # Save to history
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    history.append({
        "filename": file.filename,
        "helmet": result["helmet"],
        "vest": result["vest"],
        "bboxes": result["bboxes"],
        "overlay_url": result["overlay_url"]
    })
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    return JSONResponse(result)

@router.get("/history")
async def get_history():
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    return JSONResponse(history)
