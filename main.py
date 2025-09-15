from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from utils import measure_front_dimensions, measure_depths, combine_measurements
import shutil
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],   # allows POST, OPTIONS (preflight), etc.
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/measure")
async def measure_body(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    height_cm: float = Form(...),
    gender: str = Form("men")  # ✅ New dropdown field, default = "men"
):
    try:
        # Save uploaded images with unique names
        front_path = os.path.join(UPLOAD_DIR, f"front_{uuid.uuid4()}.jpg")
        side_path = os.path.join(UPLOAD_DIR, f"side_{uuid.uuid4()}.jpg")

        with open(front_path, "wb") as f:
            shutil.copyfileobj(front_image.file, f)
        with open(side_path, "wb") as f:
            shutil.copyfileobj(side_image.file, f)

        # --- Run measurement pipeline ---
        front_data, front_error = measure_front_dimensions(front_path, height_cm)
        if front_data is None:
            return JSONResponse(content={"error": front_error}, status_code=400)

        side_data, side_error = measure_depths(side_path, front_data["pixels_per_cm"])
        if side_data is None:
            return JSONResponse(content={"error": side_error}, status_code=400)

        result = combine_measurements(front_data, side_data, gender)

        # ✅ Add gender to the result
        result["gender"] = gender
        result["height_cm"] = height_cm

        # Clean up uploaded files
        os.remove(front_path)
        os.remove(side_path)

        return result

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
