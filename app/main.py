import os
import sys
from io import BytesIO

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from PIL import Image
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.database import get_db
from app.models import Prediction
from ml.predict import load_model, predict_digits

# Initialize FastAPI and rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
model = load_model("/emnist_digits_recognizer.ckpt")


# Rate Limiting: Maximum 10 requests per minute
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429, content={"message": "Too many requests. Try again later."}
    )


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile, db: Session = Depends(get_db)):
    # Validate file format
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(
            status_code=400, detail="Only PNG and JPEG images are allowed"
        )

    # Read image and verify dimensions
    contents = await file.read()
    image = Image.open(BytesIO(contents))  # Convert to PIL Image

    # Ensure the image height is 28 and the width is 28 * n
    if image.size[1] != 28 or image.size[0] % 28 != 0:
        raise HTTPException(status_code=400, detail="Image must be 28x28n pixels")

    # Run digits prediction
    digits, confidence = predict_digits(model, image)

    # Store in database
    # db_record = Prediction(digit=digit, confidence=confidence)
    # db.add(db_record)
    # db.commit()

    return {"prediction": digits, "confidence": confidence}
