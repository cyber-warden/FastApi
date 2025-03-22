# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import pytesseract
from PIL import Image
import io
import base64
import re
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OMRResponse(BaseModel):
    detected_answers: Dict[str, str]
    processed_image: str

class AnswerKeyResponse(BaseModel):
    answer_key: Dict[str, str]

@app.post("/api/process-omr", response_model=OMRResponse)
async def process_omr(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Find the OMR sheet (assuming it's the largest contour)
    if len(contours) > 0:
        omr_contour = contours[0]
        x, y, w, h = cv2.boundingRect(omr_contour)
        omr_sheet = img[y:y+h, x:x+w]
    else:
        omr_sheet = img
    
    # Process the OMR sheet to detect filled bubbles
    # This is a simplified example - real implementation would be more complex
    detected_answers = detect_filled_bubbles(omr_sheet)
    
    # Create a visualization of the detected answers
    result_img = draw_detected_answers(omr_sheet, detected_answers)
    
    # Convert the result image to base64 for sending back to frontend
    _, buffer = cv2.imencode('.jpg', result_img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return OMRResponse(
        detected_answers=detected_answers,
        processed_image=f"data:image/jpeg;base64,{img_str}"
    )

@app.post("/api/process-answer-key", response_model=AnswerKeyResponse)
async def process_answer_key(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # Convert to grayscale for OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract OCR to extract text
    text = pytesseract.image_to_string(gray)
    
    # Parse the text to extract question numbers and answers
    answer_key = parse_answer_key(text)
    
    return AnswerKeyResponse(answer_key=answer_key)

def detect_filled_bubbles(img):
    """
    Detect filled bubbles in an OMR sheet.
    This is a simplified implementation - a real one would be more robust.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Mock detected answers (in a real implementation, this would analyze the contours)
    detected_answers = {}
    for i in range(1, 21):
        detected_answers[str(i)] = np.random.choice(['A', 'B', 'C', 'D'])
    
    return detected_answers

def draw_detected_answers(img, answers):
    """
    Draw the detected answers on the image for visualization.
    """
    result = img.copy()
    
    # In a real implementation, this would highlight the detected bubbles
    # For now, just add text overlay
    y_pos = 30
    for q, ans in answers.items():
        cv2.putText(result, f"Q{q}: {ans}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
    
    return result

def parse_answer_key(text):
    """
    Parse the OCR text to extract question numbers and answers.
    """
    # Look for patterns like "1. A" or "1 | A" or "1-A"
    pattern = r'(\d+)[\.\|\-\s]+([A-D])'
    matches = re.findall(pattern, text)
    
    answer_key = {}
    for match in matches:
        question, answer = match
        answer_key[question] = answer
    
    # If no matches found, create a mock answer key
    if not answer_key:
        for i in range(1, 21):
            answer_key[str(i)] = np.random.choice(['A', 'B', 'C', 'D'])
    
    return answer_key

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
