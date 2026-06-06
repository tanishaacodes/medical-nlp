import cv2
import numpy as np
from PIL import Image

def clean_handwriting(uploaded_file):
    # 1. Convert the file object into a format OpenCV understands
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        return None

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Rescale to standard height
    height = 64
    ratio = height / gray.shape[0]
    width = int(gray.shape[1] * ratio)
    resized = cv2.resize(gray, (width, height))

    # 4. Adaptive Thresholding (The "Eyes")
    cleaned = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 5. Denoise
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return Image.fromarray(cleaned).convert("RGB")

def get_word_segments(uploaded_file):
    file_bytes=np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    img=cv2.imdecode(file_bytes,1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilation=cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    word_images = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        if w > 10 and h > 10:  # Ignore tiny noise/dots
            roi = img[y:y+h, x:x+w]
            # Convert back to PIL for TrOCR
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            word_images.append(roi_pil)
            
    return word_images
    
            
