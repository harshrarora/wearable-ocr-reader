import cv2
import easyocr

print("Loading image...")

image = cv2.imread("book.jpeg")

# Resize large phone images so OCR runs faster
image = cv2.resize(image, None, fx=0.5, fy=0.5)

print("Initialising OCR...")
reader = easyocr.Reader(['en'])

print("Running OCR...")

results = reader.readtext(image)

print("\nDetected text:\n")

for bbox, text, conf in results:
    print(f"{text}  (confidence {conf:.2f})")