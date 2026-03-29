import cv2
import os

# Create directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

cap = cv2.VideoCapture(0) # 1 is the second webcam
count = 0

print("Press 's' to save an image, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Logitech Brio 100 Calibration', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        img_name = f"images/calib_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
