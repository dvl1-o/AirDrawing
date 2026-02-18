import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([130, 255, 255])
prev_x, prev_y = None, None
draw_color = (0, 255, 0)  # Green color for drawing
brush_size = 6

print("\n" + "*"*60)
print("BLUE AIR DRAWING CANVAS")
print("*"*60)
print("\n How to use:")
print("  1 Hold a BLUE object (cap,marker,tape on finger)")
print("  2 Move it to draw")
print("  3 Press 'c' to clear")
print("  4 Press 's' to save")
print("  5 Press 'q' to quit")
print("\nTips:")
print("Use bright blue color")
print("Good lighting helps")
print("Move slowly for smooth lines")
print("*"*60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 500:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 15, (255, 0, 255), 3)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            if prev_x is not None and prev_y is not None:
                distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if distance < 50:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_size)

            prev_x, prev_y = x, y

            cv2.putText(frame, "DRAWING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            prev_x, prev_y = None, None
            cv2.putText(frame, "OBJECT TOO SMALL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        prev_x, prev_y = None, None
        cv2.putText(frame, "NO BLUE DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
    cv2.putText(output, "C-Clear | S-Save | Q-Quit", (10, 460),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('Blue Air Drawing', output)
    cv2.imshow('Drawing Only', canvas)
    cv2.imshow('Blue Detection', mask)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n Goodbye")
        break
    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        prev_x, prev_y = None, None
        print("Canvas cleared")
    elif key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved as {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n Done ")