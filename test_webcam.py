import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not found. Check connection.")
else:
    print("✅ Webcam detected successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(frame, "Webcam Working! Press Q to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam test complete")