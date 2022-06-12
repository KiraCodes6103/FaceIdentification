import cv2

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret == False:
        continue

    cv2.imshow("web cam", frame)
    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

