import cv2

video = cv2.VideoCapture(0)

while True:

    check, frame = video.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("WebCam", frame)
    key = cv2.waitKey(2)

    if key == ord("x"):
        break

video.release()
cv2.destroyAllWindows()
