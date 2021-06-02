import cv2 as cv

face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
smile = cv.CascadeClassifier('haarcascade_smile.xml')

cam = cv.VideoCapture(0)

while True:
    _,frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile.detectMultiScale(roi_gray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(roi_color, (sx, sy), ((sx + w), (sy + h)), (255, 0, 0), 2)

    cv.imshow("VIDEO", frame)
    if cv.waitKey(1) & 0xff == ord('x'):  # Exit dengan tombol x
        break

cam.release()
cv.destroyAllWindows()
cam.release()

