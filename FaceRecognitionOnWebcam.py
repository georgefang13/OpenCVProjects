# help from https://www.datacamp.com/tutorial/face-detection-python-opencv
import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# access webcam
video_capture = cv2.VideoCapture(0)  # 0 tells it to use default camera


# identify face
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


# Creating a Loop for Real-Time Face Detection
while True:
    result, video = video_capture.read()
    if result is False:
        break
    the_faces = detect_bounding_box(video)
    cv2.imshow('Face Detector', video)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
