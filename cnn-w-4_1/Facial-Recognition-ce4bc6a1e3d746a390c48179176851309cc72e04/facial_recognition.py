#!/usr/bin/env python3
import cv2
from fr_utils import *
from neural_network import load_database, who_is_it

create_notification = False
try:   # only create notification if module present
    from plyer import notification
    create_notification = True
except ModuleNotFoundError:
    pass

"""
Starts the web camera and detects faces
Once a face is detected the relevant portion of the image is passed for processing
where it is compared with other stored image encodings.
If the distance between the scanned image and a database image is less than the set threshold
a match is made and the name will be shown on screen with the percentage accuracy
"""

detecting = False  # flag to ensure we don't keep detecting while processing previous input
font = cv2.FONT_HERSHEY_DUPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (0, 0, 0)
lineType = 2

if __name__ == "__main__":
    database = load_database()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if create_notification:
        notification.notify(
            title="Facial Recognition Started",
            message="The Facial Recognition System has started",
            app_name="Facial Recognition",
        )

    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        x1 = y1 = x2 = y2 = 0
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            x1, y1, x2, y2 = x, y, x+w, y+h

        if not detecting:
            if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
                detecting = True
                who = who_is_it(img[y1:y2, x1:x2], database)  # compare face detected portion of image with database encodings
                if who["identity"] is not None:  # only show if match found
                    accuracy = 100 - (who['distance'] * 100)
                    result = f"{who['identity']} {accuracy:.2f}%"
                    h, w, c = img.shape
                    x1, y1 = int(w/3-200), int(h/3-100)
                    # print(f"Found {result}")
                    cv2.putText(img, result,
                                (x1, y1),
                                font,
                                fontScale,
                                fontColor,
                                lineType)
                detecting = False
        cv2.imshow('Facial Recognition', img)
        if cv2.waitKey(30) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
