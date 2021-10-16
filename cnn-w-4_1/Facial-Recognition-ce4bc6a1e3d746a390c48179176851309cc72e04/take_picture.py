#!/usr/bin/env python3
import os
from datetime import datetime
import cv2

"""
Take a picture of a person's face
Opens the webcam and tries to detect a face
Once a face is detected a bounding box will appear over the detected face(s)
Hitting enter will take a picture and try save it to the images folder
Hitting Esc exits from the webcam
Image resolution will depend on your web-camera
The image saved will be of the cropped face only
"""

MODULE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))  # location of this module (take_picture.py)
SAVE_PATH = os.path.join(MODULE_DIR_PATH, "images")  # where to save images taken
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # haarcascade to detect face

    webcam = cv2.VideoCapture(0)

    while True:
        ret_val, img = webcam.read()
        img = cv2.flip(img, 1)  # flip image to ensure it is easier to use

        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # used for face detect
        faces = face_cascade.detectMultiScale(grayscale_img, 1.3, 5)  # get list of b-boxes of detected faces
        x1 = y1 = x2 = y2 = 0   # initialize bounding box co-ords to 0, x1,y1 = top left, x2,y2 = bottom right
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)  # draw black bounding box
            x1, y1, x2, y2 = x, y, x+w, y+h  # update variables for cropping later

        cv2.imshow("Capture Face Picture", img)

        key_pressed = cv2.waitKey(30)
        if key_pressed == 27:  # Esc pressed (might vary across different hardware/os)
            break
        elif key_pressed == 13:  # Enter pressed (might vary across different hardware/os)
            if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:  # no faces detected don't take picture
                file_name = datetime.now().strftime('%Y-%m-%d %H-%M-%S.png')
                file_path = os.path.join(SAVE_PATH, file_name)
                crop_img = img[y1:y2, x1:x2]  # only use face region detected
                try:
                    cv2.imwrite(file_path, crop_img)
                    print(f"Saved {file_path}")
                except IOError as io_error:
                    print(f"Error saving image: {io_error}")
    cv2.destroyAllWindows()


