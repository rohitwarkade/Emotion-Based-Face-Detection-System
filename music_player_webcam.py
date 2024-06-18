
import time
import cv2
import label_image
import os
import random
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

size = 4
# Load the XML file for face detection
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
global text
webcam = cv2.VideoCapture(0)  # Using the default webcam connected to the PC.

now = time.time()  # For calculating seconds of video
future = now + 60  # Set the time (in seconds) for emotion recognition; you can change it

while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 0)  # Flip to act as a mirror

    # Resize the image to speed up face detection
    mini = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))

    # Detect faces
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shape size back up
        sub_face = im[y:y + h, x:x + w]
        FaceFileName = "test.jpg"  # Saving the current image from the webcam for testing
        cv2.imwrite(FaceFileName, sub_face)
        text = label_image.main(FaceFileName)  # Get the result from the label_image file
        text = text.title()  # Capitalize the recognized emotion
        font = cv2.FONT_HERSHEY_TRIPLEX

        if text == 'Angry':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0, 25, 255), 2)

        if text == 'Smile':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0, 260, 0), 2)

        if text == 'Fear':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0, 255, 255), 2)

        if text == 'Sad':
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 191, 255), 7)
            cv2.putText(im, text, (x + h, y), font, 1, (0, 191, 255), 2)

    # Show the image
    base_path = r'C:\Users\yashs\Desktop\Music\Music_player\songs'
    cv2.imshow('Music player with Emotion recognition', im)
    key = cv2.waitKey(30) & 0xff

    if time.time() > future:
        try:
            cv2.destroyAllWindows()  # Close the OpenCV window
            mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'

            if text == 'Angry':
                randomfile = random.choice(os.listdir(os.path.join(base_path, 'Angry')))
                print(f'You are angry! Please calm down, I will play a song for you: {randomfile}')
                file = os.path.join(base_path, 'Angry', randomfile)
                subprocess.call([mp, file])

            if text == 'Happy':
                randomfile = random.choice(os.listdir(os.path.join(base_path, 'Happy')))
                print(f'You are smiling! I am playing a special song for you: {randomfile}')
                file = os.path.join(base_path, 'Happy', randomfile)
                subprocess.call([mp, file])

            if text == 'Fear':
                randomfile = random.choice(os.listdir(os.path.join(base_path, 'Fear')))
                print(f'You have fear of something. I am playing a song for you: {randomfile}')
                file = os.path.join(base_path, 'Fear', randomfile)
                subprocess.call([mp, file])

            if text == 'Sad':
                randomfile = random.choice(os.listdir(os.path.join(base_path, 'Sad')))
                print(f'You are sad, dont worry! I am playing a song for you: {randomfile}')
                file = os.path.join(base_path, 'Sad', randomfile)
                subprocess.call([mp, file])
            break

        except Exception as e:
            print(f'An error occurred: {e}')
            print('Please stay focused in the camera frame for at least 15 seconds and run the program again')
            break

    if key == 27:  # Press the Esc key to exit
        break
