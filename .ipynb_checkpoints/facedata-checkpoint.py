import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

name = input("Enter name: ")
face_data = []
count = 0
dataset_path = './data/'

# Create dataset folder if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Image', frame)

    if len(faces) == 0:
        continue

    # Select the largest face
    face = sorted(faces, key=lambda f: f[2] * f[3])[-1]
    x, y, w, h = face
    offset = 10
    face_img = frame[y - offset:y + h + offset, x - offset:x + w + offset]
    face_img = cv2.resize(face_img, (100, 100))

    if count % 10 == 0:
        face_data.append(face_img)
    count += 1

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

face_data = np.array(face_data)
face_data = face_data.reshape(len(face_data), -1)

print(face_data.shape)
np.save(dataset_path + name + '.npy', face_data)
print('Data saved successfully at ' + dataset_path + name + '.npy')

cap.release()
cv2.destroyAllWindows()
