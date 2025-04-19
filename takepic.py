import cv2
import os
import time
import csv

cap = cv2.VideoCapture(0)

output_folder = './data/at/j'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Input informasi user
user_id = input("Masukkan ID user: ")
gender = input("Masukkan gender (M/F): ")
age = input("Masukkan usia: ")

# Prepare CSV
csv_filename = os.path.join(output_folder, 'features.csv')
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['filename', 'ID', 'gender', 'age', 'x', 'y', 'w', 'h', 'timestamp'])

face_cascade = cv2.CascadeClassifier('C:\\Users\\herdi\\Documents\\Proyek ML\\Haar Cascade\\haarcascade_frontalface_default.xml')

start_time = time.time()
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

        face_filename = '%s/%s_%d.pgm' % (output_folder, user_id, count)
        cv2.imwrite(face_filename, face_img)

        timestamp = time.time() - start_time
        csv_writer.writerow([face_filename, user_id, gender, age, x, y, w, h, timestamp])

        count += 1

    cv2.imshow('Face Capture', frame)

    if time.time() - start_time > 30:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
