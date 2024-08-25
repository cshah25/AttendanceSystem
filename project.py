import face_recognition
import cv2
import numpy
import csv
import os
from datetime import datetime
# from list import known_face_encodings
# from list import known_faces_name

video_capture = cv2.VideoCapture(0)

photos_dir = 'photos/'
files_in_directory = os.listdir(photos_dir)

image_files_in_directory = [file for file in files_in_directory if file.endswith(".jpg") or file.endswith(".png")]

face_encoding_dict = []
face_names = []

for img_file in image_files_in_directory:
    image = face_recognition.load_image_file(os.path.join(photos_dir, img_file))
    img_encoding = face_recognition.face_encodings(image)
    if len(img_encoding) == 0:
        print(f'No faces found in the {img_file}')
    else:
        face_encoding_dict.append(img_encoding)
        face_names.append(img_file)

students = face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv', "w+", newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = numpy.ascontiguousarray(small_frame[:, :, ::-1])
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for img_encoding in face_encodings:
            matches = face_recognition.compare_faces(face_encoding_dict, img_encoding)
            name = ""
            face_distance = face_recognition.face_distance(face_encoding_dict, img_encoding)
            best_match_index = numpy.argmin(face_distance)
            if matches[best_match_index]:
                name = face_names[best_match_index]

            face_names.append(name)
            if name in face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    time = datetime.now()
                    current_time = time.strftime("%H:%M:%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()