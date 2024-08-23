import face_recognition
import cv2
import numpy
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

asian_man_img = face_recognition.load_image_file("photos/asian_man_1.jpg")
asian_man_encoding = face_recognition.face_encodings(asian_man_img)[0]

black_man_img = face_recognition.load_image_file("photos/black_man_1.jpg")
black_man_encoding = face_recognition.face_encodings(black_man_img)[0]

chris_img = face_recognition.load_image_file("photos/chris_hemsworth.jpg")
chris_encoding = face_recognition.face_encodings(chris_img)[0]

chirayu_img = face_recognition.load_image_file("photos/chirayu.jpg")
chirayu_encoding = face_recognition.face_encodings(chirayu_img)[0]

known_face_encodings = [
    asian_man_encoding,
    black_man_encoding,
    chris_encoding,
    chirayu_encoding
]

known_faces_name = [
    "Asian Guy",
    "Black Guy",
    "Chris Hemsworth",
    "Chirayu Shah"
]

students = known_faces_name.copy()

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
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = numpy.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_name[best_match_index]

            face_names.append(name)
            if name in known_faces_name:
                if name in students:
                    students.remove(name)
                    print(students)
                    time = datetime.now()
                    current_time = time.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()