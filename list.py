import face_recognition
asian_man_img = face_recognition.load_image_file('photos/asian_man.jpg')
asian_man_encoding = face_recognition.face_encodings(asian_man_img)[0]

black_man_img = face_recognition.load_image_file('photos/black_man.jpg')
black_man_encoding = face_recognition.face_encodings(black_man_img)[0]

chirayu_img = face_recognition.load_image_file('photos/chirayu.jpg')
chirayu_encoding = face_recognition.face_encodings(chirayu_img)[0]

chris_hemsworth_img = face_recognition.load_image_file('photos/chris_hemsworth.jpg')
chris_hemsworth_encoding = face_recognition.face_encodings(chris_hemsworth_img)[0]

old_white_man_img = face_recognition.load_image_file('photos/old_white_man.jpg')
old_white_man_encoding = face_recognition.face_encodings(old_white_man_img)[0]

known_face_encodings = [
	asian_man_encoding,
	black_man_encoding,
	chirayu_encoding,
	chris_hemsworth_encoding,
	old_white_man_encoding,
]
known_faces_name = [
	'asian_man',
	'black_man',
	'chirayu',
	'chris_hemsworth',
	'old_white_man',
]
