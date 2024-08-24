import glob

photosList = glob.glob('photos/*.jpg')

with open("list.py", "w+") as f:
    f.write("import face_recognition\n")
    for i in photosList:
        i = i.replace("\\", "/")
        f.write(f"{i[7:-4]}_img = face_recognition.load_image_file('{i}')\n")
        f.write(f"{i[7:-4]}_encoding = face_recognition.face_encodings({i[7:-4]}_img)[0]\n\n")
        
    f.write("known_face_encodings = [\n")
    for x in photosList:
        f.write(f"\t{x[7:-4]}_encoding,\n")
    f.write("]\n")

    f.write("known_faces_name = [\n")
    for j in photosList:
        f.write(f"\t'{j[7:-4]}',\n")
    f.write("]\n")
