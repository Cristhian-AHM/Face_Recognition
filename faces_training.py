import os
import numpy as np
import cv2 
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

image_dir = os.path.join(BASE_DIR, "faces")

recognizer = cv2.face.LBPHFaceRecognizer_create()
label_id = {}
current_id = 0

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			#print(label, path)
			if not label in label_id:
				label_id[label] = current_id
				current_id += 1

			id_ = label_id[label]
			#x_labels.append(label)
			pillow_image = Image.open(path).convert("L")
			size = (550, 550)
			final_image = pillow_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				color = (255,0,0)	#BGR
				width = x+w
				height = y+h
				roi = image_array[y:height, x:width]
				x_train.append(roi)
				y_labels.append(id_)



with open("labels.pickle", "wb") as f:
	pickle.dump(label_id, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("faces_trained.yml")