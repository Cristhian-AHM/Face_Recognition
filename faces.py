import numpy as np
import cv2
import imutils
import pickle

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("faces_trained.yml")

labels = {"person name": 1}

with open("labels.pickle", 'rb') as f:
	labels = pickle.load(f)
	labels = {v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	#frame=imutils.resize(frame, width=min(100, frame.shape[1]))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1) #Scale Factor 1.1, minNeighbors 1 Full Body
	for (x,y,w,h) in faces:
		color = (255,0,0)	#BGR
		width = x+w
		height = y+h
		roi = gray[y:height, x:width]

		id_, confidence = recognizer.predict(roi)
		if confidence > 45 and confidence < 85:
			name = labels[id_]
		cv2.rectangle(frame, (x, y), (width, height), color, 3)
		cv2.rectangle(frame, (width-2, y+5), (width + 110, y - 22), color, -1)
		cv2.putText(frame, name, (x+w, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0,0,0), 2)
	cv2.imshow('Frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()