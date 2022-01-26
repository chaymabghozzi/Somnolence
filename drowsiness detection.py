
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2

#Initialiser Pygame et charger la musique
pygame.mixer.init()
pygame.mixer.music.load('alarm.mp3')

#Seuil minimal du rapport d'aspect de l'œil en dessous duquel l'alarme est déclenchée.
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Nombre minimum d'images consécutives pour lesquelles le ratio oculaire est inférieur au seuil de déclenchement de l'alarme.
EYE_ASPECT_RATIO_CONSEC_FRAMES = 20

#Counts nombre de trames consécutives en dessous de la valeur seuil
COUNTER = 0

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#calcule et renvoie le rapport d'aspect de l'œil
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

#rapport d'aspect de l'œil
    ear = (A+B) / (2*C)
    return ear

#Chargement du détecteur et du prédicteur de visage, utilise le fichier prédicteur de forme de la dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_70_face_landmarks.dat")

#Extraire des indices de repères faciaux pour l'œil gauche et l'œil droit
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

video_capture = cv2.VideoCapture('vidéo_somnolance.mp4')

#Give some time for camera to initialize(not required)
time.sleep(2)

while(True):
    #Lire chaque image, la retourner et la convertir en niveaux de gris.
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Détecter les points du visage
    faces = detector(gray, 0)

    #Detecter le visage haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Dessinez un rectangle autour de chaque face détectée
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Détecter les points du visage
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Tableau des coordonnées de l'œil gauche et de l'œil droit.
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculer le rapport d'aspect des deux yeux
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #supprimer les divergences du contour convexe et dessiner le contour des yeux.
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Détecter si le rapport hauteur / largeur des yeux est inférieur au seuil
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "ALERTE DE SOMMEIL!!!!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        else:
            
            pygame.mixer.music.stop()
            COUNTER = 0

    #L'affichage
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()