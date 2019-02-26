#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import os

arqCasc1 = 'haarcascade_frontalface_default.xml'
arqCasc2 = 'haarcascade_eye.xml'
arqsmile=  'haarcascade_smile.xml'
faceCascade1 = cv2.CascadeClassifier(arqCasc1)  # classificador para o rosto
faceCascade2 = cv2.CascadeClassifier(arqCasc2)  # classificador para os olhos
smileCascade = cv2.CascadeClassifier(arqsmile)  # Classificador para "Sorrir"
font = cv2.FONT_HERSHEY_SIMPLEX # fonte de escrita	
webcam = cv2.VideoCapture(0)  # instancia o uso da webcam
#sift=cv2.xfeatures2d.SIFT_create()

while True:
    s, imagem = webcam.read()  # pega efetiVamente a imagem da webcam
    imagem = cv2.flip(imagem, 180)  # espelha a imagem
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #conversão de imagem para escala cinza
    #kp=sift.detect(gray, None)
    #img=cv2.drawKeypoints(gray, kp, imagem, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    faces = faceCascade1.detectMultiScale(
        gray,
        minNeighbors=8,
        minSize=(30, 30),
        maxSize=(500, 500),
    )

    olhos = faceCascade2.detectMultiScale(
        gray,
        minNeighbors=20,
        minSize=(60, 60),
        maxSize=(90, 90),
    )

    sorriso= smileCascade.detectMultiScale(
        gray,
        scaleFactor=5.1,
        minNeighbors=10,
        minSize=(20,20),
        flags=0
    )

    # Desenha um retangulo nas faces e olhos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(imagem, 'Face', (x + w, y + h), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in olhos:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(imagem, 'Olhos', (x + w, y + h), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in sorriso:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (10, 5, 10), 2)
        cv2.putText(imagem, 'Sorrindo', (x + w, y + h), font, 0.5, (10, 5, 10), 2, cv2.LINE_AA)

    cv2.imshow('Video em tempo real', imagem)  # mostra a imagem captura na janela

    # o trecho seguinte e apenas para parar o codigo e fechar a janela
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()  # dispensa o uso da webcam
cv2.destroyAllWindows()  # fecha todas a janelas abertas
