import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

captura = cv.VideoCapture(0)
configs = vision.HandLandmarkerOptions(
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task"))
maos = vision.HandLandmarker.create_from_options(configs)

largura = 640
altura = 480

temp = time.time()
tempct = 0
fps = 0

linhas = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    #linhas

    (1, 5),
    (5, 9),
    (9, 13),
    (13, 17)
]

fingers_ind = 0
indicador = False
fingers_med = 0
medio = False
fingers_anl = 0
anelar = False
fingers_min = 0
minimo = False
fingers_ded = 0
dedao = False

#--------------------------------

indVx = 0
indVy = 0

medVx = 0
medVy = 0

anlVx = 0
anlVy = 0

minVx = 0
minVy = 0

dedVx = 0
dedVy = 0

if not captura.isOpened():
    print("Erro")

while True:
    ret, frame = captura.read()
    if not ret:
        break
    frame = cv.rotate(frame, cv.ROTATE_180)

    tempct +=1
    tempAtual = time.time()

    if (tempAtual - temp) > 1:
        fps = tempct
        tempct = 0
        temp = tempAtual

    cv.putText(frame, (f"FPS: {fps}"), (10, 20), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    rgb = cv.cvtColor( frame , cv.COLOR_BGR2RGB)
    

    imagem = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data = rgb
    )

    resultado = maos.detect(imagem)

    for m in resultado.hand_landmarks:
        for i in m:
            x = round((i.x) * largura)
            y = round((i.y) * altura)
            cv.circle(frame, (x, y), 5, (0, 128, 0), -1)
        for a, b in linhas:
            l1x = round((m[a].x) * largura)
            l1y = round((m[a].y) * altura)

            l2x = round((m[b].x) * largura)
            l2y = round((m[b].y) * altura)
            cv.line(frame, (l1x, l1y), (l2x, l2y), (255, 255, 255), 1)

        indVx = (m[6].x - m[5].x)
        indVy = (m[6].x - m[5].x)

    fingers_ind = 1 if indicador else 0
    fingers_med = 1 if medio else 0
    fingers_anl = 1 if anelar else 0
    fingers_min = 1 if minimo else 0
    fingers_ded = 1 if dedao else 0



    cv.putText(frame, (f"Dedos: {(fingers_ind + fingers_med + fingers_anl + fingers_min + fingers_ded)}"), (80, 20), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

    cv.imshow("Video", frame)
    if (cv.waitKey(1) & 0xFF) == ord("e"):
        break


captura.release()
cv.destroyAllWindows()

        
