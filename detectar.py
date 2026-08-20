import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math

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
indV1x = 0
indV1y = 0
indV1tam = 0

indV2x = 0
indV2y = 0
indV2tam = 0

indV3x = 0
indV3y = 0
indV3tam = 0

indprodV1_2 = 0
indprodV2_3 = 0

indV1_2cos = 0
indV2_3cos = 0

indAng1 = 0
indAng2 = 0
#--------------------------------

medV1x = 0
medV1y = 0
medV2tam = 0 

medV2x = 0
medV2y = 0
medV2tam = 0

medV3x = 0
medV3y = 0
medV3tam = 0

medprodV1_2 = 0
medprocV2_3 = 0

medV1_2cos = 0
medV2_3cos = 0

medAng1 = 0
medAng2 = 0
#--------------------------------

anlV1x = 0
anlV1y = 0
anlV1tam = 0

anlV2x = 0
anlV2y = 0
anlV2tam = 0

anlV3x = 0
anlV3y = 0
anlV3tam = 0

anlprodV1_2 = 0
anlprodV2_3 = 0

anlV1_2cos = 0
anlV2_3cos = 0

anlAng1 = 0
anlAng2 = 0
#--------------------------------

minV1x = 0
minV1y = 0
minV1tam = 0

minV2x = 0
minV2y = 0
minV2tam = 0

minV3x = 0
minV3y = 0
minV3tam = 0

minprodV1_2 = 0
minprodV2_3 = 0

minV1_2cos = 0
minV2_3cos = 0

minAng1 = 0
minAng2 = 0
#--------------------------------

dedV1x = 0
dedV1y = 0
dedV1tam = 0

dedV2x = 0
dedV2y = 0
dedV2tam = 0

dedV3x = 0
dedV3y = 0
dedV3tam = 0

distded = 0
#--------------------------------

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

#----------------------------Indicador----------------------------

        indV1x = (m[6].x - m[5].x)
        indV1y = (m[6].y - m[5].y)
        indV2x = (m[7].x - m[6].x)
        indV2y = (m[7].y - m[6].y)
        indV3x = (m[8].x - m[7].x)
        indV3y = (m[8].y - m[7].y)

        indprodV1_2 = (indV1x * indV2x) + (indV1y * indV2y)
        indprodV2_3 = (indV2x * indV3x) + (indV2y * indV3y)
        indV1tam = math.sqrt((indV1x * indV1x) + (indV1y * indV1y))
        indV2tam = math.sqrt((indV2x * indV2x) + (indV2y * indV2y))
        indV3tam = math.sqrt((indV3x * indV3x) + (indV3y * indV3y))
        indV1_2cos = (indprodV1_2 / (indV1tam * indV2tam))
        indV2_3cos = (indprodV2_3 / (indV2tam * indV3tam))
        indAng1 = math.degrees(math.acos(indV1_2cos))
        indAng2 = math.degrees(math.acos(indV2_3cos))

        if indAng1 < 90 and indAng2 < 90:
            indicador = True
        else:
            indicador = False

#----------------------------Medio----------------------------

    
        medV1x = (m[10].x - m[9].x)
        medV1y = (m[10].y - m[9].y)
        medV2x = (m[11].x - m[10].x)
        medV2y = (m[11].y - m[10].y)
        medV3x = (m[12].x - m[11].x)
        medV3y = (m[12].y - m[11].y)

        medprodV1_2 = (medV1x * medV2x) + (medV1y * medV2y)
        medprodV2_3 = (medV2x * medV3x) + (medV2y * medV3y)
        medV1tam = math.sqrt((medV1x * medV1x) + (medV1y * medV1y))
        medV2tam = math.sqrt((medV2x * medV2x) + (medV2y * medV2y))
        medV3tam = math.sqrt((medV3x * medV3x) + (medV3y * medV3y))
        medV1_2cos = (medprodV1_2 / (medV1tam * medV2tam))
        medV2_3cos = (medprodV2_3 / (medV2tam * medV3tam))
        medAng1 = math.degrees(math.acos(medV1_2cos))
        medAng2 = math.degrees(math.acos(medV2_3cos))

        if medAng1 < 90 and medAng2 < 90:
            medio = True
        else:
            medio = False
#----------------------------Anelas----------------------------

        anlV1x = (m[14].x - m[13].x)
        anlV1y = (m[14].y - m[13].y)
        anlV2x = (m[15].x - m[14].x)
        anlV2y = (m[15].y - m[14].y)
        anlV3x = (m[16].x - m[15].x)
        anlV3y = (m[16].y - m[15].y)

        anlprodV1_2 = (anlV1x * anlV2x) + (anlV1y * anlV2y)
        anlprodV2_3 = (anlV2x * anlV3x) + (anlV2y * anlV3y)
        anlV1tam = math.sqrt((anlV1x * anlV1x) + (anlV1y * anlV1y))
        anlV2tam = math.sqrt((anlV2x * anlV2x) + (anlV2y * anlV2y))
        anlV3tam = math.sqrt((anlV3x * anlV3x) + (anlV3y * anlV3y))
        anlV1_2cos = (anlprodV1_2 / (anlV1tam * anlV2tam))
        anlV2_3cos = (anlprodV2_3 / (anlV2tam * anlV3tam))
        anlAng1 = math.degrees(math.acos(anlV1_2cos))
        anlAng2 = math.degrees(math.acos(anlV2_3cos))

        if anlAng1 < 90 and anlAng2 < 90:
            anelar = True
        else:
            anelar = False

#----------------------------Anelas----------------------------

        minV1x = (m[10].x - m[9].x)
        minV1y = (m[10].y - m[9].y)
        minV2x = (m[11].x - m[10].x)
        minV2y = (m[11].y - m[10].y)
        minV3x = (m[12].x - m[11].x)
        minV3y = (m[12].y - m[11].y)

        minprodV1_2 = (minV1x * minV2x) + (minV1y * minV2y)
        minprodV2_3 = (minV2x * minV3x) + (minV2y * minV3y)
        minV1tam = math.sqrt((minV1x * minV1x) + (minV1y * minV1y))
        minV2tam = math.sqrt((minV2x * minV2x) + (minV2y * minV2y))
        minV3tam = math.sqrt((minV3x * minV3x) + (minV3y * minV3y))
        minV1_2cos = (minprodV1_2 / (minV1tam * minV2tam))
        minV2_3cos = (minprodV2_3 / (minV2tam * minV3tam))
        minAng1 = math.degrees(math.acos(minV1_2cos))
        minAng2 = math.degrees(math.acos(minV2_3cos))

        if minAng1 < 90 and minAng2 < 90:
            minimo = True
        else:
            minimo = False
 #----------------------------Anelas----------------------------      

        distded = math.sqrt(
            (m[4].x - m[0].x) ** 2 +
            (m[4].y - m[0].y) ** 2
        )

    fingers_ind = 1 if indicador else 0
    fingers_med = 1 if medio else 0
    fingers_anl = 1 if anelar else 0
    fingers_min = 1 if minimo else 0
    fingers_ded = 1 if distded > 0.20 else 0



    cv.putText(frame, (f"Dedos: {(fingers_ind + fingers_med + fingers_anl + fingers_min + fingers_ded)}"), (80, 20), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

    cv.imshow("Video", frame)
    if (cv.waitKey(1) & 0xFF) == ord("e"):
        break


captura.release()
cv.destroyAllWindows()

        
