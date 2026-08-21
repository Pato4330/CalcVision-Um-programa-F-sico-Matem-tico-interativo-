import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import numpy as np

captura = cv.VideoCapture(0, cv.CAP_MSMF)

captura.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
captura.set(cv.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
captura.set(cv.CAP_PROP_FPS, 60)

print("abriu:", captura.isOpened())
print("largura:", captura.get(cv.CAP_PROP_FRAME_WIDTH))
print("altura:", captura.get(cv.CAP_PROP_FRAME_HEIGHT))
print("fps:", captura.get(cv.CAP_PROP_FPS))

resultado = None

def resultado_maos(res, imagem, timestamp):
    global resultado
    resultado = res

configs = vision.HandLandmarkerOptions(
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=resultado_maos
    )

maos = vision.HandLandmarker.create_from_options(configs)

temp = time.time()
tempct = 0
fps = 0

largura = 640
altura = 480
debug = False
debug1 = False
rotateH = False

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
fingers_med = 0
fingers_anl = 0
fingers_min = 0
fingers_ded = 0

#--------------------------------
indV1x = 0
indV1y = 0

indV2x = 0
indV2y = 0

indV3x = 0
indV3y = 0

indprodV1_2 = 0
indprodV2_3 = 0
#--------------------------------

medV1x = 0
medV1y = 0

medV2x = 0
medV2y = 0

medV3x = 0
medV3y = 0

medprodV1_2 = 0
medprocV2_3 = 0

#--------------------------------

anlV1x = 0
anlV1y = 0

anlV2x = 0
anlV2y = 0

anlV3x = 0
anlV3y = 0

anlprodV1_2 = 0
anlprodV2_3 = 0

#--------------------------------

minV1x = 0
minV1y = 0

minV2x = 0
minV2y = 0

minV3x = 0
minV3y = 0

minprodV1_2 = 0
minprodV2_3 = 0

#--------------------------------

dedV1x = 0
dedV1y = 0

dedV2x = 0
dedV2y = 0

dedV3x = 0
dedV3y = 0

distded = 0
#--------------------------------
polindDistance = [0.0, 0.0]
clicando = [False, False]

le = [[[0.0, 0.0] for _ in range(21)] for _ in range(2)]
tenho = [False, False]
alpha = 0.80


if not captura.isOpened():
    print("Erro")

while True:
    indicador = [False, False]
    medio = [False, False]
    anelar = [False, False]
    minimo = [False, False]
    distded = [0.0, 0.0]
    tammao = [0.0, 0.0]

    ret, frame = captura.read()
    if not ret:
        break
    if rotateH:
        frame = cv.rotate(frame, cv.ROTATE_180)


    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    
    imagem = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data = rgb
    )
    timestamp = int(time.time() * 1000)
    maos.detect_async(imagem, timestamp)

    
    if resultado is not None:
        for ind, m in enumerate(resultado.hand_landmarks):
            if not tenho[ind]:
                for i in range(21):
                    le[ind][i][0] = m[i].x
                    le[ind][i][1] = m[i].y
                    tenho[ind] = True
            else:
                for i in range(21):
                    le[ind][i][0] = alpha * m[i].x + (1 - alpha) * le[ind][i][0]
                    le[ind][i][1] = alpha * m[i].y + (1 - alpha) * le[ind][i][1]


            for i in range(21):
                x = int(le[ind][i][0] * largura)
                y = int(le[ind][i][1] * altura)
                if debug:
                    cv.circle(frame, (x, y), 5, (0, 128, 0), -1)

            for a, b in linhas:
                l1x = int(le[ind][a][0] * largura)
                l1y = int(le[ind][a][1] * altura)

                l2x = int(le[ind][b][0] * largura)
                l2y = int(le[ind][b][1] * altura)
                if debug:
                    cv.line(frame, (l1x, l1y), (l2x, l2y), (255, 255, 255), 1)

    #----------------------------Indicador----------------------------

            indV1x = (le[ind][6][0] - le[ind][5][0])
            indV1y = (le[ind][6][1] - le[ind][5][1])
            indV2x = (le[ind][7][0] - le[ind][6][0])
            indV2y = (le[ind][7][1] - le[ind][6][1])
            indV3x = (le[ind][8][0] - le[ind][7][0])
            indV3y = (le[ind][8][1] - le[ind][7][1])

            indprodV1_2 = (indV1x * indV2x) + (indV1y * indV2y)
            indprodV2_3 = (indV2x * indV3x) + (indV2y * indV3y)

            if indprodV1_2 > 0 and indprodV2_3 > 0:
                indicador[ind] = True
            else:
                indicador[ind] = False

    #----------------------------Medio----------------------------

        
            medV1x = (le[ind][10][0] - le[ind][9][0])
            medV1y = (le[ind][10][1] - le[ind][9][1])
            medV2x = (le[ind][11][0] - le[ind][10][0])
            medV2y = (le[ind][11][1] - le[ind][10][1])
            medV3x = (le[ind][12][0] - le[ind][11][0])
            medV3y = (le[ind][12][1] - le[ind][11][1])

            medprodV1_2 = (medV1x * medV2x) + (medV1y * medV2y)
            medprodV2_3 = (medV2x * medV3x) + (medV2y * medV3y)

            if medprodV1_2 > 0 and medprodV2_3 > 0:
                medio[ind] = True
            else:
                medio[ind] = False
    #----------------------------Anelas----------------------------

            anlV1x = (le[ind][14][0] - le[ind][13][0])
            anlV1y = (le[ind][14][1] - le[ind][13][1])
            anlV2x = (le[ind][15][0] - le[ind][14][0])
            anlV2y = (le[ind][15][1] - le[ind][14][1])
            anlV3x = (le[ind][16][0] - le[ind][15][0])
            anlV3y = (le[ind][16][1] - le[ind][15][1])

            anlprodV1_2 = (anlV1x * anlV2x) + (anlV1y * anlV2y)
            anlprodV2_3 = (anlV2x * anlV3x) + (anlV2y * anlV3y)

            if anlprodV1_2 > 0 and anlprodV2_3 > 0:
                anelar[ind] = True
            else:
                anelar[ind] = False
    #----------------------------Minimo----------------------------

            minV1x = (le[ind][18][0] - le[ind][17][0])
            minV1y = (le[ind][18][1] - le[ind][17][1])
            minV2x = (le[ind][19][0] - le[ind][18][0])
            minV2y = (le[ind][19][1] - le[ind][18][1])
            minV3x = (le[ind][20][0] - le[ind][19][0])
            minV3y = (le[ind][20][1] - le[ind][19][1])

            minprodV1_2 = (minV1x * minV2x) + (minV1y * minV2y)
            minprodV2_3 = (minV2x * minV3x) + (minV2y * minV3y)

            if minprodV1_2 > 0 and minprodV2_3 > 0:
                minimo[ind] = True
            else:
                minimo[ind] = False
    #----------------------------Polegar----------------------------      

            distded[ind] = math.sqrt(
                (le[ind][4][0] - le[ind][0][0]) ** 2 +
                (le[ind][4][1] - le[ind][0][1]) ** 2
            )

            tammao[ind] = math.sqrt(
            (le[ind][9][0] - le[ind][0][0]) ** 2 +
            (le[ind][9][1] - le[ind][0][1]) ** 2
            )

            distded[ind] = distded[ind]/tammao[ind]

            if debug == False:
                cv.circle(frame, (x, y), 5, (0, 0, 128), -1)

            if indicador[ind] == True:
                x = int(le[ind][8][0] * largura)
                y = int(le[ind][8][1] * altura)
                

            polindDistance[ind] = math.sqrt(
                (le[ind][4][0] - le[ind][8][0]) ** 2 +
                (le[ind][4][1] - le[ind][8][1]) ** 2)

            polindDistance[ind] = polindDistance[ind]/tammao[ind]

            if polindDistance[ind] < 0.25 and clicando[ind] == False:
                clicando[ind] = True
                
            elif polindDistance[ind] > 0.3:
                clicando[ind] = False
        
                if debug1:
                    x = int(le[ind][4][0] * largura)
                    y = int(le[ind][4][1] * altura)
                    x1 = int(le[ind][8][0] * largura)
                    y1 = int(le[ind][8][1] * altura)
                    cv.line(frame, (x, y), (x1, y1), (255, 255, 255), 1)
                    cv.putText(frame, (f"{polindDistance[ind]:.2f}"), ((x - 30), (y - 30)), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)



    
    fingers_ind = sum(1 for x in indicador if x)
    fingers_med = sum(1 for x in medio if x)
    fingers_anl = sum(1 for x in anelar if x)
    fingers_min = sum(1 for x in minimo if x)
    fingers_ded = sum(1 for x in distded if x > 1)

    tempct +=1
    tempAtual = time.time()
    
    if (tempAtual - temp) > 1:
        fps = tempct
        tempct = 0
        temp = tempAtual

    if debug:
        cv.putText(frame, (f"Dedos: {(fingers_ind + fingers_med + fingers_anl + fingers_min + fingers_ded)}"), (10, 40), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv.putText(frame, (f"FPS: {fps}"), (10, 20), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv.putText(frame, (f"Alpha: {alpha:.2f}"), (10, 60), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

    key = (cv.waitKey(1) & 0xFF)

    cv.imshow("Video", frame)

    if key == ord('d'):
        debug = not debug
    elif key == ord('e'):
        break
    elif key == ord('r'):
        rotateH = not rotateH
    elif key == ord('+'):
        alpha = min(alpha + 0.01, 1.0)
    elif key == ord('-'):
        alpha = max(alpha - 0.01, 0.0)
    elif key == ord('1'):
        debug1 = not debug1

    



captura.release()
cv.destroyAllWindows()

        
