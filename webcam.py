import numpy as np
import face_recognition as fr
import cv2
from engine import get_rostos

db_rostos_conhecido, db_nomes_rostos = get_rostos()

cap_video = cv2.VideoCapture(0)
while True:
    ret, frame = cap_video.read()

    rgb_frame = frame[:, :, ::-1]

    face_localizacao = fr.face_locations(rgb_frame)
    rosto_desconhecido = fr.face_encodings(rgb_frame, face_localizacao)

    if(len(rosto_desconhecido) < 1):
        print('Nenhum rosto localizado!')
        continue

    for (top, rigth, bottom, left), rosto_desconhecido in zip(face_localizacao, rosto_desconhecido):
        resultados =fr.compare_faces(db_rostos_conhecido, rosto_desconhecido)
        print(resultados)

        face_distancia = fr.face_distance(db_rostos_conhecido, rosto_desconhecido)        
        melhor_distantacia_id = np.argmin(face_distancia)

        if resultados[melhor_distantacia_id]:
            nome = db_nomes_rostos[melhor_distantacia_id]
        else:
            nome = 'Desconhecido'
        
        # Marcação em torno do rosto
        cv2.rectangle(frame, (left, top), (rigth, bottom), (0, 0, 255), 2)

        # Marcação da base do rosto
        cv2.rectangle(frame, (left, bottom+35), (rigth, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Inserção de texto
        cv2.putText(frame, nome, (left+6, bottom+20), font, 0.5, (255,255,255), 1)

        cv2.imshow('Webcam_facerecognition', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_video.release()
cv2.destroyAllWindows()
