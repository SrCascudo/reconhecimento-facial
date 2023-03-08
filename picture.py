import face_recognition as fr
from engine import reconhece_face, get_rostos

desconhecido = reconhece_face('./img/john_teste.png')
if(desconhecido[0]):
    rosto_desconhecido = desconhecido[1][0]
    db_rostos_conhecido, db_nomes_rostos = get_rostos()
    resultados =fr.compare_faces(db_rostos_conhecido, rosto_desconhecido)
    for i, rosto in enumerate(resultados):
        if(rosto):
            print('Essa foto é igual a ' + db_nomes_rostos[i])
        else:
            print('Essa foto não é igual a ' + db_nomes_rostos[i])
else:
    print('Não foi encontrada nenhuma face!')
