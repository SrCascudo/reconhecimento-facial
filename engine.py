import face_recognition as fr

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)

    if(len(rostos) > 0):
        return True, rostos
    
    return False, []

def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []

    john_base = reconhece_face('./img/john_base.jpg')
    if(john_base[0]):
        rostos_conhecidos.append(john_base[1][0])
        nomes_dos_rostos.append("John Helder (Adulto)")

    john_kids = reconhece_face('./img/john_kids.jpg')
    if(john_kids[0]):
        rostos_conhecidos.append(john_kids[1][0])
        nomes_dos_rostos.append("John Helder (Criança)")


    stark = reconhece_face('./img/stark.jpg')
    if(stark[0]):
        rostos_conhecidos.append(stark[1][0])
        nomes_dos_rostos.append("Tony Stark")
    
    return rostos_conhecidos, nomes_dos_rostos
