import cv2
import os
import numpy as np

def entrenar_modelo(folder_name):
    data_path = 'C:/Cursos/Phyton/SistemaReconocimientoFacial/Data/' + folder_name
    if not os.path.exists(data_path):
        print("La carpeta especificada no existe.")
        return

    labels = []
    faces_data = []
    label = 0

    for file_name in os.listdir(data_path):
        print('Leyendo imágenes de', folder_name)
        print('Rostros:', folder_name + '/' + file_name)
        labels.append(label)

        image_path = os.path.join(data_path, file_name)
        faces_data.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print('Entrenando...')
    face_recognizer.train(faces_data, np.array(labels))

    model_path = 'C:/Cursos/Phyton/SistemaReconocimientoFacial/Entrenamiento/Modelo_' + folder_name + '.xml'
    face_recognizer.write(model_path)
    print('Modelo guardado en', model_path)

# Llamar a la función para entrenar el modelo con una carpeta específica
if __name__ == "__main__":
    carpeta_entrenamiento = "WhiteMoi"
    entrenar_modelo(carpeta_entrenamiento)
