import cv2
import os
import numpy as np
from tkinter import messagebox

def entrenar_modelo():
    data_path = 'C:/Cursos/Phyton/SistemaReconocimientoFacial/Data'
    people_list = os.listdir(data_path)
    print('Lista de personas: ', people_list)

    labels = []
    faces_data = []
    label = 0
    for name_dir in people_list:
        person_path = os.path.join(data_path, name_dir)
        print('Leyendo imágenes de', name_dir)
        for file_name in os.listdir(person_path):
            print('Rostros:', name_dir + '/' + file_name)
            labels.append(label)

            faces_data.append(cv2.imread(os.path.join(person_path, file_name), cv2.IMREAD_GRAYSCALE))

        label += 1

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Mostrar ventana de entrenamiento
    print('Entrenando...')
    messagebox.showwarning("Advertencia", "Entrenando...")
    face_recognizer.train(faces_data, np.array(labels))

    model_path = 'C:\Cursos\Phyton\SistemaReconocimientoFacial\Entrenamiento\ModeloFaceFrontalData2024.xml'
    face_recognizer.write(model_path)
    print('Modelo guardado en', model_path)
    messagebox.showwarning("Advertencia", 'Modelo guardado en ' + model_path)

# Llamar a la función para entrenar el modelo
if __name__ == "__main__":
    entrenar_modelo()
