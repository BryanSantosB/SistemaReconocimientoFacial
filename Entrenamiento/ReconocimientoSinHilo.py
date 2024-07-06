# reconocimiento.py
import cv2
import os
from PIL import Image, ImageTk

def iniciar_reconocimiento(label_imagen, ventana):
    global cap, face_recognizer, faceClassif

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Iniciar captura de video desde la cámara
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    dataPath = 'C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\Data'
    imagePaths = os.listdir(dataPath)
    print('imagePath', imagePaths)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\Entrenamiento\\ModeloFaceFrontalData2024.xml')

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            if result[1] < 50:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(frame_rgb)
        imagen_tk = ImageTk.PhotoImage(imagen)
        label_imagen.config(image=imagen_tk)
        label_imagen.image = imagen_tk  # Mantener una referencia para evitar la recolección de basura
        ventana.update()

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

iniciar_reconocimiento()