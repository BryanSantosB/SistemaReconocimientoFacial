import cv2
import os

cap = None  # Variable global para la captura de video

def iniciar_reconocimiento():
    global cap
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

        if ret:
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

            cv2.imshow('Ventana de reconocimiento', frame)

            # Salir del bucle si se presiona la tecla 'Esc'
            if cv2.waitKey(30) == 27:
                break

    # Liberar recursos al finalizar
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

# Ejemplo de uso
if __name__ == "__main__":
    iniciar_reconocimiento()
