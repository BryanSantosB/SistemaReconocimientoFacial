import cv2
import os
from PIL import Image, ImageTk
import threading

stop_event = threading.Event()
cap = None  # Declaración global de cap

def reconocimiento_facial(label):
    global cap  # Declarar cap como global para acceder a la misma variable global en todo el script
    dataPath = 'C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\Data'
    imagePaths = os.listdir(dataPath)
    print('imagePath', imagePaths)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\Entrenamiento\\ModeloFaceFrontalData2024.xml')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Asignar a la variable global cap
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def update_frame():
        if stop_event.is_set():
            cap.release()  # Liberar la cámara
            return

        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = gray.copy()

            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
                result = face_recognizer.predict(rostro)

                if result[1] < 42:
                    cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), (0, 255, 0), -1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, '{}'.format(result), (x, y + h + 25), 1, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Convertir el frame de OpenCV a un formato que Tkinter pueda usar
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            label.imgtk = imgtk
            label.configure(image=imgtk)

        label.after(10, update_frame)

    update_frame()

def detener_reconocimiento():
    stop_event.set()
    global cap  # Declarar cap como global para acceder a la misma variable global en todo el script
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
