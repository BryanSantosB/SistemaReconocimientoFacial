import cv2
import os
import imutils

def capturar_imagenes(person_name):
    data_path = 'C:/Cursos/Phyton/SistemaReconocimientoFacial/Data'
    person_path = os.path.join(data_path, person_name)

    if not os.path.exists(person_path):
        print('Carpeta creada:', person_path)
        os.makedirs(person_path)

    face_cascade = cv2.CascadeClassifier('C:\Cursos\Phyton\SistemaReconocimientoFacial\haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=320)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = frame.copy()

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rostro = aux_frame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(person_path, 'rostro_{}.jpg'.format(count)), rostro)
            count += 1

        cv2.imshow('Ventana de reconocimiento', frame)
        k = cv2.waitKey(30)
        if k == 27 or count == 99:
            break

    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para capturar imágenes
if __name__ == "__main__":
    person_name = 'WhiteMoi'
    capturar_imagenes(person_name)
