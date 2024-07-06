import cv2
import os

def reconocimiento_facial():
    data_path = 'C:/Cursos/Phyton/SistemaReconocimientoFacial/Data'
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    models = {}
    for person_folder in os.listdir(data_path):
        model_path = os.path.join('C:/Cursos/Phyton/SistemaReconocimientoFacial/Entrenamiento', f'Modelo_{person_folder}.xml')
        if os.path.exists(model_path):
            models[person_folder] = cv2.face.LBPHFaceRecognizer_create()
            models[person_folder].read(model_path)
        else:
            print(f'Advertencia: No se encontró un modelo para {person_folder}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_roi = aux_frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (720, 720), interpolation=cv2.INTER_CUBIC)

            predictions = {}
            for person, model in models.items():
                predictions[person] = model.predict(face_roi)

            best_prediction = min(predictions.values(), key=lambda x: x[1])

            best_person = min(predictions, key=lambda x: predictions[x][1])

            cv2.putText(frame, '{}'.format(best_prediction[1]), (x, y + h + 25), 1, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if best_prediction[1] < 45:
                cv2.putText(frame, best_person, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), (0, 255, 0), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


        cv2.imshow('Ventana de reconocimiento', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función si se ejecuta este archivo directamente
if __name__ == "__main__":
    reconocimiento_facial()
