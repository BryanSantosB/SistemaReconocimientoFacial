import cv2
from PIL import Image, ImageTk
import time
import os
import shutil
from Entrenamiento.GenerarImagenes import capturar_imagenes
from Entrenamiento.Entrenamiento import entrenar_modelo
from Notificacion.Notificacion import enviar_correo
import customtkinter as ctk
import tkinter.messagebox as messagebox

cap = None
face_recognizer = None
faceClassif = None
run_recognition = True
hilo_reconocimiento = None
desconocido_detectado = False
tiempo_inicio = None

def cerrar_programa():
    global run_recognition
    run_recognition = False
    if hilo_reconocimiento:
        hilo_reconocimiento.join()
    ventana.destroy()

def registrar_entrenar():
    nombre = entrada_nombre.get()
    if not nombre:
        messagebox.showwarning("Advertencia", "Debe ingresar el nombre de la persona.")
    else:
        messagebox.showwarning("Advertencia",
                               "Por favor, asegúrate de estar solo/a en la cámara y de contar con buena iluminación antes de proceder. Ten en cuenta que el proceso puede tardar un máximo de 2 minutos. ¡Gracias por tu cooperación!")
        capturar_imagenes(nombre)
        entrenar_modelo()

def mostrar_personas():
    opciones_frame.pack_forget()
    contenedor_personas.pack(pady=20, padx=20, fill="both", expand=True)

    # Limpiar el contenido del contenedor_personas antes de agregar nuevos datos
    for widget in contenedor_personas.winfo_children():
        widget.destroy()

    # Mostrar los nombres de las carpetas en la carpeta Data
    ruta_data = 'C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\Data'
    if os.path.exists(ruta_data):
        carpetas = os.listdir(ruta_data)
        if carpetas:
            for carpeta in carpetas:
                frame = ctk.CTkFrame(contenedor_personas, fg_color="#191C23")
                frame.pack(pady=5, padx=5, fill="both")
                label = ctk.CTkLabel(frame, text=carpeta, text_color="white", font=("Helvetica", 12))
                label.pack(side="left", padx=10)
                btnEliminar = ctk.CTkButton(frame, text="Eliminar", width=50, command=lambda c=carpeta: eliminar_persona(c),
                                            fg_color="#FF4C4C", text_color="white", font=("Helvetica", 12), corner_radius=10)
                btnEliminar.pack(side="right", padx=5, pady=5)
        else:
            label = ctk.CTkLabel(contenedor_personas, text="No hay personas registradas.", text_color="white",
                                 font=("Helvetica", 12))
            label.pack(pady=10, padx=10)
    else:
        label = ctk.CTkLabel(contenedor_personas, text="La carpeta Data no existe.", text_color="white",
                             font=("Helvetica", 12))
        label.pack(pady=10, padx=10)

    # Agregar botón "Regresar" al final del contenedor_personas
    btnRegresar = ctk.CTkButton(contenedor_personas, text='Regresar', command=regresar_a_opciones,
                                fg_color="#191C23", text_color="white", corner_radius=10)
    btnRegresar.pack(pady=20)

def eliminar_persona(carpeta):
    respuesta = messagebox.askyesno("Confirmar", f"¿Estás seguro de que deseas eliminar la carpeta '{carpeta}'?")
    if respuesta:
        ruta_data = 'C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\Data'
        ruta_carpeta = os.path.join(ruta_data, carpeta)
        if os.path.exists(ruta_carpeta):
            shutil.rmtree(ruta_carpeta)
            messagebox.showinfo("Información", f"La carpeta '{carpeta}' ha sido eliminada.")
            mostrar_personas()
        else:
            messagebox.showwarning("Advertencia", f"La carpeta '{carpeta}' no existe.")

def regresar_a_opciones():
    contenedor_personas.pack_forget()
    opciones_frame.pack(pady=20, padx=20, fill="both", expand=True)

def iniciar_reconocimiento():
    global cap, face_recognizer, faceClassif, desconocido_detectado, tiempo_inicio

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

        rostro_desconocido_encontrado = False

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            if result[1] < 30 and result[0] < len(imagePaths):
                cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), (0, 255, 0), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, '{}'.format(result), (x, y + h + 25), 1, 1, (255, 255, 255), 2, cv2.LINE_AA)
                desconocido_detectado = False
                tiempo_inicio = None
            else:
                cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                rostro_desconocido_encontrado = True

        if rostro_desconocido_encontrado:
            if not desconocido_detectado:
                desconocido_detectado = True
                tiempo_inicio = time.time()
            elif tiempo_inicio and (time.time() - tiempo_inicio) > 5:
                enviar_correo()
                desconocido_detectado = False
                tiempo_inicio = None
        else:
            desconocido_detectado = False
            tiempo_inicio = None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(frame_rgb)
        imagen = imagen.resize((600, 400))  # Ajustar el tamaño del video a 600x400
        imagen_tk = ImageTk.PhotoImage(imagen)
        label_imagen.configure(image=imagen_tk)
        label_imagen.image = imagen_tk  # Mantener una referencia para evitar la recolección de basura
        ventana.update()

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def activar_reconocimiento():
    contenedor_derecho.pack(side="right", fill="both", expand=True)
    label_imagen.configure(text="")
    iniciar_reconocimiento()

def finalizar_reconocimiento():
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    # Limpiar la imagen del label
    label_imagen.configure(image='')
    label_imagen.image = None

    # Opcional: Mostrar un texto o imagen predeterminada
    label_imagen.configure(text="Reconocimiento desactivado")

# Crear ventana
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ventana = ctk.CTk()
ventana.title("FaceSecure")
ventana.geometry("1000x650")

# Crear contenedores
contenedor_principal = ctk.CTkFrame(ventana, fg_color="#1E2A38")
contenedor_principal.pack(pady=20, padx=20, fill="both", expand=True)

contenedor_izquierdo = ctk.CTkFrame(contenedor_principal, fg_color="#1E2A38", width=300)
contenedor_izquierdo.pack(side="left", fill="both", expand=True, padx=(0, 10))

contenedor_derecho = ctk.CTkFrame(contenedor_principal, fg_color="#1E2A38")
contenedor_derecho.pack(side="right", fill="both", expand=True)

contenedor_personas = ctk.CTkFrame(contenedor_izquierdo, fg_color="#2E3B4E")

# Contenedor izquierdo
opciones_frame = ctk.CTkFrame(contenedor_izquierdo, fg_color="#2E3B4E", corner_radius=15)
opciones_frame.pack(pady=20, padx=20, fill="both", expand=True)

logo = ctk.CTkImage(Image.open('C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\img\\logo.png'), size=(200, 180))
label_logo = ctk.CTkLabel(opciones_frame, image=logo, text='')
label_logo.pack(pady=20, padx=20)

# Caja de texto para ingresar el nombre de la persona
entrada_nombre = ctk.CTkEntry(opciones_frame, placeholder_text="Ingrese nombre",
                              fg_color="#1E2A38", text_color="white", font=("Roboto", 14), corner_radius=10)
entrada_nombre.pack(pady=10, padx=20, fill="x")

# Botones de opciones
btnRegistrarEntrenar = ctk.CTkButton(opciones_frame, text="Registrar y entrenar", command=registrar_entrenar,
                                     fg_color="#3498DB", text_color="white", font=("Roboto", 14), corner_radius=10,
                                     hover_color="#2980B9")
btnRegistrarEntrenar.pack(pady=10, padx=20, fill="x")

btnMostrarPersonas = ctk.CTkButton(opciones_frame, text="Mostrar personas", command=mostrar_personas,
                                   fg_color="#3498DB", text_color="white", font=("Roboto", 14), corner_radius=10,
                                   hover_color="#2980B9")
btnMostrarPersonas.pack(pady=10, padx=20, fill="x")

btnActivarReconocimiento = ctk.CTkButton(opciones_frame, text="Activar reconocimiento", command=activar_reconocimiento,
                                         fg_color="#3498DB", text_color="white", font=("Roboto", 14), corner_radius=10,
                                         hover_color="#2980B9")
btnActivarReconocimiento.pack(pady=10, padx=20, fill="x")

btnCerrar = ctk.CTkButton(opciones_frame, text="Cerrar", command=cerrar_programa,
                          fg_color="#E74C3C", text_color="white", font=("Roboto", 14), corner_radius=10,
                          hover_color="#C0392B")
btnCerrar.pack(pady=10, padx=20, fill="x")

# Contenedor derecho
contenedor_video = ctk.CTkFrame(contenedor_derecho, fg_color="#2E3B4E", corner_radius=15)
contenedor_video.pack(pady=20, padx=20, fill="both", expand=True)

label_video_titulo = ctk.CTkLabel(contenedor_video, text="Video en tiempo real", font=("Roboto", 18, "bold"))
label_video_titulo.pack(pady=10)

# Ajustar el tamaño del label que contiene el video
label_imagen = ctk.CTkLabel(contenedor_video, text="", width=600, height=400)
label_imagen.configure(text="Reconocimiento desactivado")
label_imagen.pack(pady=10, padx=10)

btnDetenerReconocimiento = ctk.CTkButton(contenedor_derecho, text="Detener reconocimiento", command=finalizar_reconocimiento,
                                         fg_color="#E74C3C", text_color="white", font=("Roboto", 14), corner_radius=10,
                                         hover_color="#C0392B")
btnDetenerReconocimiento.pack(pady=20, padx=20, fill="x")

# Variables
nombrePersona_var = ctk.StringVar()

#Centrar la ventana
ventana.eval('tk::PlaceWindow . center')
ventana.mainloop()
