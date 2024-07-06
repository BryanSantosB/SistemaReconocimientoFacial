import cv2
from PIL import Image, ImageTk
import time
import tkinter as tk
from tkinter import messagebox
import os
import shutil
from Entrenamiento.GenerarImagenes import capturar_imagenes
from Entrenamiento.Entrenamiento import entrenar_modelo
from Notificacion.Notificacion import enviar_correo
from Recursos.diseño import centrar_ventana, crear_boton_redondeado, aplicar_fondo, configurar_contenedores, \
    crear_caja_texto_redondeada

cap = None
face_recognizer = None
faceClassif = None
run_recognition = True
hilo_reconocimiento = None

def cerrar_programa():
    global run_recognition
    run_recognition = False
    if hilo_reconocimiento:
        hilo_reconocimiento.join()
    ventana.destroy()

def registrar_entrenar():
    nombre = nombrePersona_var.get()
    if not nombre:
        messagebox.showwarning("Advertencia", "Debe ingresar el nombre de la persona.")
    else:
        messagebox.showwarning("Advertencia",
                               "Por favor, asegúrate de estar solo/a en la cámara y de contar con buena iluminación antes de proceder. Ten en cuenta que el proceso puede tardar un máximo de 2 minutos. ¡Gracias por tu cooperación!")
        capturar_imagenes(nombre)
        entrenar_modelo()

def mostrar_personas():
    contenedor_izquierdo.pack_forget()
    contenedor_personas.pack(fill="both", expand=True)

    # Limpiar el contenido del contenedor_personas antes de agregar nuevos datos
    for widget in contenedor_personas.winfo_children():
        widget.destroy()

    # Mostrar los nombres de las carpetas en la carpeta Data
    ruta_data = 'C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\Data'
    if os.path.exists(ruta_data):
        carpetas = os.listdir(ruta_data)
        if carpetas:
            for carpeta in carpetas:
                frame = tk.Frame(contenedor_personas, bg="#191C23")
                frame.pack(pady=5, padx=5, fill="x")
                label = tk.Label(frame, text=carpeta, bg="#191C23", fg="white", font=("Helvetica", 12))
                label.pack(side="left", padx=10)
                btnEliminar = tk.Button(frame, text="Eliminar", command=lambda c=carpeta: eliminar_persona(c),
                                        bg="#FF4C4C", fg="white", font=("Helvetica", 12), relief="flat")
                btnEliminar.pack(side="right", padx=10)
        else:
            label = tk.Label(contenedor_personas, text="No hay personas registradas.", bg="#191C23", fg="white",
                             font=("Helvetica", 12))
            label.pack(pady=10, padx=10)
    else:
        label = tk.Label(contenedor_personas, text="La carpeta Data no existe.", bg="#191C23", fg="white",
                         font=("Helvetica", 12))
        label.pack(pady=10, padx=10)

    # Agregar botón "Regresar" al final del contenedor_personas
    btnRegresar = crear_boton_redondeado(contenedor_personas, texto='Regresar', comando=regresar_a_opciones,
                                         color_fondo="#191C23", color_texto="white", color_borde="white",
                                         grosor_borde=1, radio=20)
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
    contenedor_izquierdo.pack(fill="both", expand=True)
    ventana.geometry("260x575")

def iniciar_reconocimiento():
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

        rostro_desconocido_encontrado = False

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            if result[1] < 50:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
        imagen_tk = ImageTk.PhotoImage(imagen)
        label_imagen.config(image=imagen_tk)
        label_imagen.image = imagen_tk  # Mantener una referencia para evitar la recolección de basura
        ventana.update()

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def activar_reconocimiento():
    contenedor_izquierdo.pack_forget()
    contenedor_derecho.pack(fill="both", expand="True")
    ventana.geometry("800x620")
    iniciar_reconocimiento()

def finalizar_reconocimiento():
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    contenedor_derecho.pack_forget()
    contenedor_izquierdo.pack(fill="both", expand=True)
    ventana.geometry("260x580")

# Crear ventana
ventana = tk.Tk()
ventana.title("FaceSecure")
ventana.geometry("260x575")
ventana.config(bg="#F1F1F1")
logo = tk.PhotoImage(file='C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\img\\Logo.png')
logo = logo.subsample(2, 2)
ventana.iconphoto(True, logo)

# Aplicar fondo
ruta_fondo = 'C:\\Cursos\\Phyton\\SistemaReconocimientoFacial\\img\\fondo_app.jpg'
aplicar_fondo(ventana, ruta_fondo)

# Crear contenedores
contenedor_principal = tk.Frame(ventana, bg="#191C23", highlightthickness=0)
contenedor_principal.pack(pady=10, padx=10, fill="both", expand=True)
contenedor_izquierdo = tk.Frame(contenedor_principal, bg="#191C23", highlightthickness=0)
contenedor_izquierdo.pack(side="left", fill="both", expand=True)
contenedor_derecho = tk.Frame(contenedor_principal, bg="#191C23", highlightthickness=0)
contenedor_personas = tk.Frame(contenedor_principal, bg="#191C23", highlightthickness=0)

# Contenedor izquierdo
opciones_frame = tk.Frame(contenedor_izquierdo, bg="#191C23", highlightthickness=0)
opciones_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Logo del aplicativo
Logo = tk.Label(opciones_frame, image=logo, bg="#191C23")
Logo.pack(pady=10, padx=10)

nombrePersona_var = crear_caja_texto_redondeada(opciones_frame)
btnRegistrarPersona = crear_boton_redondeado(opciones_frame, texto='Autorizar Persona', comando=registrar_entrenar,
                                             color_fondo="#191C23", color_texto="white", color_borde="white",
                                             grosor_borde=1, radio=20)
btnActivarReconocimiento = crear_boton_redondeado(opciones_frame, texto='Activar Seguridad', comando=activar_reconocimiento,
                                                  color_fondo="#191C23", color_texto="white", color_borde="white",
                                                  grosor_borde=1, radio=20)
btnMostrarPersonas = crear_boton_redondeado(opciones_frame, texto='Mostrar Personas', comando=mostrar_personas,
                                            color_fondo="#191C23", color_texto="white", color_borde="white",
                                            grosor_borde=1, radio=20)
btnSalir = crear_boton_redondeado(opciones_frame, texto='Cerrar Software', comando=cerrar_programa,
                                  color_fondo="#191C23", color_texto="white", color_borde="white", grosor_borde=1,
                                  radio=20)

# Contenedor derecho
logo_frame = tk.Frame(contenedor_derecho, bg="#191C23", highlightthickness=0)
logo_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Etiqueta del reconocimiento facial en tiempo real
label_imagen = tk.Label(logo_frame, bg="black")
label_imagen.pack(pady=10, padx=10, fill="both", expand=True)

# Botón para finalizar reconocimiento
btnFinalizarReconocimiento = crear_boton_redondeado(logo_frame, texto='Finalizar Reconocimiento', comando=finalizar_reconocimiento,
                                                    color_fondo="#191C23", color_texto="white", color_borde="white",
                                                    grosor_borde=1, radio=20)
btnFinalizarReconocimiento.pack(pady=10, padx=10)

# Configurar los anchos de los contenedores después de crear la ventana
ventana.update_idletasks()
configurar_contenedores(contenedor_izquierdo, contenedor_derecho)

# Ocultar contenedor derecho y contenedor_personas inicialmente
contenedor_derecho.pack_forget()
contenedor_personas.pack_forget()

# Correr ventana
centrar_ventana(ventana)
ventana.mainloop()