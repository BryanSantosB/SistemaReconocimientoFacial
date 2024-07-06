import tkinter as tk
from tkinter import Label, Tk

import cv2
from PIL import Image, ImageTk

# Ventana de Tkinter
root = Tk()
root.title("Reconocimiento Facial")
root.geometry("800x600")

# Label para mostrar el video
label_video = Label(root)
label_video.pack(padx=10, pady=10)

def actualizar_frame(frame):
    # Convertir el frame de OpenCV a un formato que Tkinter pueda usar
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    # Actualizar el label con la nueva imagen
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

# Función principal de Tkinter para iniciar la ventana
def iniciar_interfaz():
    root.mainloop()

# Ejecutar la interfaz de usuario
if __name__ == "__main__":
    iniciar_interfaz()
