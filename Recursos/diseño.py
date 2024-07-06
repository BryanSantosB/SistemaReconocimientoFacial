import tkinter as tk
from PIL import Image, ImageDraw, ImageTk, ImageFilter


def centrar_ventana(ventana):
    ventana.update_idletasks()
    ancho_ventana = ventana.winfo_width()
    alto_ventana = ventana.winfo_height()
    x = (ventana.winfo_screenwidth() // 2) - (ancho_ventana // 2)
    y = (ventana.winfo_screenheight() // 2) - (alto_ventana // 2)
    ventana.geometry('{}x{}+{}+{}'.format(ancho_ventana, alto_ventana, x, y))


def crear_boton_redondeado(ventana, texto, comando=None, color_fondo="#191C23", color_texto="white", color_borde="white", grosor_borde=1, radio=20):
    # Crear imagen de botón redondeado
    ancho = 200
    alto = 50
    imagen = Image.new('RGBA', (ancho, alto), (0, 0, 0, 0))
    draw = ImageDraw.Draw(imagen)

    # Dibujar el fondo del botón
    draw.rounded_rectangle(
        [(grosor_borde, grosor_borde), (ancho - grosor_borde, alto - grosor_borde)],
        radius=radio,
        fill=color_fondo,
        outline=color_borde,
        width=grosor_borde
    )

    imagen_tk = ImageTk.PhotoImage(imagen)

    # Crear canvas para manejar los eventos de clic
    canvas = tk.Canvas(ventana, width=ancho, height=alto, bg=color_fondo, highlightthickness=0)
    canvas.image = imagen_tk  # Guardar referencia para evitar que la imagen sea recolectada por el recolector de basura

    # Colocar la imagen en el canvas
    canvas.create_image(0, 0, anchor='nw', image=imagen_tk)
    canvas.create_text(ancho//2, alto//2, text=texto, fill=color_texto, font=("Arial", 12))

    # Asignar el comando al botón
    if comando:
        canvas.bind("<Button-1>", lambda event: comando())

    canvas.pack(padx=10, pady=10)

    return canvas

def crear_caja_texto_redondeada(ventana, color_fondo="#191C23", color_texto="white", color_borde="white", grosor_borde=1, radio=20):
    ancho = 200
    alto = 50
    imagen = Image.new('RGBA', (ancho, alto), (0, 0, 0, 0))
    draw = ImageDraw.Draw(imagen)

    draw.rounded_rectangle(
        [(grosor_borde, grosor_borde), (ancho - grosor_borde, alto - grosor_borde)],
        radius=radio,
        fill=color_fondo,
        outline=color_borde,
        width=grosor_borde
    )

    imagen_tk = ImageTk.PhotoImage(imagen)

    canvas = tk.Canvas(ventana, width=ancho, height=alto, bg=color_fondo, highlightthickness=0)
    canvas.create_image(0, 0, image=imagen_tk, anchor='nw')
    canvas.image = imagen_tk

    entry_var = tk.StringVar()
    entry = tk.Entry(canvas, textvariable=entry_var, fg=color_texto, bg=color_fondo, font=("Arial", 12), bd=0, highlightthickness=0, justify="center")
    entry.place(relx=0.5, rely=0.5, anchor='center', width=ancho-20, height=alto-20)

    canvas.pack(padx=10, pady=10)

    return entry_var

def aplicar_fondo(ventana, ruta_imagen):
    imagen = Image.open(ruta_imagen)
    imagen = imagen.resize((ventana.winfo_screenwidth(), ventana.winfo_screenheight()), Image.LANCZOS)
    imagen = imagen.filter(ImageFilter.GaussianBlur(15))

    imagen_tk = ImageTk.PhotoImage(imagen)

    fondo = tk.Label(ventana, image=imagen_tk, bg='#F1F1F1')  # Usamos el color de fondo de la ventana principal
    fondo.image = imagen_tk
    fondo.place(x=0, y=0, relwidth=1, relheight=1)

    # Configurar fondo de los contenedores como transparente
    for widget in ventana.winfo_children():
        widget.config(bg='black')  # Establecer el fondo de los widgets hijos como transparente

    # Forzar la actualización de los contenedores
    ventana.update_idletasks()



def configurar_contenedores(contenedor_izquierdo, contenedor_derecho):
    # Obtener el ancho de la pantalla
    ancho_pantalla = contenedor_izquierdo.winfo_screenwidth()

    # Configurar el ancho de los contenedores
    contenedor_izquierdo.config(width=int(ancho_pantalla * 0.4))
    contenedor_derecho.config(width=int(ancho_pantalla * 0.6))
