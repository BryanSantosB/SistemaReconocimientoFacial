import smtplib

remitente = 'xbandiiidoxbandiiido@gmail.com'
#gus04tavo17@gmail.com
destinatario = 'xbandiiidoxbandiiido@gmail.com'
asunto = 'Alerta de reconocimiento facial'
cuerpo = 'Se ha detectado un rostro desconocido'

mensaje = f'Subject: {asunto}\n\n{cuerpo}'

def enviar_correo():
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(remitente, 'ajms iwoq rpxs wnak')
            smtp.sendmail(remitente, destinatario, mensaje)
            print('Correo enviado correctamente')
    except Exception as e:
        print(f'Error al enviar el correo: {e}')

