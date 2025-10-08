# Usa una imagen oficial de Python (ajusta la versión si lo deseas, 3.11 es estable)
FROM python:3.11

# Instala dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y libgl1

# Crea el directorio de trabajo
WORKDIR /app

# Copia los archivos de la app
COPY . /app

# Instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expón el puerto 8080 (Railway lo usa por defecto)
EXPOSE 8080

# Comando para arrancar tu app de Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
