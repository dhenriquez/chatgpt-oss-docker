FROM python:3.11-slim

# Se crea un directorio de trabajo llamado /app
WORKDIR /app

# Se copian los archivos de requerimientos
COPY requerimientos.txt .
# Se instalan las dependencias en el contenedor
RUN pip install --no-cache-dir -r requerimientos.txt
# Se copian los archivos de la aplicación en el contenedor
COPY app/ ./app/

# Se expone el puerto 8000
EXPOSE 8000

# Se inicia el servidor Uvicorn para la aplicación FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]