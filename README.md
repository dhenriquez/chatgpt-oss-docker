# chatgpt-oss-docker
Test de Chat GPT OSS con Docker

Luego de realizar docker compose build se debe instalar el modelo, para eso se debe ejecutar
'''
docker exec -it ollama-container ollama pull gemma:2b
'''
de esta manera se instalará en el contenedor "ollama-container" el modelo gemma:2b

'''
docker exec -it ollama-container ollama pull gpt-oss:20b
'''
para el modelo gpt-oss:20b