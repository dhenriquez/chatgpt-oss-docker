# chatgpt-oss-docker
Test de Chat GPT OSS con Docker

Luego de realizar docker compose build se debe instalar el modelo, para eso se debe ejecutar:

'''bash
docker exec -it ollama-container ollama pull gemma:2b
'''

de esta manera se instalará en el contenedor "ollama-container" el modelo gemma:2b

'''bash
docker exec -it ollama-container ollama pull gpt-oss:20b
'''
para el modelo gpt-oss:20b

Para entrar en el contenedor:

'''bash
docker exec -it ollama-container bash
'''


alternativa: eliminar el "entrypoint: >" de docker-compose.yml y luego entrar en el contenedor para hacer pull a los modelos necesarios. tambien en vez de entrar en el contenedor solo ejecutar:

- docker exec ollama-container ollama pull gemma:2b
- docker exec ollama-container ollama pull llama3:8b
- docker exec ollama-container ollama pull deepseek-r1:1.5b
- docker exec ollama-container ollama pull gpt-oss:20b
- docker exec ollama-container ollama pull tinyllama

Esos son los modelos que tengo en main.py