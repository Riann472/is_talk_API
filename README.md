# API de Reconhecimento de Fala

API para detecção e transcrição de fala em arquivos de áudio. Utiliza o modelo YAMNet do TensorFlow Hub para classificar o áudio e, caso detecte fala, envia para uma API externa de transcrição.

---

## Requisitos

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

---

## Como rodar

1. **Clone o repositório e acesse a pasta do projeto:**
    ```bash
    git clone https://github.com/Riann472/is_talk_API.git
    cd is_talk_API
    ```

2. **Suba o container com Docker Compose:**
    ```bash
    docker compose up -d
    ```

3. **Acesse a API:**

    O serviço estará disponível em:  
    [http://localhost:8000](http://localhost:8000)

---

## Como usar

- Faça um POST para `/` enviando um arquivo de áudio (campo `files`).

### Usando `curl`:
```bash
curl -F "files=@seuarquivo.wav" http://localhost:8000/
```

### Usando Postman ou Insomnia:
- Crie uma nova requisição do tipo `POST` para `http://localhost:8000/`.
- No corpo da requisição, selecione o tipo `form-data`.
- Adicione um campo chamado `files` e faça o upload do(s) arquivo(s) de áudio desejado(s), conforme exemplo abaixo:

| Chave  | Valor                |
|--------|----------------------|
| files  | outroarquivo.mp3     |

- Envie a requisição.

- A resposta será uma categoria detectada e, se for fala, a transcrição.

### Exemplo de resposta
Fala:
```json
  {
   "categoria": "Fala",
   "message": "Audio transcrito",
   "filename": "audio.ogg"
  }
```

Risada:
```json
  {
   "categoria": "Risada",
   "error": "Audio não classificado como fala, não sera transcrito."
  }
```

---

## Estrutura dos arquivos

- `server.py`: Código principal da API FastAPI.
- `Dockerfile`: Configuração do container.
- `docker-compose.yml`: Orquestração do serviço.
- `requirements.txt`: Dependências Python.

---

## Observações

- O container expõe a porta 8000.
- O sistema utiliza o modelo YAMNet para classificar sons e uma API externa para transcrição de fala.
- Para desenvolvimento local sem Docker, crie um ambiente virtual e instale as dependências manualmente.

---