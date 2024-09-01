# Use uma imagem base do Python
FROM python:3.9-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo de requisitos
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código-fonte
COPY . .

# Comando para rodar a aplicação
CMD ["python", "main.py"]
