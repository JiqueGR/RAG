Recuperação Aumentada de Conhecimento com ChromaDB e Hugging Face

Este projeto implementa uma aplicação para Recuperação Aumentada de Conhecimento (RAG), utilizando tecnologias como Python, LangChain, ChromaDB, Docker, e transformers da Hugging Face. A aplicação treina um modelo de linguagem para responder a perguntas sobre um texto específico, extraído de um documento PDF. 

Funcionalidades
Extração de texto de PDFs usando pdfplumber.
Processamento do texto usando RecursiveCharacterTextSplitter para dividir o conteúdo em fragmentos menores.
Armazenamento e consulta de vetores de texto usando ChromaDB.
Treinamento de um modelo de linguagem com transformers da Hugging Face.
Utilização de Docker para criação de um ambiente consistente.

Tecnologias Utilizadas
Python
LangChain
ChromaDB
Docker
Hugging Face Transformers
pdfplumber

Dockerfile: Define a imagem Docker que será usada para rodar a aplicação.
docker-compose.yml: Configura e orquestra o contêiner Docker, definindo como o ambiente será iniciado.
main.py: Script principal que executa todas as etapas da aplicação, desde a extração do texto do PDF até o treinamento do modelo e a realização de consultas.
requirements.txt: Lista as dependências Python necessárias para a execução da aplicação.
