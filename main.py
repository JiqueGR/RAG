import pdfplumber
import os
import chromadb
from datasets import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def treinar_modelo(texto_completo):
    token = os.getenv("TOKEN_HUGGINGFACE")
    nome_modelo = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(nome_modelo, use_auth_token=token)
    modelo = AutoModelForCausalLM.from_pretrained(nome_modelo, use_auth_token=token)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    encodings = tokenizer(texto_completo, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'].squeeze(),
        'attention_mask': encodings['attention_mask'].squeeze(),
        'labels': encodings['input_ids'].squeeze()
    })

    argumentos_treinamento = TrainingArguments(
        output_dir="./resultados",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
    )

    treinador = Trainer(
        model=modelo,
        args=argumentos_treinamento,
        train_dataset=dataset,
    )

    treinador.train()

    modelo.save_pretrained("./trained_llama3")
    tokenizer.save_pretrained("./trained_llama3")

    return modelo, tokenizer

def carregar_modelo_e_tokenizer():
    diretorio_modelo = "./trained_llama3"
    modelo = AutoModelForCausalLM.from_pretrained(diretorio_modelo)
    tokenizer = AutoTokenizer.from_pretrained(diretorio_modelo)
    return modelo, tokenizer

def perguntar_ollama(modelo, tokenizer, contexto, pergunta):
    contexto_formatado = f"{contexto}\n\nPergunta: {pergunta}"
    inputs = tokenizer(contexto_formatado, return_tensors="pt", padding=True, truncation=True, max_length=512)

    resultado = modelo.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )

    resposta = tokenizer.decode(resultado[0], skip_special_tokens=True)
    return resposta

def main():
    caminho_pdf = "Politica_de_Privacidade.pdf"
    texto_completo = ""

    with pdfplumber.open(caminho_pdf) as pdf:
        num_paginas = len(pdf.pages)
        for pagina in range(num_paginas):
            pagina_pdf = pdf.pages[pagina]
            texto = pagina_pdf.extract_text()
            if texto:
                texto_completo += texto.replace("\n", " ")

    diretorio_modelo = "./modelo_treinado"
    if not os.path.exists(diretorio_modelo):
        treinar_modelo(texto_completo)
    modelo, tokenizer = carregar_modelo_e_tokenizer()

    divisor_texto = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=3
    )

    vetores = divisor_texto.split_text(texto_completo)

    cliente_chroma = chromadb.Client(Settings())
    colecao = cliente_chroma.create_collection(name="minha_colecao")

    for id, vetor in enumerate(vetores):
        colecao.add(
            documents=[vetor],
            ids=[str(id)]
        )

    consulta = colecao.query(
        query_texts=["contato"],
        n_results=1
    )

    if consulta and consulta['documents']:
        contexto_relevante = consulta['documents'][0]
        resposta = perguntar_ollama(modelo, tokenizer, contexto_relevante, "contato")
        resposta = resposta.split("\n")
        print(f"Resposta gerada: {resposta[0]}")

if __name__ == "__main__":
    main()
