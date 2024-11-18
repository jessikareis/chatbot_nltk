# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 09:56:04 2024

@author: jessi
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
import pickle
import os

# Configuração inicial - Baixar pacotes necessários do NLTK
nltk.download('punkt')
nltk.download('rslp')
nltk.download('stopwords')

# Configuração do stemmer e stop words em português
stemmer = RSLPStemmer()
stop_words = set(stopwords.words('portuguese'))
use_stemming = True  # Configuração para uso de stemming (True or False)

# Respostas para cada intenção
respostas = {
    "comprar": "Para escolher o modelo e finalizar sua compra, acesse nosso site em: www.techimage.com/compra",
    "informação_produto": "Para detalhes técnicos completos, consulte a seção de especificações em: www.techimage.com/produtos",
    "suporte": "Nosso suporte técnico pode ser encontrado em: www.techimage.com/suporte",
    "agradecimento": "De nada! Estamos à disposição para ajudar. Visite-nos novamente em: www.techimage.com",
    "recomendacao_produto": "Para recomendações personalizadas, veja: www.techimage.com/recomendacoes",
    "entrega_pagamento": "Para saber mais sobre entrega e pagamento, acesse: www.techimage.com/entrega-pagamento",
    "disponibilidade": "Consulte a disponibilidade de produtos em: www.techimage.com/disponibilidade",
}

# Função de pré-processamento com opção para stemming
def preprocess(text):
    tokens = word_tokenize(text.lower())
    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]
    else:
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)

# Carregar modelo treinado 
modelo_path = "modelo_chatbot.pkl"
if os.path.exists(modelo_path):
    with open(modelo_path, "rb") as f:
        modelo, vectorizer = pickle.load(f)
    print("Modelo carregado com sucesso.")
else:
    print("Erro: Modelo não encontrado. Certifique-se de treinar o modelo primeiro executando 'treinamento.py'.")
    exit()

# Função de Resposta do Chatbot
def responder(pergunta):
    pergunta_processada = preprocess(pergunta)
    pergunta_vectorizada = vectorizer.transform([pergunta_processada])
    intent_pred = modelo.predict(pergunta_vectorizada)[0]
    return respostas.get(intent_pred, "Desculpe, não consegui entender. Poderia reformular sua pergunta?")

# Conjunto de perguntas para teste cego
testes_cegos = [
    ("Quais tipos de laptops vocês têm?", "comprar"),
    ("Quais as especificações de memória?", "informação_produto"),
    ("Vocês parcelam as compras?", "entrega_pagamento"),
    ("Qual notebook você recomenda para jogos?", "recomendacao_produto"),
    ("Esse produto está disponível?", "disponibilidade"),
    ("Vocês têm assistência remota?", "suporte"),
    ("Obrigado pela ajuda!", "agradecimento"), 
    ("Posso pagar com cartão de crédito?", "entrega_pagamento"),
    ("Vocês têm desktops para jogos?", "comprar"),
    ("Qual é a política de reembolso?", "entrega_pagamento"),
    ("Que tipo de computador recomendam para edição de vídeos?", "recomendacao_produto"),
]

# Realizando o teste cego
acertos = 0
erros = []
for pergunta, classe_correta in testes_cegos:
    classe_predita = modelo.predict(vectorizer.transform([preprocess(pergunta)]))[0]
    print(f"Pergunta: '{pergunta}' | Classe Correta: '{classe_correta}' | Classe Predita: '{classe_predita}'")
    if classe_predita == classe_correta:
        acertos += 1
    else:
        erros.append((pergunta, classe_correta, classe_predita))

acuracia_teste_cego = acertos / len(testes_cegos)
print(f"Acurácia no teste cego: {acuracia_teste_cego:.2f}")

# Exibir erros detalhados para análise
if erros:
    print("Erros no teste cego:")
    for pergunta, classe_correta, classe_predita in erros:
        print(f"Pergunta: '{pergunta}' | Classe Correta: '{classe_correta}' | Classe Predita: '{classe_predita}'")

# Interação do chatbot com o usuário
print("Digite 'sair' para encerrar.")
while True:
    pergunta_usuario = input("Você: ")
    if pergunta_usuario.lower() in ["sair", "obrigado"]:
        print("Chatbot: Foi um prazer ajudar! Até a próxima.")
        break
    resposta = responder(pergunta_usuario)
    print("Chatbot:", resposta)
