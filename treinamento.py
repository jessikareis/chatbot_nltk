# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 09:56:04 2024

@author: jessi
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
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

# Conjunto expandido de intenções e perguntas
intents = {
    "comprar": [
        "Quero comprar computador?",
        "Quero comprar um laptop?",
        "Vocês vendem monitores?",
        "Quero comprar um HD, pode me ajudar?",
        "Quero comprar um SSD?",
        "Quero comprar um mouse?",
        "Quero comprar um teclado, pode me ajudar?",
        "Quero comprar um pen-drive?",
        "Quero comprar um carregador, pode me ajudar?",
        "Vocês têm computadores de alto desempenho?",
        "Comprar Headset?",
        "Quero comprar um tablet?",
        "Quero comprar uma memória RAM, pode me ajudar?",
        "Quero comprar placa mãe, pode me ajudar?",
        "Quero comprar impressora?"
    ],
    "informação_produto": [
        "Gostaria de informação?",
        "Pode me dar mais informações de armazenamento?",
        "Informações sobre impressora?",
        "Quais as informações da resolução do monitor?",
        "Informações sobre garantia?",
        "Informação do laptop?",
        "Gostaria de informação da placa de vídeo?",
        "Gostaria de informação de memória RAM?",
        "Gostaria de informação da bateria?",
        "Uma informação sobre computador à prova de água?",
        "Ele suporta upgrade de memória?",
        "O produto tem suporte a Wi-Fi 6?",
        "Quais informações da câmera?",
        "O teclado é retroiluminado?",
        "A tela é de LCD ou LED?"
    ],
    "suporte": [
        "Suporte técnico é oferecido?",
        "Como posso instalar o software?",
        "O suporte resolve problema no computador.",
        "O suporte verifica produto não funcionando?",
        "Vocês têm suporte remoto?",
        "Existe suporte por telefone?",
        "Vocês oferecem manutenção periódica?",
        "O suporte formata computador, podem ajudar?",
        "O suporte atualiza o sistema?",
        "O suporte faz backup de dados?",
        "Existe um suporte especializado para empresas?",
        "O suporte é gratuito?",
        "Abri um chamado para o suporte?",
        "Suporte realiza troca de peça?",
        "Suporte ajuda nas instruções?"
    ],
    "agradecimento": [
        "Obrigada pela ajuda!",
        "Muito obrigada!",
        "Valeu!",
        "Grata pela informação.",
        "Agradeço pela assistência!",
        "Obrigada por me ajudar!",
        "Foi muito útil, obrigada!",
        "Estou satisfeita com o suporte, obrigada!",
        "Obrigada por esclarecer minhas dúvidas.",
        "Valeu pelo atendimento!",
        "Agradeço a atenção!",
        "Muito obrigada pelo suporte!",
        "Grata pelo rápido atendimento!",
        "Vocês foram muito prestativos!",
        "Obrigada!"
    ],
    "recomendacao_produto": [
        "Qual recomendação sobre notebook para jogos?",
        "Que tipo de computador vocês recomendam para trabalho remoto?",
        "Quais acessórios são recomendados para produtividade?",
        "Qual mouse você recomenda para designer gráfico?",
        "Preciso de recomendação de teclado para programar?",
        "Que tipo de computador recomendam para edição de vídeos?",
        "O que recomendam para estudantes?",
        "Tem algum modelo para programadores?",
        "Qual o melhor monitor para evitar cansaço visual?",
        "Tem sugestões de cadeiras ergonômicas?",
        "Qual impressora é mais econômica?",
        "Quais acessórios ajudam na organização da mesa?",
        "Qual mouse é ideal para longas horas de trabalho?",
        "Vocês recomendam algum fone com cancelamento de ruído?",
        "Qual notebook tem o melhor custo-benefício?"
    ],
    "entrega_pagamento": [
        "Quais são as opções de entrega?",
        "Quais formas de pagamento vocês aceitam?",
        "Como posso fazer o pagamento das compras?",
        "O pagamento pode ser no cartão de crédito?",
        "A entrega tem frete grátis?",
        "Qual o prazo de entrega?",
        "Aceitam pagamento via Pix?",
        "Posso pagar com boleto?",
        "O parcelamento do pagamento tem juros?",
        "Vocês entregam no Brasil inteiro?",
        "Como acompanhar meu pedido?",
        "Posso retirar o produto na loja ou so entregam?",
        "Tem desconto para pagamento à vista?",
        "Qual é a política de reembolso?",
        "Quais são as taxas de entrega?"
    ],
    "disponibilidade": [
        "Qual a disponibilidade desse produto?", 
        "Quando vocês terão notebooks disponíveis?",
        "Qual a disponibilidade do monitor?",
        "Qual a disponibilidade do mouse?",
        "Há previsão de reposição do estoque?",
        "Quais produtos estão disponíveis?",
        "Qual a disponibilidade do SSD?",
        "Qual a disponibilidade do HD?",
        "Esse produto está esgotado?",
        "Vocês trabalham com pronta entrega?",
        "Qual a disponibilidade de cores?",
        "Posso reservar um produto?",
        "Vocês têm estoque do Headset?",
        "Esse modelo estará disponível em breve?",
        "Qual a disponibilidade desse dos laptops?"
    ]
}

# Função de pré-processamento com opção para stemming
def preprocess(text):
    tokens = word_tokenize(text.lower())
    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]
    else:
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)

# Caminho do arquivo salvo
modelo_path = "modelo_chatbot.pkl"

# Preparação dos dados para treinamento
dados = []
classes = []
for intent, perguntas in intents.items():
    for pergunta in perguntas:
        dados.append(pergunta)
        classes.append(intent)

dados_processados = [preprocess(texto) for texto in dados]

# Vetorização dos dados
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dados_processados)
y = classes

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# Avaliação do modelo
y_pred = modelo.predict(X_test)
print("Acurácia do modelo:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Salvar modelo treinado e vectorizer
with open(modelo_path, "wb") as f:
    pickle.dump((modelo, vectorizer), f)
print("Modelo treinado e salvo com sucesso.")