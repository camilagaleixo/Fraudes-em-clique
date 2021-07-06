# Projeto 1: Detecção de Fraudes no Tráfego de Cliques 
# em Propagandas de Aplicações Mobile

setwd("C:/Users/camil/OneDrive/Documentos/FCD/BigDataRAzure/Projetos_feedback/Execução/Projeto 1")
getwd()

## ETAPA 1: Definição do problema

# Objetivo: Prever se um usuário fará o download de um app após clicar em um anúncio de um aplicativo móvel.

## ETAPA 2: Coleta dos dados

# Os dados foram baixados do site kaggle, no endereço abaixo:
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

# Inicialmente será carregado o arquivo de treino "train_sample",
# utilizando o pacote 'readr', devido o arquivo ser grande:

library('readr')

dados <- read_csv("train_sample.csv")

## ETAPA 3: Análise exploratória

# Após os dados serem carregados, os mesmos serão analisados:
# Em uma visualização prévia, é possível verificar que os dados possuem 8 colunas
head(dados)

# Em uma visualização mais profunda, verifica-se a existência de 100 mil
# observações (linhas) no conjunto de treino
View(dados)

# Verificando a classificação dos dados
str(dados)

# Analisando a distribuição dos dados
hist(dados$is_attributed)
# Há muito mais dados com zero do que com 1. Sendo assim, será feita uma nova divisão dos dados
# em treino e teste.

# Verificando se existem dados missing no dataset
library("Amelia")
missmap(dados, main = "Fraude de clicks - Mapa de Dados Missing", col = c("yellow", "black"), legend = FALSE)
# A única coluna que possui dados missing é a coluna "attributed_time". Essa coluna está vazia
# quando o usuário não fez o download. Dessa forma, era provável que isso ocorresse.

# Verificando as variáveis mais relevantes no dataset
library(randomForest)
varRelev <- randomForest(is_attributed ~ ip + app + device + os + channel + click_time,
                       data = dados,
                       ntree = 100, nodesize = 10, importance = T)
varImpPlot(varRelev)
# Para verificar as variáveis importantes, a variável target foi comparada com todas as outras
# variáveis, exceto a "attributed_time", visto que esta só possui dados, quando "is_attributed"
# é igual a TRUE.


## ETAPA 4: Pré-processamento (se necessário)

# Como a última coluna ("is_attributed") é a variável target e está classificada
# como numérica, a mesma será transformada para o tipo fator:
str(dados)
dados$is_attributed <- as.factor(dados$is_attributed)
str(dados)

## ETAPA 5: Divisão dos dados em treino e teste

# Como há muitas linhas no dataset e poucos dados são referentes ao "is_attributed" == 0,
# para ter um dataset mais equilibrado, será inserido em um novo dataset(dadosT) apenas os dados
# em que "is_attributed" for igual a 1. Posteriormente, será retirada uma amostra de mesmo
# tamanho com "is_attributed" == 0 e salva em um novo dataset(dadosF).
# Os dois novos datasets serão agrupados, formando um novo dataset equilibrado para fazer a
# divisão(dados2):

library(dplyr)
dadosT <- dados %>%
  filter(is_attributed == 1)
View(dadosT)

dadosF <- dados %>%
  filter(is_attributed == 0) %>%
  sample_n(size = 227)
View(dadosF)

dados2 <- bind_rows(dadosT, dadosF)

# Após criado um novo dataset, o mesmo será dividido em dados de treino e teste:

library(caTools)
splits <- sample.split(dados2$is_attributed, SplitRatio = 0.70)

dados_treino <- subset(dados2, splits == T)
dados_teste <- subset (dados2, splits == F)

## ETAPA 6: Treinamento do modelo
# O modelo será treinado utilizando o algoritmo de árvode de decisão
modelo <- randomForest(is_attributed ~ ip + app + device + os + channel + click_time,
                       data = dados_treino,
                       ntree = 100, 
                       nodesize = 10)

## ETAPA 7: Avaliação do modelo
modelo

# Aplicando o modelo sobre os dados de teste
previsao <- predict(modelo, dados_teste)
previsao

# Analisando a Confusion Matrix
library("caret")
table(dados_teste$is_attributed, previsao)
confusionMatrix(table(dados_teste$is_attributed, previsao), positive = "1")

## ETAPA 8: Otimização do modelo
# Como o modelo atingiu uma acurácia de 91%, não será feita nenhuma otimização, 
# pois este percentual já é considerado ótimo.

## ETAPA 9: Entrega do resultado
# O trabalho será entregue em pdf, o qual é gerado a partir do próprio R.