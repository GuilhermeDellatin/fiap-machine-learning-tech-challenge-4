# FIAP Machine Learning Tech Challenge 4

Seu desafio é criar um modelo preditivo de redes neurais Long Short 
Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores 
de uma empresa à sua escolha e realizar toda a pipeline de desenvolvimento, 
desde a criação do modelo preditivo até o deploy do modelo em uma API que 
permita a previsão de preços de ações.

|![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)  |
|:-----------------------------------------------------------------:|

-----------------------------------

## Sumário

- [Descrição](#descrição)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Licença e Autores](#licença-e-autores)

-----------------------------------

## Descrição

Seu Tech Challenge precisa seguir os seguintes requisitos:

1. Coleta e Pré-processamento dos Dados
• Coleta de Dados: utilize um dataset de preços históricos de ações, 
como o Yahoo Finance ou qualquer outro dataset financeiro disponível 
(dica: utilize a biblioteca yfinance). 

2. Desenvolvimento do Modelo LSTM 
• Construção do Modelo: implemente um modelo de deep learning 
utilizando LSTM para capturar padrões temporais nos dados de preços 
das ações. 
• Treinamento: treine o modelo utilizando uma parte dos dados e ajuste 
os hiperparâmetros para otimizar o desempenho. 
• Avaliação: avalie o modelo utilizando dados de validação e utilize 
métricas como MAE (Mean Absolute Error), RMSE (Root Mean Square 
Error), MAPE (Erro Percentual Absoluto Médio) ou outra métrica 
apropriada para medir a precisão das previsões.

3. Salvamento e Exportação do Modelo 
• Salvar o Modelo: após atingir um desempenho satisfatório, salve o 
modelo treinado em um formato que possa ser utilizado para 
inferência.

4. Deploy do Modelo 
• Criação da API: desenvolva uma API RESTful utilizando Flask ou 
FastAPI para servir o modelo. A API deve permitir que o usuário 
forneça dados históricos de preços e receba previsões dos preços 
futuros. 

5. Escalabilidade e Monitoramento 
• Monitoramento: configure ferramentas de monitoramento para 
rastrear a performance do modelo em produção, incluindo tempo de 
resposta e utilização de recursos. 
Entregáveis: 
• Código-fonte do modelo LSTM no seu repositório do GIT + 
documentação do projeto. 
• Scripts ou contêineres Docker para deploy da API. 
• Link para a API em produção, caso tenha sido deployada em um 
ambiente de nuvem.

-----------------------------------

## Tecnologias Utilizadas

- **Python 3.13**

-----------------------------------

## Licença e Autores

### 🧑‍💻 Desenvolvido por

- `Beatriz Rosa Carneiro Gomes - RM365967`
- `Cristine Scheibler - RM365433`
- `Guilherme Fernandes Dellatin - RM365508`
- `Iana Alexandre Neri - RM360484`
- `João Lucas Oliveira Hilario - RM366185`

Este projeto é apenas para fins educacionais e segue a licença MIT.