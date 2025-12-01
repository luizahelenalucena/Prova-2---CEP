# Prova 2 – CEP (Controle Estatístico de Processo)

Este repositório contém a solução da **Prova 2 de CEP**, utilizando um conjunto de dados de defeitos de manufatura para construção de **cartas de controle por atributos** (P, NP, C e U) em Python.

O projeto foi desenvolvido no **Google Colab** e utiliza bibliotecas de análise de dados, estatística e visualização gráfica.

---

## 📘 Abrir o projeto no Google Colab

Clique no botão abaixo para abrir o notebook diretamente no Google Colab (versão usada na prova):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TUp5vLS8XWPExXPntbMbnBkEj54hlBtD?usp=sharing)

---

## 📁 Arquivos do repositório

- `LuizaHelena_CEP_Prova2.ipynb` – Notebook com todo o código da análise.
- `manufacturing_defect_dataset.csv` – Base de dados utilizada (defeitos de manufatura).
- `README.md` – Este arquivo, com explicação do projeto.

---

## 📊 Objetivo do projeto

O objetivo é aplicar **Controle Estatístico de Processo (CEP)** a um processo de manufatura, avaliando a estabilidade do processo por meio de cartas de controle por atributos:

- **P-Chart** – proporção de unidades não conformes por subgrupo;  
- **NP-Chart** – número de unidades defeituosas por subgrupo;  
- **C-Chart** – contagem de defeitos por subgrupo;  
- **U-Chart** – número de defeitos por unidade de inspeção.

As cartas utilizam um nível de significância `α = 0,0027`, correspondente a aproximadamente **3σ**.

---

## 🧮 Metodologia implementada no código

O script em Python realiza as seguintes etapas:

1. **Instalação e importação das bibliotecas**  
   Atualiza o `pip` e instala/importe: `pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels` e `seaborn`.

2. **Carrega**
