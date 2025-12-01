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

2. **Carregamento dos dados**  
   Lê o arquivo `manufacturing_defect_dataset.csv` diretamente do GitHub usando `pandas.read_csv`, com separador `;`, e exibe as primeiras linhas para conferência.

3. **Formação dos subgrupos**  
   - Reorganiza o índice do DataFrame.  
   - Cria subgrupos de tamanho fixo (`subgroup_size = 30`).  
   - Agrega por subgrupo calculando:
     - número de unidades defeituosas (`defective_count`),
     - quantidade inspecionada (`inspected`),
     - contagem total de defeitos (`defects_count`).

4. **Definição da exposição**  
   Define a coluna `exposure` (número inspecionado) para uso na **Carta U**.

5. **Cálculo das cartas de controle**  
   Implementa quatro funções:
   - `compute_p_chart` – usa a distribuição Binomial para obter `p̄`, LCL e UCL da carta P.  
   - `compute_np_chart` – calcula `np̄`, LCL e UCL da carta NP (quando o tamanho amostral é constante).  
   - `compute_c_chart` – usa a distribuição de Poisson para limites da carta C.  
   - `compute_u_chart` – calcula `ū`, LCL e UCL da carta U, ajustando os limites conforme a exposição de cada subgrupo.

6. **Geração dos gráficos**  
   Cria gráficos com o `matplotlib` para:
   - P-Chart  
   - NP-Chart (se aplicável)  
   - C-Chart  
   - U-Chart  

   Cada gráfico mostra:
   - valores observados por subgrupo,  
   - linha da média (`p̄`, `np̄`, `c̄`, `ū`),  
   - limites de controle inferior e superior (LCL/UCL).

7. **Exportação dos resultados**  
   Salva o DataFrame agregado dos subgrupos em `spc_by_subgroup.csv`, contendo os dados usados nas cartas de controle.

---

## 🧾 Código completo (versão Python)

```python
!pip install --upgrade pip >/dev/null
!pip install pandas numpy matplotlib scipy statsmodels >/dev/null

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from statsmodels.stats.proportion import proportion_confint

import seaborn as sns
sns.set_theme(style="whitegrid")

# -------- CONFIGURAÇÕES ----------
CSV_URL = "https://raw.githubusercontent.com/luizahelenalucena/Prova-2---CEP/main/manufacturing_defect_dataset.csv"
col_defect_status = "DefectStatus"       # 0 = bom, 1 = defeituoso
col_defects_count = "SafetyIncidents"    # contagem de defeitos (c/u)
subgroup_size = 30
alpha = 0.0027

# -------- IMPORTAR DO GITHUB ----------
df = pd.read_csv(CSV_URL, sep=';')
print("Arquivo carregado — primeiras linhas:")
display(df.head())

# -------- CRIAR SUBGRUPOS ----------
df = df.reset_index(drop=True)
df["subgroup"] = (df.index // subgroup_size) + 1

# -------- AGRUPAR POR SUBGRUPO ----------
agg = df.groupby("subgroup").agg(
    defective_count=(col_defect_status, "sum"),
    inspected=(col_defect_status, "count"),
    defects_count=(col_defects_count, "sum")
).reset_index()
display(agg)

# Exposição para U-chart (padrão: inspecionados)
agg["exposure"] = agg["inspected"]

print("\nResumo por subgrupo:")
display(agg.head())

# -------- FUNÇÕES PARA CÁLCULO (USANDO SciPy / statsmodels) ----------
def compute_p_chart(agg, alpha=0.0027):
    total_def = agg["defective_count"].sum()
    total_ins = agg["inspected"].sum()
    p_bar = total_def / total_ins if total_ins>0 else 0.0

    LCL, UCL = [], []
    for n in agg["inspected"]:
        l = binom.ppf(alpha/2, int(n), p_bar)
        u = binom.ppf(1 - alpha/2, int(n), p_bar)
        LCL.append((l / n) if n>0 else 0.0)
        UCL.append((u / n) if n>0 else 0.0)

    agg = agg.copy()
    agg["p"] = agg["defective_count"] / agg["inspected"]
    agg["p_LCL"] = np.maximum(LCL, 0)
    agg["p_UCL"] = UCL
    return p_bar, agg

def compute_np_chart(agg, alpha=0.0027):
    if agg["inspected"].nunique() != 1:
        return None
    n = int(agg["inspected"].iloc[0])
    total_def = agg["defective_count"].sum()
    p_bar = total_def / (n * len(agg)) if n*len(agg)>0 else 0.0
    LCL = binom.ppf(alpha/2, n, p_bar)
    UCL = binom.ppf(1 - alpha/2, n, p_bar)
    np_bar = p_bar * n
    return np_bar, LCL, UCL

def compute_c_chart(agg, alpha=0.0027):
    c_bar = agg["defects_count"].mean()
    LCL = poisson.ppf(alpha/2, c_bar)
    UCL = poisson.ppf(1 - alpha/2, c_bar)
    LCL = 0 if np.isnan(LCL) else LCL
    UCL = 0 if np.isnan(UCL) else UCL
    return c_bar, LCL, UCL

def compute_u_chart(agg, alpha=0.0027):
    total_d = agg["defects_count"].sum()
    total_e = agg["exposure"].sum()
    u_bar = total_d / total_e if total_e>0 else 0.0

    LCL_list, UCL_list = [], []
    for exp in agg["exposure"]:
        mu = u_bar * exp
        l = poisson.ppf(alpha/2, mu)
        u = poisson.ppf(1 - alpha/2, mu)
        l = 0 if np.isnan(l) else l
        u = 0 if np.isnan(u) else u
        LCL_list.append((l / exp) if exp>0 else 0.0)
        UCL_list.append((u / exp) if exp>0 else np.inf)

    agg = agg.copy()
    agg["u"] = agg["defects_count"] / agg["exposure"]
    agg["u_LCL"] = LCL_list
    agg["u_UCL"] = UCL_list
    return u_bar, agg

# -------- GERAR E MOSTRAR GRÁFICOS ----------
# P-Chart
p_bar, agg_p = compute_p_chart(agg, alpha)
plt.figure(figsize=(12,5))
plt.plot(agg_p["subgroup"], agg_p["p"], marker='o', label="p (observado)")
plt.plot(agg_p["subgroup"], agg_p["p_UCL"], '--', label="UCL")
plt.plot(agg_p["subgroup"], agg_p["p_LCL"], '--', label="LCL")
plt.axhline(p_bar, color='green', label=f"p̄ = {p_bar:.4f}")
plt.title("P-Chart")
plt.xlabel("Subgrupo")
plt.ylabel("Proporção não conforme (p)")
plt.legend()
plt.ylim(bottom=0)
plt.show()

# NP-Chart (se aplicável)
np_res = compute_np_chart(agg, alpha)
if np_res:
    np_bar, LCL_np, UCL_np = np_res
    plt.figure(figsize=(12,5))
    plt.plot(agg["subgroup"], agg["defective_count"], marker='o', label="np (observado)")
    plt.axhline(np_bar, color='green', label=f"np̄ = {np_bar:.2f}")
    plt.axhline(LCL_np, linestyle="--", color='red', label='LCL/UCL (binom)')
    plt.axhline(UCL_np, linestyle="--", color='red')
    plt.title("NP-Chart")
    plt.xlabel("Subgrupo")
    plt.ylabel("Número de unidades defeituosas")
    plt.legend()
    plt.ylim(bottom=0)
    plt.show()
else:
    print("NP-Chart não gerado: n NÃO é constante entre subgrupos.")

# C-Chart
c_bar, LCL_c, UCL_c = compute_c_chart(agg, alpha)
plt.figure(figsize=(12,5))
plt.plot(agg["subgroup"], agg["defects_count"], marker='o', label="c (observado)")
plt.axhline(c_bar, color='green', label=f"c̄ = {c_bar:.2f}")
plt.axhline(LCL_c, linestyle="--", color='red', label='LCL/UCL (poisson)')
plt.axhline(UCL_c, linestyle="--", color='red')
plt.title("C-Chart")
plt.xlabel("Subgrupo")
plt.ylabel("Contagem de defeitos")
plt.legend()
plt.ylim(bottom=0)
plt.show()

# U-Chart
u_bar, agg_u = compute_u_chart(agg, alpha)
plt.figure(figsize=(12,5))
plt.plot(agg_u["subgroup"], agg_u["u"], marker='o', label="u (observado)")
plt.plot(agg_u["subgroup"], agg_u["u_UCL"], '--', label="UCL")
plt.plot(agg_u["subgroup"], agg_u["u_LCL"], '--', label="LCL")
plt.axhline(u_bar, color='green', label=f"ū = {u_bar:.4f}")
plt.title("U-Chart")
plt.xlabel("Subgrupo")
plt.ylabel("Defeitos por unidade (u)")
plt.legend()
plt.ylim(bottom=0)
plt.show()

# -------- SALVAR CSV DE SAÍDA (opcional) ----------
agg.to_csv("spc_by_subgroup.csv", index=False)
print("Arquivo 'spc_by_subgroup.csv' salvo no diretório atual do Colab.")

