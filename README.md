# Análise de Dados de Clientes e Previsão de Cancelamento de Cartões de Crédito

## Visão Geral do Projeto

Este projeto tem como objetivo analisar um conjunto de dados de clientes de um banco para prever o cancelamento de cartões de crédito. O dataset contém informações cruciais, como idade, limite de crédito, frequência de transações e status de relacionamento, permitindo uma análise abrangente do comportamento dos clientes.

## Técnicas Utilizadas

### 1. [Tratamento de Dados](https://angejesufern.wixsite.com/angel-mansilla/cópia-data-visualization-and-cleaning)

- **Limpeza do Dataset:**
  - Eliminação de colunas irrelevantes que não contribuem para a análise.
  - Identificação e substituição de valores nulos, transformando entradas como "Unknown" em `np.nan` para facilitar o tratamento.
  - Remoção de outliers utilizando o método de John Tukey, que se baseia em quartis para determinar limites superior e inferior.

- **Criação de Variáveis Dummy:**
  - Aplicação de `pd.get_dummies` em variáveis categóricas como `Gender` e `Marital_Status`, com `drop_first=True` para evitar a armadilha da multicolinearidade.
  - Para outras variáveis categóricas com ordem implícita, como tipo de cartão, foi utilizada a codificação numérica para simular a ordem.

### 2. [Análises Realizadas](https://angejesufern.wixsite.com/angel-mansilla/eda)

- **Análise Descritiva:**
  - Estatísticas descritivas para entender a distribuição dos dados, como médias, medianas e desvios padrão.
  - Análise da correlação entre variáveis, visando identificar padrões significativos.

- **Visualizações:**
  - Criação de gráficos de barras e boxplots para visualizar a distribuição de variáveis e a presença de outliers.
  - Geração de heatmaps para observar correlações entre atributos, destacando quais características impactam mais no cancelamento.

### 3. [Modelagem Preditiva](https://angejesufern.wixsite.com/angel-mansilla/machine-learning-and-data-cleaning)

- **Separação de Features e Target:**
  - As variáveis foram divididas em `features` (X) e a variável de previsão (`target` - Y) que indica se o cliente cancelou o cartão.

- **Normalização:**
  - Uso de `StandardScaler` para normalizar os dados, garantindo que diferenças nas escalas das variáveis não afetassem o desempenho dos modelos.

- **Modelos de Machine Learning:**
  - Implementação e comparação de três modelos:
    - **Decision Tree Classifier**
    - **Random Forest Classifier**
    - **XGBoost**
  - Realização de validação cruzada para garantir a robustez dos resultados e prevenir overfitting.

## Resultados

- **Acurácia dos Modelos:**
  - Decision Tree: 93.06% de precisão original, aumentou para 94.33% após tunagem.
  - Random Forest: 96.04% de precisão original, aumentou para 96.149% após tunagem.
  - XGBoost: 96.82% de precisão original, aumentou para 97.43% após tunagem.

- **Insights Relevantes:**
  - Identificação de perfis de clientes com maior risco de cancelamento, como aqueles com limites de crédito baixos e baixa frequência de uso.
  - Análise de padrões de uso por gênero, destacando que mulheres tendem a utilizar mais seus cartões, mas possuem limites de crédito menores.
  - Clientes com níveis de educação mais altos apresentaram maior propensão ao cancelamento.

### 4. Visualização das Árvores de Decisão

- **Visualização das Árvores:**
  - Uso da função `plot_tree` do Scikit-learn para visualizar a estrutura das árvores de decisão, permitindo uma interpretação clara das regras de decisão.
  - Visualização das árvores de XGBoost utilizando a função apropriada para este modelo, permitindo uma comparação visual entre os modelos.
