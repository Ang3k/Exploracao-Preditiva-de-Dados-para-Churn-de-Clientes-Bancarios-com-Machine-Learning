# Churn Prediction - Previsão de Atrição de Clientes

## Descrição do Projeto
Este projeto tem como objetivo prever a **atrition** (perda de clientes) em um banco, utilizando técnicas de Machine Learning para criar um modelo preditivo. Diferentes algoritmos de classificação foram testados e otimizados para melhorar a precisão da previsão. O modelo final visa identificar clientes com maior risco de deixar o serviço, permitindo que ações preventivas sejam tomadas.

## Pré-Processamento dos Dados
1. **Remoção de Colunas Desnecessárias**: Foram removidas colunas como `CLIENTNUM` e variáveis do Naive Bayes que não seriam usadas no treinamento.
2. **Tratamento de Valores Faltantes**: Valores "Unknown" foram convertidos para `np.NaN`, e técnicas como `ffill` foram usadas para preencher as lacunas.
3. **Remoção de Outliers**: Utilizou-se o método de John Tukey (IQR) para identificar e remover outliers em colunas como `Credit_Limit`, `Total_Trans_Amt`, entre outras.
4. **Transformação de Variáveis**: Aplicamos `get_dummies()` para variáveis binárias e `OrdinalEncoder` para variáveis com ordens naturais (ex: `Education_Level`, `Income_Category`).

## Algoritmos Utilizados
Os principais modelos de Machine Learning testados foram:
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **XGBoost**

Cada modelo foi avaliado utilizando **Validação Cruzada** com 10 dobras para garantir uma avaliação precisa. Os três melhores algoritmos passaram por uma fase de ajuste fino (tuning) de hiperparâmetros.

## Tuning de Hiperparâmetros
Os hiperparâmetros foram ajustados utilizando `GridSearchCV` nos seguintes modelos:
1. **Decision Tree Classifier**: Ajuste da profundidade máxima, critério de divisão, e outros parâmetros.
2. **Random Forest Classifier**: Ajuste do número de árvores, amostras mínimas por folha, e critério de divisão.
3. **XGBoost**: Ajuste da taxa de aprendizado, profundidade da árvore, subsample, e colsample_bytree.

## Avaliação de Performance
A métrica principal utilizada foi a **precisão**. Após o tuning dos modelos, os seguintes resultados foram obtidos:

- **DecisionTreeClassifier**: 94.332% de precisão
- **RandomForestClassifier**: 96.303% de precisão
- **XGBoost**: 97.243% de precisão

## Visualizações
- **Importância das Features**: Gráficos de barras mostrando a importância das variáveis para cada modelo.
- **Árvores de Decisão**: Visualizações gráficas das árvores de decisão para DecisionTree, RandomForest e XGBoost.

## Exportação dos Modelos
Os modelos otimizados foram salvos utilizando `joblib` e podem ser reutilizados sem a necessidade de retreinamento. Para carregar os modelos treinados:

```python
from joblib import load
optimized_dtc = load('optimized_dtc.joblib')
optimized_rfc = load('optimized_rfc.joblib')
optimized_xgb = load('optimized_xgb.joblib')
