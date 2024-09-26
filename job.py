import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('./datasets/SMSSpamCollection', sep='\t', header=None)
df.columns = ['target', 'text']

df['target'] = df['target'].apply(lambda x: 1 if x == 'spam' else 0)

balance = df.target.value_counts(normalize=True)
print("Balanceamento da base: ", balance)

X = df['text']
y= df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train_indices = X_train.index
X_test_indices = X_test.index

X_train_text = df['text'].iloc[X_train_indices]
X_test_text = df['text'].iloc[X_test_indices]

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)

#Isolando apenas a probabilidade da classe positiva (ex: é spam)
y_proba_1 = y_proba[:,1]

df_results = pd.DataFrame({
  'y_proba_1': y_proba_1,
  'y_test': y_test
})

lista = df_results['y_proba_1'].tolist()

# Reconstruir o resumo dos bins de probabilidade
bins = np.linspace(0, 1, 11)
df_results['proba_bin'] = pd.cut(df_results['y_proba_1'], bins=10, include_lowest=True)

bin_summary = df_results.groupby('proba_bin')['y_test'].agg(['count', 'sum']).reset_index()
bin_summary['percentage'] = bin_summary['sum'] / bin_summary['count']

# Criar um dataframe resumido com as contagens e porcentagens
bin_summary_melted = pd.melt(bin_summary[['proba_bin', 'count', 'sum']], id_vars='proba_bin', 
                             value_vars=['count', 'sum'], var_name='Tipo', value_name='Contagem')

# Ajustar os nomes para 'População geral' e 'População positiva'
bin_summary_melted['Tipo'] = bin_summary_melted['Tipo'].replace({'count': 'População geral', 'sum': 'População positiva'})

# Gráfico de barras lado a lado
fig, ax = plt.subplots(figsize=(8, 6))

# Plotar as barras lado a lado
bin_summary_pivot = bin_summary_melted.pivot(index='proba_bin', columns='Tipo', values='Contagem')
bin_summary_pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')

ax.set_title('Proporção de amostras positivas por bin de probabilidade')
ax.set_xlabel('Bins de probabilidade')
ax.set_ylabel('Contagem')
plt.xticks(rotation=45)
ax.grid(True)

plt.tight_layout()
plt.show()