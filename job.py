import pandas as pd
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