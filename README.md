import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ── Load & clean ───────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\lilyz\OneDrive\Documents\Kaggle datasets\spam_Emails_data.csv")
df = df.dropna(subset=['text'])
print(df.shape)

# ── Convert labels to numbers ──────────────────────────────────
df['label_num'] = df['label'].map({'Ham': 0, 'Spam': 1})
print(df['label_num'].value_counts())

# ── Convert text to numbers ────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label_num']

# ── Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)

# ── Train & score ──────────────────────────────────────────────
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # prints your accuracy
