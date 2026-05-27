import pandas as pd                                        # load the pandas library for working with data tables
from sklearn.feature_extraction.text import TfidfVectorizer  # tool to convert email text into numbers
from sklearn.model_selection import train_test_split      # tool to split data into training and test sets
from sklearn.linear_model import LogisticRegression       # the model we're going to train

# ── Load & clean ───────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\lilyz\OneDrive\Documents\Kaggle datasets\spam_Emails_data.csv")  # read the CSV file into a table called df
df = df.dropna(subset=['text'])                           # remove the 2 rows where the email text was missing
print(df.shape)                                           # print how many rows and columns we have

# ── Convert labels to numbers ──────────────────────────────────
df['label_num'] = df['label'].map({'Ham': 0, 'Spam': 1}) # create a new column: Ham becomes 0, Spam becomes 1
print(df['label_num'].value_counts())                     # print how many 0s and 1s we have

# ── Convert text to numbers ────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000)           # set up the TF-IDF tool, keeping only the top 5000 most useful words
X = vectorizer.fit_transform(df['text'])                  # convert every email into a row of 5000 numbers (the word scores)
y = df['label_num']                                       # y is the answer column — 0 for ham, 1 for spam

# ── Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                                                 # split both the emails (X) and labels (y) together
    test_size=0.2,                                        # use 20% of emails for testing, 80% for training
    random_state=42                                       # fix the random shuffle so results are the same every run
)
print(X_train.shape)                                      # print the size of the training set (should be ~155,080 rows)
print(X_test.shape)                                       # print the size of the test set (should be ~38,770 rows)

# ── Train & score ──────────────────────────────────────────────
model = LogisticRegression()                              # create a blank logistic regression model (not trained yet)
model.fit(X_train, y_train)                               # train the model — show it the emails and correct answers so it learns
print(model.score(X_test, y_test))                        # test on unseen emails and print accuracy (you got 0.97 — excellent)
