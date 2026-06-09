Spam Email Classifier

A beginner machine-learning project that uses Python to classify emails as either spam or ham using TF-IDF text features and logistic regression.

Project overview

The aim of this project was to explore how natural language processing and supervised machine learning can be used to classify email text.

The project follows a basic data-science pipeline:

Load a labelled email dataset.
inspect and clean the data.
Convert text labels into numerical values.
Convert email text into TF-IDF features.
Split the data into training and test sets.
Train a logistic-regression classifier.
Evaluate its performance on unseen emails.
Technologies used
Python
pandas
scikit-learn
TF-IDF
Logistic regression
Dataset

The project uses a labelled spam-email dataset downloaded from Kaggle.

Each row contains:

text: the content of an email
label: whether the email is Spam or Ham

The dataset is not included in this repository. Download it from Kaggle and place the CSV file inside a local data folder.

Suggested folder structure:

spam-email-classifier/
├── data/
│   └── spam_Emails_data.csv
├── spam_classifier.py
├── requirements.txt
├── .gitignore
└── README.md
Data preparation

The following preparation steps are included:

Removal of records with missing email text
Conversion of text labels into numerical values
Ham becomes 0
Spam becomes 1
Inspection of the class distribution
Conversion of raw email text into numerical TF-IDF features

TF-IDF gives greater importance to terms that are useful within a particular email while reducing the importance of terms that appear very frequently across the dataset.

The vocabulary is limited to 5,000 features to reduce memory and processing requirements.

Model

The project uses logistic regression.

Although logistic regression has “regression” in its name, it can be used as a classification algorithm. It was selected because it provides a relatively simple and efficient baseline for high-dimensional text data.

The data is divided into:

80% training data
20% test data

A fixed random state is used so that the split can be reproduced.

Evaluation

The first version of the model achieved approximately 97% accuracy on the test data.

However, accuracy alone does not describe all types of classification error. Future evaluation should also include:

Precision: when the model predicts spam, how often it is correct
Recall: how much of the actual spam the model successfully detects
F1 score: a balance between precision and recall
Confusion matrix: a breakdown of correct and incorrect predictions

These metrics are important because a spam filter can make two different types of mistake:

Sending a legitimate email to the spam folder
Allowing a spam email into the inbox
Improvements identified

While reviewing the first version, I identified several ways to make the project more robust.

Preventing data leakage

In the original version, TF-IDF was fitted before the train/test split. This meant that the vectorizer had seen text from the test set while building its vocabulary.

A more reliable approach is to:

Split the raw text into training and test sets.
Fit TF-IDF only on the training text.
Transform the test text using the fitted training vectorizer.

A scikit-learn pipeline can keep the preprocessing and model together and help prevent this type of leakage.

Stratified splitting

Using stratify=y would help preserve a similar proportion of spam and ham emails in both the training and test sets.

Duplicate checking

The dataset should be checked for duplicate email text. Identical emails appearing in both the training and test sets could make the model appear more effective than it would be on genuinely unseen messages.

Wider evaluation

The model should be assessed using precision, recall, F1 score and a confusion matrix rather than accuracy alone.

Model development

Further work could include:

Testing different TF-IDF feature limits
Including word pairs using n-grams
Comparing logistic regression with other suitable classifiers
Using cross-validation
Tuning model parameters
Creating a function that classifies new email text
Examining which terms have the strongest influence on predictions
Installation

Clone the repository:

git clone https://github.com/YOUR-USERNAME/spam-email-classifier.git
cd spam-email-classifier

Create and activate a virtual environment.

On Windows:

python -m venv .venv
.venv\Scripts\activate

On macOS or Linux:

python3 -m venv .venv
source .venv/bin/activate

Install the required packages:

pip install -r requirements.txt
Requirements

The requirements.txt file should contain:

pandas
scikit-learn
Running the project

Place the dataset inside the data folder and make sure the code uses a relative path:

df = pd.read_csv("data/spam_Emails_data.csv")

Run the program:

python spam_classifier.py
What I learned

This project developed my understanding of:

Loading and inspecting tabular data with pandas
Handling missing data
Converting categorical labels into numerical targets
Converting text into machine-readable features
Training a supervised classification model
Separating training and test data
Evaluating predictions on unseen examples
The limitations of relying only on accuracy
The importance of avoiding data leakage
The value of reproducible and clearly documented code

The project also showed me that building a model is only one part of the data-science process. Careful data preparation, validation and interpretation are essential for producing reliable results.

Limitations

This is an introductory learning project rather than a production-ready spam filter.

Current limitations include:

The model has only been tested on one dataset.
The original score may have been influenced by the preprocessing sequence.
Email text may contain patterns that do not represent newer or different types of spam.
Accuracy does not show the full impact of false positives and false negatives.
The model does not currently account for attachments, sender details or other email metadata.
Author

Lily Chen

Aspiring data scientist with an interest in Python, machine learning and using data to solve real-world problems.
