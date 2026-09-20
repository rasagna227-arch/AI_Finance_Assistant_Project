import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA = [
    ("swiggy dinner", "Food"), ("restaurant lunch", "Food"),
    ("pizza order", "Food"), ("grocery shopping", "Food"),
    ("uber ride", "Transport"), ("ola cab", "Transport"),
    ("bus ticket", "Transport"), ("petrol fuel", "Transport"),
    ("amazon shoes", "Shopping"), ("clothes purchase", "Shopping"),
    ("shopping mall", "Shopping"), ("online shopping", "Shopping"),
    ("electricity bill", "Bills"), ("mobile bill", "Bills"),
    ("internet bill", "Bills"), ("water bill", "Bills"),
    ("movie ticket", "Entertainment"), ("netflix subscription", "Entertainment"),
    ("concert ticket", "Entertainment"), ("game subscription", "Entertainment"),
    ("doctor consultation", "Health"), ("medicine purchase", "Health"),
    ("hospital bill", "Health"), ("pharmacy", "Health"),
    ("college fees", "Education"), ("course fee", "Education"),
    ("books purchase", "Education"), ("online course", "Education"),
    ("salary", "Other"), ("miscellaneous payment", "Other"),
]

df = pd.DataFrame(DATA, columns=["description", "category"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(df["description"], df["category"])

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/expense_classifier.joblib")

print("Model trained and saved to model/expense_classifier.joblib")
print("Classes:", sorted(df["category"].unique()))
