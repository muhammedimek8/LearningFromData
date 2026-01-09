import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack

from preprocessing import clean_text
from features import custom_features
from models import get_models

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Preprocessing
df["text"] = df["text"].apply(clean_text)

X = df["text"].values
y = df["label"].values

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Custom features
X_custom = custom_features(X)

# Combine features
X_final = hstack([X_tfidf, X_custom])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

models = get_models()

for name, model in models.items():
    print("\n==============================")
    print(f"Model: {name}")

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="f1_macro"
    )
    print("5-Fold CV F1 Score:", cv_scores.mean())

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
