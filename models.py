from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Linear SVM": LinearSVC(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    }
