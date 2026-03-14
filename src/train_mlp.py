# train_mlp.py
from sklearn.neural_network import MLPClassifier

def train_mlp(X_train, y_train):
    """
    Train a simple neural network for multi-class classification.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(256,128),
        activation="relu",
        solver="adam",
        max_iter=20,   # TF-IDF features are large, don't train too long
        random_state=42,
        verbose=True
    )
    model.fit(X_train, y_train)
    return model