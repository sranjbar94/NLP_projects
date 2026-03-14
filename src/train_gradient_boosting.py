# train_gradient_boosting.py
from sklearn.ensemble import HistGradientBoostingClassifier

def train_gradient_boosting(X_train, y_train):
    """
    Train a gradient boosting classifier for multi-class classification.
    """
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_depth=7,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
