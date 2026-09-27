import numpy as np
import sys
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import DecisionTree

def main():
    try:
        data = load_breast_cancer()
        X, y = data.data, data.target
    except Exception as e:
        print(f"Data loading failed: {e}")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("[main.py] : Initializing C-backed Decision Tree (maxDepth=5, minSamplesSplit=2)")
    clf = DecisionTree(maxDepth=5, minSamplesSplit=2)
    
    print("[main.py] : Fitting model via C engine")
    clf.fit(X_train, y_train)
    
    print("[main.py] : Executing predictions via C engine")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
