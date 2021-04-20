from rsvm import RSvm
from sklearn.datasets import load_wine


X, y = load_wine(return_X_y=True)
clf = RSvm(random_state=0)
accuracy = clf.fit(X, y).score(X, y)
print(f"Accuracy: {accuracy:.3f}")