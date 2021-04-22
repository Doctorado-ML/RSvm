from rsvm import RSvm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X, y = load_iris(return_X_y=True)
# make a binary problem
y[y == 2] = 1
scaler = StandardScaler()
X = scaler.fit_transform(X)
xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, random_state=0, test_size=0.3, shuffle=True
)
clf = RSvm(random_state=0, kernel="sigmoid", max_iter=3000)
accuracy = clf.fit(xtrain, ytrain).score(xtest, ytest)
print(f"Accuracy: {accuracy:.3f}")
