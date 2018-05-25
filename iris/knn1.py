from sklearn.datasets        import load_iris
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test =  train_test_split(iris.data, iris.target, test_size=0.1)


knn = KNeighborsRegressor(n_neighbors=1)

knn.fit(x_train, y_train)

y_result = knn.predict(x_test)


print "accuracy : {}".format(accuracy_score(y_test, y_result))
