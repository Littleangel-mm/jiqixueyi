import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



if __name__ == "__main__":
    dataset =  load_iris()
    x = dataset.data
    y = dataset.target
    Xd_train,xd_test,y_train,y_test = train_test_split(x,y,random_state=14)
    clf = RandomForestClassifier(max_depth=2,random_state=0)
    clf = clf.fit(Xd_train ,y_train)
    y_predicted = clf.predict(xd_test)
    accuracy = np.mean(y_predicted==y_test) * 100
    print("y_test",y_test)
    print("y_predicted",y_predicted)
    print('accuracy',accuracy)
    