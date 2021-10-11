import pandas as pd
#import keras
#import matplotlib.pyplot as plt
import numpy as np
#import scipy
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
#from matplotlib import style
#import pickle
'''
data = pd.read_csv('student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)


fit = 0
for _ in range(60):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > fit:
        fit = acc
        with open ("studmod.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studmod.pickle", "rb")
linear = pickle.load(pickle_in)

#print("coeffcient: ", + linear.coef_)
#print("intercept: ", + linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

a = 'G1'
style.use('ggplot')
plt.scatter(data[a], data['G3'])
plt.xlabel(a)
plt.ylabel("final G")
plt.show()
'''
data2 = pd.read_csv("car.data")
#print(data2.head())

o = preprocessing.LabelEncoder()
buying = o.fit_transform(list(data2['buying']))
maint = o.fit_transform(list(data2['maint']))
doors = o.fit_transform(list(data2['doors']))
persons = o.fit_transform(list(data2['persons']))
lug_boot = o.fit_transform(list(data2['lug_boot']))
safety = o.fit_transform(list(data2['safety']))
cls = o.fit_transform(list(data2['class']))

predict = "class"

x = list(zip(buying,maint,doors,persons,lug_boot,safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors = 13)
model.fit(x_train, y_train)
acc = model.score(x_test,y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

'''
for x in range(len(predicted)):
    print("predicted:", names[predicted[x]] ,"actual;", names[y_test[x]], "data:", x_test[x])
    n = model.kneighbors([x_test[x]], 13, True)
    print("N:", n)
'''

