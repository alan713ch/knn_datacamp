# k-nn and preprocessing

# summon the panda
import pandas as pd
import matplotlib.pyplot as plt

# specify the style
plt.style.use('ggplot')  # same as R!

# save the data in a dataframe - from a csv with ; separator
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')

X = df.drop('quality', 1).values  # create a new data frame without the target variable
y1 = df['quality'].values  # save the target variable in a new array

pd.DataFrame.hist(df, figsize=[15, 15])  # plot the variables
# plt.show()

# what is a good quality and bad quality? if Quality <= 5 is bad!
y = y1 <= 5
# print(y)

# This plots the quality of the wines
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.hist(y1)  # this shows what's the quality of the wines, it goes from 3 to 8, and anything above 5 is good quality
plt.xlabel('original target value')
plt.ylabel('count')
plt.subplot(1, 2, 2)
plt.hist(y)  # this plots how many are good and how many are bad (with 0 being good, and 1 being bad)
plt.xlabel('aggregated target value')
plt.show()

# starting the knn process
# we are going to split the data in training (where the model develops) and testing (where we check accuracy)
# we summon the splitter
from sklearn.cross_validation import train_test_split

# we create our four arrays: X and y, train and test. 20% will go to test size,
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# we now build the model
# we summon the packages!
from sklearn import neighbors, linear_model

knn = neighbors.KNeighborsClassifier(n_neighbors=5)  # the model will compare with 5 neighbor data points
knn_model_1 = knn.fit(X_train, y_train)  # model will be created with the train data, using the fit function
print('k-NN accuracy for test set %f' % knn_model_1.score(X_test, y_test))  # default mode for knn is ACCURACY

# other metrics of how good this is working?
from sklearn.metrics import classification_report

y_true, y_pred = y_test, knn_model_1.predict(
    X_test)  # we are using the predict function with X_test to obtain a y_pred and compare against it y_true (the same as y_test for X_test)
print(classification_report(y_true, y_pred))

# time to scale and standarize
from sklearn.preprocessing import scale

Xs = scale(X)  # gotta scale our X!
Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, y, test_size=0.2,
                                                        random_state=42)  # again, splitting in 80% train 20% test but on the scaled data
knn_model_2 = knn.fit(Xs_train, ys_train)
print('k-NN accuracy for scaled test set %f' % knn_model_2.score(Xs_test, ys_test))
print('k-NN accuracy for scaled training set %f' % knn_model_2.score(Xs_train, ys_train))
ys_true, ys_pred = ys_test, knn_model_2.predict(Xs_test)
print(classification_report(ys_true, ys_pred))
