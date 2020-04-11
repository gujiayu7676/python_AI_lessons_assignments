import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import tree
from sklearn.model_selection  import train_test_split
from pylab import *
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']

def load_data():
    df = pd.read_csv('data.csv')
    x = df[['AT', 'V', 'AP', 'RH']]
    y = df['PE']
    x = np.array(x)
    y = np.array(y)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12345)
    return x_train, x_test, y_train, y_test


def plot_svr(x_train, x_test, y_train, y_test):
    svr1 = svm.LinearSVR()
    svr1.fit(x_train,y_train)
    score1 = svr1.score(x_test,y_test)
    y_pred = svr1.predict(x_test)
    plt.figure('SVR真实值-预测值')
    plt.scatter(y_test,y_pred)
    plt.plot([y_test.min(),y_test.max()],[y_pred.min(),y_test.max()],'k--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.show()
    print("coef:{0},intercept:{1}".format(svr1.coef_,svr1.intercept_))
    print("svr_score:{0}".format(score1))

def plot_dtr(x_train, x_test, y_train, y_test):
    dt = tree.DecisionTreeRegressor(random_state=1)
    dt.fit(x_train,y_train)
    score1 = dt.score(x_test,y_test)
    y_pred = dt.predict(x_test)
    plt.figure("DTR真实值-预测值")
    plt.scatter(y_test,y_pred)
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_test.max()], 'k--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.show()
    print("dtr_score:{0}".format(score1))



if __name__ == "__main__":
   x_train, x_test, y_train, y_test = load_data()
   plot_svr(x_train, x_test, y_train, y_test)
   plot_dtr(x_train, x_test, y_train, y_test)