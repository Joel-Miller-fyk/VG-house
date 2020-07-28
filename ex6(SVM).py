import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat 
from sklearn import svm

mat = loadmat(r"C:\Users\Fu'Y'u'ke\Desktop\machine-learning-ex1 (1)\machine-learning-ex6\ex6\ex6data1")
print(mat.keys())
X = mat['X']
y = mat['y']

def plotData(X,y):
	plt.figure(figsize=(8,5))
	plt.scatter(X[:,0],X[:,1],c=y.flatten(),cmap='rainbow',label='a')
	plt.xlabel('X1')
	plt.ylabel('X2')
	

plotData(X,y)
plt.show()
 
def plotBoundary(clf, X):
    '''plot decision bondary'''
    x_min, x_max = X[:,0].min()*1.2, X[:,0].max()*1.1
    y_min, y_max = X[:,1].min()*1.1,X[:,1].max()*1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)


models = [svm.SVC(C, kernel='linear') for C in [1, 100]]
clfs = [model.fit(X, y.ravel()) for model in models]


title = ['SVM Decision Boundary with C = {} (Example Dataset 1'.format(C) for C in [1, 100]]
for model,title in zip(clfs,title):
    plotData(X, y)
    plotBoundary(model, X)
    plt.title(title)
    
plt.show()


def gaussKernel(x1, x2, sigma):
    return np.exp(- ((x1 - x2) ** 2).sum() / (2 * sigma ** 2))

gaussKernel(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.)  # 0.32465246735834974

mat = loadmat(r"C:\Users\Fu'Y'u'ke\Desktop\machine-learning-ex1 (1)\machine-learning-ex6\ex6\ex6data2.mat")
X2 = mat['X']
y2 = mat['y']

plotData(X2, y2)

sigma = 0.1
gamma = np.power(sigma,-2.)/2
clf = svm.SVC(C=1, kernel='rbf', gamma=gamma)
modle = clf.fit(X2, y2.flatten())
plotData(X2, y2)
plotBoundary(modle, X2)
plt.show()

mat3 = loadmat(r"C:\Users\Fu'Y'u'ke\Desktop\machine-learning-ex1 (1)\machine-learning-ex6\ex6\ex6data3.mat")
X3, y3 = mat3['X'], mat3['y']
Xval, yval = mat3['Xval'], mat3['yval']
plotData(X3, y3)


Cvalues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
sigmavalues = Cvalues
best_pair, best_score = (0, 0), 0

for C in Cvalues:
    for sigma in sigmavalues:
        gamma = np.power(sigma,-2.)/2
        model = svm.SVC(C=C,kernel='rbf',gamma=gamma)
        model.fit(X3, y3.flatten())
        this_score = model.score(Xval, yval)
        if this_score > best_score:
            best_score = this_score
            best_pair = (C, sigma)
print('best_pair={}, best_score={}'.format(best_pair, best_score))
# best_pair=(1.0, 0.1), best_score=0.965

model = svm.SVC(C=1., kernel='rbf', gamma = np.power(.1, -2.)/2)
model.fit(X3, y3.flatten())
plotData(X3, y3)
plotBoundary(model, X3)
plt.show()



