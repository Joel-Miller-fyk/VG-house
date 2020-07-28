#coding=utf-8
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

mat = loadmat(r"C:\Users\Fu'Y'u'ke\Desktop\ML_data\machine-learning-ex8\ex8\ex8_movies.mat")
print(mat.keys())
Y, R = mat['Y'], mat['R']
nm, nu = Y.shape 
nf = 100
print(Y.shape, R.shape)
# ((1682, 943), (1682, 943))

print(Y[0].sum() / R[0].sum())  
# "Visualize the ratings matrix"
fig = plt.figure(figsize=(8,8*(1682./943.)))
plt.imshow(Y, cmap='rainbow')
plt.colorbar()
plt.ylabel('Movies (%d)'%nm,fontsize=20)
plt.xlabel('Users (%d)'%nu,fontsize=20)
plt.show()

#collaborative filtering leaning algorithm

mat = loadmat(r"C:\Users\Fu'Y'u'ke\Desktop\ML_data\machine-learning-ex8\ex8\ex8_movieParams.mat")
X = mat['X']
Theta = mat['Theta']
nu = int(mat['num_users'])
nm = int(mat['num_movies'])
nf = int(mat['num_features'])
# For now, reduce the data set size so that this runs faster
nu = 4; nm = 5; nf = 3
X = X[:nm,:nf]
Theta = Theta[:nu,:nf]
Y = Y[:nm,:nu]
R = R[:nm,:nu]

print(X.shape, Theta.shape)

def serialize(X, Theta):
    '''unfold parameter'''
    return np.r_[X.flatten(),Theta.flatten()]

def deserialize(seq, nm, nu, nf):
    '''extract parameter'''
    return seq[:nm*nf].reshape(nm, nf), seq[nm*nf:].reshape(nu, nf)

def cofiCostFunc(params, Y, R, nm, nu, nf, l=0):
    """
   
    """
    X, Theta = deserialize(params, nm, nu, nf)
    
    
    error = 0.5*np.square((X@Theta.T - Y)*R).sum()
    reg1 = .5*l*np.square(Theta).sum()
    reg2 = .5*l*np.square(X).sum()
    
    return error + reg1 +reg2


print(cofiCostFunc(serialize(X,Theta),Y,R,nm,nu,nf),cofiCostFunc(serialize(X,Theta),Y,R,nm,nu,nf,1.5))
# (22.224603725685675, 31.34405624427422)

#???Collaborative filtering gradient

def cofiGradient(params, Y, R, nm, nu, nf, l=0):
    """
    计算X和Theta的梯度，并序列化输出。
    """
    X, Theta = deserialize(params, nm, nu, nf)
    
    X_grad = ((X@Theta.T-Y)*R)@Theta + l*X
    Theta_grad = ((X@Theta.T-Y)*R).T@X + l*Theta
    
    return serialize(X_grad, Theta_grad)
def checkGradient(params, Y, myR, nm, nu, nf, l = 0.):
    """
    Let's check my gradient computation real quick
    """
    print('Numerical Gradient \t cofiGrad \t\t Difference')
    
    # 分析出来的梯度
    grad = cofiGradient(params,Y,myR,nm,nu,nf,l)
    
    # 用 微小的e 来计算数值梯度。
    e = 0.0001
    nparams = len(params)
    e_vec = np.zeros(nparams)

    # Choose 10 random elements of param vector and compute the numerical gradient
    # 每次只能改变e_vec中的一个值，并在计算完数值梯度后要还原。
    for i in range(10):
        idx = np.random.randint(0,nparams)
        e_vec[idx] = e
        loss1 = cofiCostFunc(params-e_vec,Y,myR,nm,nu,nf,l)
        loss2 = cofiCostFunc(params+e_vec,Y,myR,nm,nu,nf,l)
        numgrad = (loss2 - loss1) / (2*e)
        e_vec[idx] = 0
        diff = np.linalg.norm(numgrad - grad[idx]) / np.linalg.norm(numgrad + grad[idx])
        print('%0.15f \t %0.15f \t %0.15f' %(numgrad, grad[idx], diff))

print("Checking gradient with lambda = 0...")
checkGradient(serialize(X,Theta), Y, R, nm, nu, nf)
print("\nChecking gradient with lambda = 1.5...")
checkGradient(serialize(X,Theta), Y, R, nm, nu, nf, l=1.5)

#Learning movie recommendations
movies = []  # 包含所有电影的列表
with open(r"C:\Users\Fu'Y'u'ke\Desktop\ML_data\machine-learning-ex8\ex8\movie_ids.txt",'r', encoding='ISO-8859-1') as f:
    for line in f:
#         movies.append(' '.join(line.strip().split(' ')[1:]))
        movies.append(' '.join(line.strip().split(' ')[1:]))

my_ratings = np.zeros((1682,1))

my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(my_ratings[i], movies[i])

mat = loadmat(r"C:\Users\Fu'Y'u'ke\Desktop\ML_data\machine-learning-ex8\ex8\ex8_movies.mat")
Y, R = mat['Y'], mat['R']
Y.shape, R.shape

Y = np.c_[Y, my_ratings]  # (1682, 944)
R = np.c_[R, my_ratings!=0]  # (1682, 944)
nm, nu = Y.shape

nf = 10 # 我们使用10维的特征向量

def normalizeRatings(Y, R):
    """
    The mean is only counting movies that were rated
    """
    Ymean = (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1,1)
#     Ynorm = (Y - Ymean)*R  # 这里也要注意不要归一化未评分的数据
    Ynorm = (Y - Ymean)*R  # 这里也要注意不要归一化未评分的数据
    return Ynorm, Ymean

Ynorm, Ymean = normalizeRatings(Y, R)
print(Ynorm.shape, Ymean.shape)
# ((1682, 944), (1682, 1))

X = np.random.random((nm, nf))
Theta = np.random.random((nu, nf))
params = serialize(X, Theta)
l = 10

import scipy.optimize as opt
res = opt.minimize(fun=cofiCostFunc,
                   x0=params,
                   args=(Y, R, nm, nu, nf, l),
                   method='TNC',
                   jac=cofiGradient,
                   options={'maxiter': 100})

ret = res.x

# import scipy.optimize as opt
# ret = opt.fmin_cg(cofiCostFunc, x0=params, fprime=cofiGradient, 
#                   args=(Y, R, nm, nu, nf, l), maxiter=100)

fit_X, fit_Theta = deserialize(ret, nm, nu, nf)
# 所有用户的剧场分数矩阵
pred_mat = fit_X @ fit_Theta.T
# 最后一个用户的预测分数， 也就是我们刚才添加的用户
pred = pred_mat[:,-1] + Ymean.flatten()
pred_sorted_idx = np.argsort(pred)[::-1] # 排序并翻转，使之从大到小排列

print("Top recommendations for you:")
for i in range(10):
    print('Predicting rating %0.1f for movie %s.' \
          %(pred[pred_sorted_idx[i]],movies[pred_sorted_idx[i]]))

print("\nOriginal ratings provided:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for movie %s.'% (my_ratings[i],movies[i]))
'''   
Top recommendations for you:
Predicting rating 8.5 for movie Titanic (1997).
Predicting rating 8.4 for movie Star Wars (1977).
Predicting rating 8.3 for movie Shawshank Redemption, The (1994).
Predicting rating 8.3 for movie Schindler's List (1993).
Predicting rating 8.3 for movie Raiders of the Lost Ark (1981).
Predicting rating 8.2 for movie Good Will Hunting (1997).
Predicting rating 8.1 for movie Empire Strikes Back, The (1980).
Predicting rating 8.1 for movie Wrong Trousers, The (1993).
Predicting rating 8.0 for movie Casablanca (1942).
Predicting rating 8.0 for movie Usual Suspects, The (1995).

Original ratings provided:
Rated 4 for movie Toy Story (1995).
Rated 3 for movie Twelve Monkeys (1995).
Rated 5 for movie Usual Suspects, The (1995).
Rated 4 for movie Outbreak (1995).
Rated 5 for movie Shawshank Redemption, The (1994).
Rated 3 for movie While You Were Sleeping (1995).
Rated 5 for movie Forrest Gump (1994).
Rated 2 for movie Silence of the Lambs, The (1991).
Rated 4 for movie Alien (1979).
Rated 5 for movie Die Hard 2 (1990).
Rated 5 for movie Sphere (1998).
'''

