import numpy as np
import pandas as pd
def normalize(X):
	pass
def initParam(X):			## Takes Input of n*m X where n is features & m is Training Eg no & returns n*1 object theta
	return np.zeros((X.shape[0],1))
def sigmoid(X):
	return 1/(1+np.exp(-X))
def compute(X,theta):		## Compute takes theta & X & calculates theta.T*X & applies sigmoid Activation function .
    a=np.dot(theta.T,X)		## 1,n*n,m return 1,m vector having results
    return sigmoid(a)

def cost(pred,Y,m):			## Returns cost by formula
    a=Y*np.log(pred)+(1-Y)*np.log(1-pred)
    return np.sum(a)/(-m)

def grad_ret(pred,Y,X,m):
	a=pred-Y				## Size : (1,m)
	return np.dot(X,a.T)/m	## X- Size :(n,m)

def updatation(theta,grad,alpha):
    theta=theta-alpha*grad
    return theta

def train_result(theta,costu):
	print("Training Complete\n-------------------------\nCost At Last Iteration :",costu)

def train(X,Y,learning_rate=0.01,num_iter=1000): 
	m=X.shape[1]
	theta=initParam(X)
	for i in range(num_iter):
		hx=compute(X,theta)
		if(i%100==0):
			print("Cost at "+str(i)+" Iteration :",cost(hx,Y,m))
		grad=grad_ret(hx,Y,X,m)
		theta=updatation(theta,grad,learning_rate)
	costu=cost(hx,Y,m)
	train_result(theta,costu)
	return theta

def predict(X,theta):
	a=compute(X,theta)
	b=a>0.5
	b.astype(int)
	return b

from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
X=data['data'].T.astype('Float64')
Y=data['target']
Y=Y.reshape(Y.shape[0],1).T
mean=np.mean(X,axis=1).reshape(-1,1)
std=np.std(X,axis=1).reshape(-1,1)
X=(X-mean)/std
X_train=X[:,:400]
Y_train=Y[:,:400]
X_test=X[:,400:]
Y_test=Y[:,400:]
theta=train(X_train,Y_train)
pred=predict(X_test,theta)
res=np.isclose(pred,Y_test).astype(int)
num_ones = (res == 1).sum()
print("% Predicted Correct : ",num_ones/res.shape[1]*100)
