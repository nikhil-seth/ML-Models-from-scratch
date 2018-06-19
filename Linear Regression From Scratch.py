
# coding: utf-8

# In[1]:


#Importing Numpy for Scientific Calculation, Matplotlib for Plotting Data & sklearn.metrics for caluclating mean Error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as MAE


# In[2]:


#Defining Class LinReg
class LinReg:
    def __init__(self,rate,iteration):
        self.iteration=iteration
        self.rate=rate
        self.parameters=np.zeros((2,1))
    def cost_funct(self):
        self.cost=np.sum(np.square((self.prediction-self.Y))/(2*len(self.Y)))

    def predict(self,X=[],flag=False):
        if flag==False:
            self.prediction=np.dot(self.X,self.parameters)
        else:
            self.prediction=np.dot(X,self.parameters)
    def descent(self):
        q=(self.prediction-self.Y)	
        q=(np.dot(self.X.transpose(),q))
        self.parameters=self.parameters-(self.rate/len(self.Y))*q
    def fit(self,X_train,Y_train):
        X_train=X_train.reshape(len(X_train),1)
        b=np.ones(len(X_train)).reshape(X_train.shape)
        self.X=np.hstack((b,X_train))
        Y_train=Y_train.reshape(len(Y_train),1)
        self.Y=Y_train
        while(self.iteration):
            self.iteration-=1
            self.iterate()
        self.pr()

    def iterate(self):
        self.predict()
        self.cost_funct()
        print("Cost : ",self.cost)
        self.descent()
    def pr(self):
        print("\nParameters : ",self.parameters[0][0],"\t",self.parameters[1][0],"\nCost : ",self.cost)
    def ret_para(self):
        return self.parameters
    def predictions(self,X_test):
        X_test=X_test.reshape(len(X_test),1)
        b=np.ones(len(X_test)).reshape(X_test.shape)
        X_test=np.hstack((b,X_test))
        self.predict(X_test,True)
    def accuracy(self,X_test,Y_test):
        self.predictions(X_test)
        Y_test=Y_test.reshape(len(Y_test),1)
        print("\nMean Absolute Error : ",MAE(Y_test,self.prediction))


# In[3]:


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--',color='red',label='Linear Line Fit')


# In[4]:


#Importing of Data
print("Linear Regression With One Input Variable/Feature giving univariable Output\n")
data = np.genfromtxt('train.csv', delimiter=',')
data=data[1:,:]
X_train=data[:,0]
Y_train=data[:,1]
data = np.genfromtxt('train.csv', delimiter=',')
test = np.genfromtxt('test.csv', delimiter=',')
test=test[1:,:]
X_test=test[:,0]
Y_test=test[:,1]


# In[5]:


#Setting Learning Rate lr & no of iteration
lr=0.0005
iteration=100


# In[6]:


#Creating of obj of LinReg class
obj=LinReg(lr,iteration)
obj.fit(X_train,Y_train)


# In[7]:


#Plotting X_Train & Y_Train with Predicticted Linear Fitting
plt.scatter(X_train,Y_train,color='blue',label='Training Data')
plt.legend()
a=obj.ret_para()
abline(a[1,0],a[0,0])


# In[8]:


#Finding Object Accuracy Using Test Set
#X_test ,Y_test
obj.accuracy(X_test,Y_test)


# In[15]:


#Plotting X_test & Y_test
plt.scatter(X_train,Y_train,color='blue',label='Training Data')
plt.scatter(X_test,Y_test,c='g',label='Test Data')
plt.legend()
abline(a[1,0],a[0,0])

