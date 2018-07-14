import numpy as np
class Neural_Network(): ## 2 Layer Neural Network with 1 hidden layer of 5 neurons
	def __init__(self,X,Y,learning_rate=0.1,num_iter=10000):
		np.random.seed(1)
		self.X=X	# (n,m)
		self.Y=Y	#  (k,m)
		self.lr=learning_rate
		self.num_iter=num_iter

	def sigmoid(self,X):
		return 1/(1+np.exp(-X))

	def layer_size(self):
		self.n_x=self.X.shape[0]
		self.n_y=self.Y.shape[0]
		self.n_h=5
		self.m=self.X.shape[1]

	def initParams(self,n_x,n_y,n_h): # Initialize weights/Synapse . Cond :Layer Size must be called before calling this function
		w1=np.random.randn(n_h,n_x)*0.01 #(n,5)
		w2=np.random.randn(n_y,n_h)*0.01 #(5,k)
		self.W=[w1,w2] # 2D list
		b1=np.zeros((n_h,1))			 #(5,1)
		b2=np.zeros((n_y,1))			 #(k,1)
		self.B=[b1,b2]


	def forward_propagate(self):
		Z1=np.dot(self.W[0],self.X)+self.B[0]	#(5,m)
		A1=self.sigmoid(Z1)
		Z2=np.dot(self.W[1],A1)+self.B[1]	#(k,m)
		A2=self.sigmoid(Z2)
		cache ={
		'Z1':Z1,
		'A1':A1,
		'Z2':Z2,
		'A2':A2,
		}
		return A2,cache
	def compute_cost(self,A2):
		return -1*np.sum(np.log(A2)*self.Y+(1-self.Y)*np.log(1-A2))/self.m

	def backwardPropagate(self,cache):
		A1=cache['A1']
		Z1=cache['Z1']
		A2=cache['A2']
		Z2=cache['Z2']
		dZ2=A2-self.Y   #(n_y,m)
		dW2=np.dot(dZ2,A1.T)/self.m 	#(n_y,5)
		dB2=np.sum(dZ2,axis=1,keepdims=True)/self.m  #(n_y,1)
		dZ1=np.dot(self.W[1].T,dZ2)*(1-A1*A1) # (5,m)
		dW1=np.dot(dZ1,self.X.T)/self.m   # (5,3)
		dB1=np.sum(dZ1,axis=1,keepdims=True)/self.m
	
		grads={
		'dW1':dW1,
		'dW2':dW2,
		'dB1':dB1,
		'dB2':dB2
		}
		return grads
	def updateParams(self,grads):
		self.W[0]=self.W[0]-self.lr*grads['dW1']
		self.W[1]=self.W[1]-self.lr*grads['dW2']
		self.B[0]-=self.lr*grads['dB1']
		self.B[1]=self.B[1]-self.lr*grads['dB2']

	def model(self):
		self.layer_size()
		self.initParams(self.n_x,self.n_y,self.n_h)
		for i in range(self.num_iter):
			A2,cache=self.forward_propagate()
			if(i%10==0):
				print(self.compute_cost(A2))
			grads=self.backwardPropagate(cache)
			self.updateParams(grads)
		self.compute_cost(A2)
		print("Neural Network Trained")
		#print(self.W[0],self.B[0],self.W[1],self.B[1])

	def predict(self,X):
		Z1=np.dot(self.W[0],X)+self.B[0]	#(5,m)
		A1=self.sigmoid(Z1)
		Z2=np.dot(self.W[1],A1)+self.B[1]	#(k,m)
		A2=self.sigmoid(Z2)
		print(A2)
X=np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]).T
Y=np.array([[0, 1, 1, 0]])
nn = Neural_Network(X,Y)
nn.model()
X_test=np.array([[1, 0, 0]]).T
print(X_test.shape)
nn.predict(X_test)










