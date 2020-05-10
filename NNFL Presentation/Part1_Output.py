import numpy as np
import matplotlib.pyplot as plt
import time

class NeuralNetwork():
  def __init__(self,n_inputs,n_outputs,hidden_sizes=[2],w1=0.1,w3=0.4,w2=0.5,w4=0.3,w5=0.2,w6=0.6):
    self.nx=n_inputs
    self.ny=n_outputs
    self.nh=len(hidden_sizes)
    self.sizes=[self.nx]+hidden_sizes+[self.ny]
    self.W={}
    self.learning_parameter=0.7

    for i in range(self.nh+1):
      self.W[i+1]=np.random.randn(self.sizes[i],self.sizes[i+1])

    self.W[self.nh+1]=np.array([np.array([w5]),np.array([w6])])
    self.W[self.nh]=np.array([np.array([w1,w2]),np.array([w3,w4])])


  def sigmoid(self,x):
    return 1.0/(1.0+np.exp(-x))

  def softmax(self,x):
    exps=np.exp(x)
    return exps/np.sum(exps)


  def forward_pass(self,x):
    self.A={}
    self.H={}
    self.H[0]=x.reshape(1,-1)
    for i in range(self.nh):
      self.A[i+1]=np.matmul(self.H[i],self.W[i+1])
      self.H[i+1]=self.sigmoid(self.A[i+1])
    self.A[self.nh+1]=np.matmul(self.H[self.nh],self.W[self.nh+1])
    self.H[self.nh+1]=self.sigmoid(self.A[self.nh+1])
    return self.H[self.nh+1]

  def predict(self,X):
    Y_pred=[]
    for x in X:
      y_pred=self.forward_pass(x)
      Y_pred.append(y_pred)
    return np.array(Y_pred).squeeze()

  def predict_H(self,X,layer,neuron):
    Y_pred=[]
    for x in X:
      y_pred=self.forward_pass(x)
      Y_pred.append([item[neuron] for item in self.H[layer]])
    return np.array(Y_pred)

  def grad_sigmoid(self,x):
    return x*(1-x)

  def grad_softmax(self,x):
    return x*(1-x)

  def loss_function(self,x,y_true):
    y=self.predict(x)
    return 0.5*(y-y_true)**2

  def grad(self,x,y):
    self.forward_pass(x)
    self.dW={}
    self.dH={}
    self.dA={}
    L=self.nh+1
    self.dA[L]=np.multiply((self.H[L]-y),self.grad_sigmoid(self.H[L]))
    for k in range(L,0,-1):
      self.dW[k]=np.matmul(self.H[k-1].T,self.dA[k])
      self.dH[k-1]=np.matmul(self.dA[k],self.W[k].T)
      self.dA[k-1]=np.multiply(self.dH[k-1],self.grad_sigmoid(self.H[k-1]))

  def grad_update(self):
    for k in range(self.nh+1,0,-1):
      self.W[k]=self.W[k]-self.learning_parameter*self.dW[k]

x=np.array([0.23,0.82])
NN=NeuralNetwork(n_inputs=2,n_outputs=1,hidden_sizes=[2])
print(NN.loss_function([x],1))
xdata=[]
ydata1=[]
ydata2=[]
ydata3=[]
ydata4=[]
fig, axes = plt.subplots(2, 2)

line1, = axes[0,0].plot(xdata, ydata1, 'r-')
line2, = axes[0,1].plot(xdata, ydata2, 'r-')
line3, = axes[1,0].plot(xdata, ydata3, 'r-')
line4, = axes[1,1].plot(xdata, ydata4, 'r-')

axes[0,0].set_ylim(0,1)
axes[0,0].set_xlim(0,100)
axes[0,1].set_ylim(0,0.1)
axes[0,1,].set_xlim(0,100)
axes[1,0].set_ylim(0,1)
axes[1,0].set_xlim(0,100)
axes[1,1].set_ylim(0,1)
axes[1,1].set_xlim(0,100)

for i in range(100):
  NN.grad(x,y=1)
  NN.grad_update()
  xdata.append(i)
  ydata1.append(NN.predict([x]))
  ydata2.append(NN.loss_function([x],1))
  line1.set_xdata(xdata)
  line1.set_ydata(ydata1)
  line2.set_xdata(xdata)
  line2.set_ydata(ydata2)
    
  plt.draw()
  plt.pause(1e-17)
  time.sleep(0.1)
plt.plot()
plt.show()
