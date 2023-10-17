# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and Load the dataset.
2. Define X and Y array and Define a function for costFunction,cost and gradient.
3. Define a function to plot the decision boundary.
4. Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: karnan k
RegisterNumber:212222230062  
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
### Array value of x:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/d95c4639-1913-42bc-be0e-e48bab5c29ab)

### Array value of y:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/e91f0f1c-599c-4620-bb23-d5489d25aad8)

### Score graph:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/65f9aa2e-647b-406f-a0bc-1303cce9ac44)

### Sigmoid function graph:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/3bdcc9f6-f971-42f5-b2de-12b322a27d9e)

### X train grad value:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/384e993a-2580-4818-abc7-6cd150ba2f8c)

### Y train grad value:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/5b521a59-02cc-4755-9c4b-9655bca29c5e)

### Regrssion value:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/da645e6f-0ae5-477a-ae61-23e1753f0df5)

### Decison boundary graph:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/b1b1ecaa-01a1-4ef5-a2dd-a1995f874aa7)
### Probability value:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/c5760259-20fa-4f0a-9086-1cd7e700023f)
### Prediction value of mean:
![image](https://github.com/karnankasinathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118787064/ce33b76b-d816-44ac-a145-c55d53ccb7ee)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

