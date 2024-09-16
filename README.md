# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ARPAN BARDHAN
RegisterNumber:  212222040013
*/
```
```PYTHON
import numpy as np
import pandas as pd

dataset = pd.read_csv('/content/Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset['gender'] = dataset['gender'].astype('category')
dataset['ssc_b'] = dataset['ssc_b'].astype('category')
dataset['hsc_b'] = dataset['hsc_b'].astype('category')
dataset['degree_t'] = dataset['degree_t'].astype('category')
dataset['workex'] = dataset['workex'].astype('category')
dataset['specialisation'] = dataset['specialisation'].astype('category')
dataset['status'] = dataset['status'].astype('category')
dataset['status'] = dataset['status'].astype('category')
dataset.dtypes

dataset['gender'] = dataset['gender'].cat.codes
dataset['ssc_b'] = dataset['ssc_b'].cat.codes
dataset['hsc_b'] = dataset['hsc_b'].cat.codes
dataset['degree_t'] = dataset['degree_t'].cat.codes
dataset['workex'] = dataset['workex'].cat.codes
dataset['specialisation'] = dataset['specialisation'].cat.codes
dataset['status'] = dataset['status'].cat.codes
dataset['hsc_s'] = dataset['hsc_s'].astype('category')
dataset['hsc_s'] = dataset['hsc_s'].cat.codes
dataset.dtypes

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

Y

theta = np.random.randn(X.shape[1])
y = Y

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def loss(theta,X,y):
  h = sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h-y) / m
    theta -= alpha * gradient

  return theta

theta = gradient_descent(theta,X,y,alpha = 0.01,num_iterations=1000)

def predict(theta,X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5, 1, 0)
  return y_pred

y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
#print(y_pred)

print(y_pred)
print(Y)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```
## Output:
![Screenshot 2024-09-16 104718](https://github.com/user-attachments/assets/16fd0e1f-af95-4cee-9a5e-39256f09c6ac)

![Screenshot 2024-09-16 104805](https://github.com/user-attachments/assets/edd6fb37-bfcc-47fa-b77d-630b6465a60e)

![Screenshot 2024-09-16 104811](https://github.com/user-attachments/assets/adc0f105-1915-4c4e-b90a-98647c07bce3)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

