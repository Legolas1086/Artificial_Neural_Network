import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#importing data
data=pd.read_csv('Bank.csv')

x=data.iloc[:,3:-1].values
y=data.iloc[:,-1].values.reshape(-1,1)

#preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

#Encoding gender
lb=LabelEncoder()
x[:,2]=lb.fit_transform(x[:,2])

#Encoding Country
ct=ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=ct.fit_transform(x)

#Splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Standardising the features
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)

#Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

#NN Build

network=Sequential()
network.add(Dense(units=6,activation='relu'))
network.add(Dense(units=6,activation='relu'))
network.add(Dense(units=1,activation='sigmoid'))   #Output Layer


#NN Compile
optimize=keras.optimizers.Adam(learning_rate=1e-1)
network.compile(optimizer=optimize,loss='binary_crossentropy',metrics=['accuracy'])

##NN Trainig 
network.fit(x_train,y_train,batch_size=50,epochs=100)

##NN Testing
y_pred=network.predict(x_test)
y_pred=y_pred>0.5

#Analysing NN
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))












