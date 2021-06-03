# -*- coding: utf-8 -*-
"""
Introduction au Machine Learning
TP 3
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Voici un commentaire inutile

#Quelques fonctions inutiles
def sum(a,b):
	return a+b

plt.close("all")

# Chargement du dataset
house_data=pd.read_csv('house_data.csv')

# Affichage du nuage de points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(house_data['surface'],house_data['arrondissement'],\
           house_data['price'])
plt.show()

# Suppression des NaN et outliers :
mylist=[]
for column in house_data:
    for i in range(len(house_data)):
        if math.isnan(house_data[column][i]) and i not in mylist:
            mylist=mylist+[i]
house_data=house_data.drop(mylist)
house_data=house_data[house_data['price']<10000]

# On "numérise" la variable catégorielle "arrondissement" : on crée des
# catégories pour chaque arrondissement
house_data["arrondissement"]=house_data['arrondissement'].apply(str)
data2=pd.get_dummies(house_data['arrondissement'])
del house_data["arrondissement"]
for key in data2:
    house_data["arrondissement_"+key]=data2[key]

# Inputs
X=np.matrix([np.ones(house_data.shape[0]),\
            house_data["surface"].values,\
                        house_data["arrondissement_1.0"].values,\
                            house_data["arrondissement_2.0"].values,\
                                house_data["arrondissement_3.0"].values,\
                                    house_data["arrondissement_4.0"].values,\
                                        house_data["arrondissement_10.0"].values]).T


# Valeur que l'on cherche à prédire : montant du loyer
Y=np.matrix(house_data["price"]).T


# Calcul analytique du paramètre theta de la régression
theta=np.linalg.inv(X.T*X)*X.T*Y

# Calcul en utilisant les algorithmes de la bibliothèque scikit-learn
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=.8)
regr=linear_model.LinearRegression()
regr.fit(X_train,Y_train)
predicted=regr.predict(X_test)
print(1-regr.score(X_test,Y_test))