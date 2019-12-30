import os
import pandas as pd
from pandas import DataFrame,Series
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_fscore_support as prfs

f = pd.read_csv("movie_metadata.csv")
data=pd.DataFrame(f)
X_data=data.dtypes[data.dtypes!='object'].index
X_train=data[X_data]
X_train=X_train.fillna(0)
columns=X_train.columns.tolist()
yr=X_train['imdb_score']
y=X_train['imdb_score'].astype(int)
X_train.drop(['imdb_score'],axis=1,inplace=True)
X_train.drop(['duration'],axis=1,inplace=True)

corr_mat=X_train.corr(method='pearson')
plt.figure(1,figsize=(17,8))
sns.heatmap(corr_mat,vmax=1,vmin=-0.5,square=True,annot=True,cmap='tab20b')
plt.show()

X_Train1=X_train['gross']
X_Train2=X_train['num_user_for_reviews']
X_Train1=np.asarray(X_Train1)
X_Train2=np.asarray(X_Train2)
X_Train=np.column_stack((X_Train1,X_Train2))

# Finding normalised array of X_Train
X_std=StandardScaler().fit_transform(X_Train)
pca = PCA().fit(X_std)
plt.figure(2)
plt.plot(np.cumsum(pca.explained_variance_ratio_),color='red')
plt.xlim(0,9)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

sklearn_pca=PCA(n_components=2)
X_Train=sklearn_pca.fit_transform(X_std)

number_of_samples = len(y)
np.random.seed(0)
random_indices = np.random.permutation(number_of_samples)
num_training_samples = int(number_of_samples*0.75)
x_train = X_Train[random_indices[:num_training_samples]]
y_train=y[random_indices[:num_training_samples]]
yr_train=yr[random_indices[:num_training_samples]]
x_test=X_Train[random_indices[num_training_samples:]]
y_test=y[random_indices[num_training_samples:]]
yr_test=yr[random_indices[num_training_samples:]]
y_train=list(y_train)
y_test=list(y_test)
yr_train=list(yr_train)
yr_test=list(yr_test)

#Ridge Regression

model=linear_model.Ridge()
model.fit(x_train,yr_train)
y_rrtrain=model.predict(x_train)
y_rrtrain=list(y_rrtrain)

error=0
for i in range(len(yr_train)):
    error+=(abs(yr_train[i]-y_rrtrain[i])/yr_train[i])
train_error_ridge=error/len(yr_train)*100
print("Train error for ridge regression= "'{}'.format(train_error_ridge)+" %")

y_rr=model.predict(x_test)
y_rr=list(y_rr)

error=0
for i in range(len(yr_test)):
    error+=(abs(y_rr[i]-yr_test[i])/yr_test[i])
test_error_ridge=error/len(yr_test)*100
print("Test error for ridge regression= "'{}'.format(test_error_ridge)+" %")

#Knn Classifier

n_neighbors=5
knn=neighbors.KNeighborsClassifier(n_neighbors,weights='uniform')
knn.fit(x_train,y_train)
y1_knn=knn.predict(x_train)
y1_knn=list(y1_knn)

error=0
for i in range(len(y_train)):
    error+=(abs(y1_knn[i]-y_train[i])/y_train[i])
train_error_knn=error/len(y_train)*100
print("Train error for knn Classifier= "'{}'.format(train_error_knn)+" %")
y2_knn=knn.predict(x_test)
y2_knn=list(y2_knn)
error=0
for i in range(len(y_test)):
    error+=(abs(y2_knn[i]-y_test[i])/y_test[i])
test_error_knn=error/len(y_test)*100
print("Test error for knn Classifier= "'{}'.format(test_error_knn)+" %")
#print("Precision and Recall for KNN")
knntest = confusion_matrix(y2_knn,y_test)
fpr, tpr, thresholds = roc_curve(y_test, y2_knn, pos_label=2)
aucknn=auc(fpr,tpr)
plt.figure(4,figsize=(20,10))
plt.subplot(231)
plt.plot(fpr,tpr)
plt.title("KNN Classifier -> AUC = %1.3f" %aucknn)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)

knn = pd.DataFrame(knntest, index = [i for i in "234567890"], columns = [i for i in "234567890"])
knn = knn.T
plt.figure(3, figsize=(17,10))
plt.subplot(221)
plt.title('K-Nearest Neighbour')
sns.heatmap(knn, annot=True,square=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')

#Knn regressor

n_neighbors = 5
knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')
knn.fit(x_train,yr_train)
y1_knn=knn.predict(x_train)
y1_knn=list(y1_knn)

error=0
for i in range(len(yr_train)):
    error+=(abs(y1_knn[i]-yr_train[i])/yr_train[i])
train_error_knnr=error/len(yr_train)*100
print("Train error for knn regressor= "'{}'.format(train_error_knnr)+" %")

y2_knn=knn.predict(x_test)
y2_knn=list(y2_knn)
error=0
for i in range(len(yr_test)):
    error+=(abs(y2_knn[i]-yr_test[i])/yr_test[i])
test_error_knnr=error/len(yr_test)*100
print("Test error for knn regressor = "'{}'.format(test_error_knnr)+" %")

fpr, tpr, thresholds = roc_curve(y_test, y2_knn, pos_label=2)
aucknn=auc(fpr,tpr)
plt.figure(5,figsize=(20,10))
plt.subplot(231)
plt.plot(fpr,tpr)
plt.title("KNN Regressor -> AUC = %1.3f" %aucknn)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)

#Naive Bayes Classifier

NB = GaussianNB()
NB.fit(x_train,y_train)
y1_reg=NB.predict(x_train)
y1_reg=list(y1_reg)
y2_reg=NB.predict(x_test)
y2_reg=list(y2_reg)

error=0
for i in range(len(y_train)):
    error+=(abs(y1_reg[i]-y_train[i])/y_train[i])
train_error_bay=error/len(y_train)*100
print("Train error for Gaussian Naive Bayes Classifier= "+'{}'.format(train_error_bay)+" %")

error=0
for i in range(len(y_test)):
    error+=(abs(y2_reg[i]-y_test[i])/y_test[i])
test_error_bay=(error/len(y_test))*100
print("Test error for Gaussian Naive Bayes Classifier= "+'{}'.format(test_error_bay)+" %")

nbtest = confusion_matrix(y2_reg,y_test)
fpr, tpr, thresholds = roc_curve(y_test, y2_reg, pos_label=2)
aucnb=auc(fpr,tpr)
plt.figure(4)
plt.subplot(232)
plt.plot(fpr,tpr)
plt.title("Naive Bayes Classifier -> AUC = %1.3f" %aucnb)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)

nb = pd.DataFrame(nbtest, index = [i for i in "234567890"], columns = [i for i in "234567890"])
nb = nb.T
plt.figure(3, figsize=(17,10))
plt.subplot(222)
plt.title('Naive Bayes')
sns.heatmap(nb, annot=True,square=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')

#Naive Bayes Classifier Bernoulli

NB = BernoulliNB()
NB.fit(x_train,y_train)
y1_reg=NB.predict(x_train)
y1_reg=list(y1_reg)
y2_reg=NB.predict(x_test)
y2_reg=list(y2_reg)

error=0
for i in range(len(y_train)):
    error+=(abs(y1_reg[i]-y_train[i])/y_train[i])
train_error_bayb=error/len(y_train)*100
print("Train error for Bernoulli Naive Bayes Classifier= "+'{}'.format(train_error_bayb)+" %")

error=0
for i in range(len(y_test)):
    error+=(abs(y2_reg[i]-y_test[i])/y_test[i])
test_error_bayb=(error/len(y_test))*100
print("Test error for Bernoulli Naive Bayes Classifier= "+'{}'.format(test_error_bayb)+" %")
fpr, tpr, thresholds = roc_curve(y_test, y2_reg, pos_label=2)
aucnb=auc(fpr,tpr)
plt.figure(4)
plt.subplot(133)
plt.plot(fpr,tpr)
plt.title("Bernoulli Naive Bayes Classifier -> AUC = %1.3f" %aucnb)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)


#Bayesian regression

NB = linear_model.BayesianRidge()
NB.fit(x_train,yr_train)
y1_reg=NB.predict(x_train)
y1_reg=list(y1_reg)
y2_reg=NB.predict(x_test)
y2_reg=list(y2_reg)

error=0
for i in range(len(yr_train)):
    error+=(abs(y1_reg[i]-yr_train[i])/yr_train[i])
train_error_bayr=error/len(yr_train)*100
print("Train error for Bayesian Regression= "+'{}'.format(train_error_bayr)+" %")

error=0
for i in range(len(yr_test)):
    error+=(abs(y2_reg[i]-yr_test[i])/yr_test[i])
test_error_bayr=(error/len(yr_test))*100
print("Test error for Bayesian Regression = "+'{}'.format(test_error_bayr)+" %")
fpr, tpr, thresholds = roc_curve(y_test, y2_reg, pos_label=2)
aucnb=auc(fpr,tpr)
plt.figure(5)
plt.subplot(232)
plt.plot(fpr,tpr)
plt.title("Bayesian regressor -> AUC = %1.3f" %aucnb)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)


#DecisionTree Classifier

dec = tree.DecisionTreeClassifier(max_depth=1)
dec.fit(x_train,y_train)
y1_dec=dec.predict(x_train)
y1_dec=list(y1_dec)
y2_dec=dec.predict(x_test)
y2_dec=list(y2_dec)

error=0
for i in range(len(y_train)):
    error+=(abs(y1_dec[i]-y_train[i])/y_train[i])
train_error_tree=error/len(y_train)*100
print("Train error for Decision Trees Classifier= "+'{}'.format(train_error_tree)+" %")

error=0
for i in range(len(y_test)):
    error+=(abs(y2_dec[i]-y_test[i])/y_test[i])
test_error_tree=error/len(y_test)*100
print("Test error for Decision Trees Classifier= "'{}'.format(test_error_tree)+" %")

dttest = confusion_matrix(y2_dec,y_test)
fpr, tpr, thresholds = roc_curve(y_test, y2_dec, pos_label=2)
aucdt=auc(fpr,tpr)
plt.figure(4)
plt.subplot(234)
plt.plot(fpr,tpr)
plt.title("Decision Tree Classifier -> AUC = %1.3f"%aucdt)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)
#print("Test Confusion Matrix Decision Trees")
#print(dttest)

dt = pd.DataFrame(dttest, index = [i for i in "234567890"], columns = [i for i in "234567890"])
dt=dt.T
plt.figure(3, figsize=(17,10))
plt.subplot(223)
plt.title('Decision Tree')
sns.heatmap(dt, annot=True,square=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')

#DecisionTree Regression

dec = tree.DecisionTreeRegressor(max_depth=1)
dec.fit(x_train,yr_train)
y1_dec=dec.predict(x_train)
y1_dec=list(y1_dec)
y2_dec=dec.predict(x_test)
y2_dec=list(y2_dec)

error=0
for i in range(len(yr_train)):
    error+=(abs(y1_dec[i]-yr_train[i])/yr_train[i])
train_error_treer=error/len(yr_train)*100
print("Train error for Decision Trees Regressor= "+'{}'.format(train_error_treer)+" %")

error=0
for i in range(len(yr_test)):
    error+=(abs(y2_dec[i]-yr_test[i])/yr_test[i])
test_error_treer=error/len(yr_test)*100
print("Test error for Decision Trees Regressor= "'{}'.format(test_error_treer)+" %")

fpr, tpr, thresholds = roc_curve(y_test, y2_dec, pos_label=2)
aucdt=auc(fpr,tpr)
plt.figure(5)
plt.subplot(233)
plt.plot(fpr,tpr)
plt.title("Decision Tree Regressor -> AUC = %1.3f"%aucdt)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)


#SVM Classifier

svm_cls=svm.SVC()
svm_cls.fit(x_train,y_train)
y1_svm=svm_cls.predict(x_train)
y1_svm=list(y1_svm)
y2_svm=svm_cls.predict(x_test)
y2_svm=list(y2_svm)

error=0
for i in range(len(y_train)):
    error+=(abs(y1_svm[i]-y_train[i])/y_train[i])
train_error_svm=error/len(y_train)*100
print("Train error for SVM Classifier= "+'{}'.format(train_error_svm)+" %")


error=0
for i in range(len(y_test)):
    error+=(abs(y2_svm[i]-y_test[i])/y_test[i])
test_error_svm=error/len(y_test)*100
print("Test error for SVM Classifier= "'{}'.format(test_error_svm)+" %")

"""svmtrain = confusion_matrix(y1_svm,y_train)
print("Train Confusion Matrix Decision Trees")
print(svmtrain)"""

svmtest = confusion_matrix(y2_svm,y_test)
#print("Test Confusion Matrix SVM")
#print(svmtest)
fpr, tpr, thresholds = roc_curve(y_test, y2_svm, pos_label=2)
aucsvm=auc(fpr,tpr)
plt.figure(4)
plt.subplot(235)
plt.tight_layout()
plt.plot(fpr,tpr)
plt.title("SVM Classifier -> AUC = %1.3f" %aucsvm)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)

sv = pd.DataFrame(svmtest, index = [i for i in "234567890"], columns = [i for i in "234567890"])
sv=sv.T
plt.figure(3, figsize=(17,10))
plt.subplot(224)
plt.title('Support vector machine')
plt.tight_layout()
sns.heatmap(sv, annot=True,square=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')

#SVM Regressor RBF

svm_clsf=svm.SVR(kernel='rbf',gamma=0.1)
svm_clsf.fit(x_train,yr_train)
y1_svmf=svm_clsf.predict(x_train)
y1_svmf=list(y1_svmf)
y2_svmf=svm_clsf.predict(x_test)
y2_svmf=list(y2_svmf)

error=0
for i in range(len(yr_train)):
    error+=(abs(y1_svmf[i]-yr_train[i])/yr_train[i])
train_error_svmrf=error/len(yr_train)*100
print("Train error for SVM Regressor rbf= "+'{}'.format(train_error_svmrf)+" %")


error=0
for i in range(len(yr_test)):
    error+=(abs(y2_svmf[i]-yr_test[i])/yr_test[i])
test_error_svmrf=error/len(yr_test)*100
print("Test error for SVM Regressor rbf= "'{}'.format(test_error_svmrf)+" %")
fpr, tpr, thresholds = roc_curve(y_test, y2_svmf, pos_label=2)
aucsvmf=auc(fpr,tpr)
plt.figure(5)
plt.subplot(236)
plt.plot(fpr,tpr)
plt.title("SVM Regressor RBF-> AUC = %1.3f" %aucsvmf)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)


#SVM regressor Polynomial

svm_clsp=svm.SVR(kernel='poly',degree=2)
svm_clsp.fit(x_train,yr_train)
y1_svmp=svm_clsp.predict(x_train)
y1_svmp=list(y1_svmp)
y2_svmp=svm_clsp.predict(x_test)
y2_svmp=list(y2_svmp)

error=0
for i in range(len(yr_train)):
    error+=(abs(y1_svmp[i]-yr_train[i])/yr_train[i])
train_error_svmrp=error/len(yr_train)*100
print("Train error for SVM Regressor Poly= "+'{}'.format(train_error_svmrp)+" %")


error=0
for i in range(len(yr_test)):
    error+=(abs(y2_svmp[i]-yr_test[i])/yr_test[i])
test_error_svmrp=error/len(yr_test)*100
print("Test error for SVM Regressor Poly= "'{}'.format(test_error_svmrp)+" %")
fpr, tpr, thresholds = roc_curve(y_test, y2_svmp, pos_label=2)
aucsvmp=auc(fpr,tpr)
plt.figure(5)
plt.subplot(235)
plt.plot(fpr,tpr)
plt.title("SVM Regressor Poly-> AUC = %1.3f" %aucsvmp)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)


#SVM regressor Linear

svm_cls=svm.SVR(kernel='linear')
svm_cls.fit(x_train,yr_train)
y1_svm=svm_cls.predict(x_train)
y1_svm=list(y1_svm)
y2_svm=svm_cls.predict(x_test)
y2_svm=list(y2_svm)

error=0
for i in range(len(yr_train)):
    error+=(abs(y1_svm[i]-yr_train[i])/yr_train[i])
train_error_svmr=error/len(yr_train)*100
print("Train error for SVM Regressor= "+'{}'.format(train_error_svmr)+" %")


error=0
for i in range(len(yr_test)):
    error+=(abs(y2_svm[i]-yr_test[i])/yr_test[i])
test_error_svmr=error/len(yr_test)*100
print("Test error for SVM Regressor= "'{}'.format(test_error_svmr)+" %")
fpr, tpr, thresholds = roc_curve(y_test, y2_svm, pos_label=2)
aucsvm=auc(fpr,tpr)
plt.figure(5)
plt.subplot(234)
plt.plot(fpr,tpr)
plt.title("SVM Regressor -> AUC = %1.3f" %aucsvm)
plt.xlabel("False Positive Ratio")
plt.ylabel("True Positive Ratio")
plt.xlim(0,1)
plt.ylim(-0.05,1.05)

plt.show(3)
plt.show(4)
plt.show(5)
#plt.show(6)
#plt.show(7)

test_error = [test_error_ridge,test_error_knnr,test_error_bayr,test_error_treer,test_error_svmr,test_error_svmrp,test_error_svmrf,test_error_knn,test_error_bay,test_error_bayb,test_error_tree,test_error_svm]
train_error = [train_error_ridge,train_error_knnr,train_error_bayr,train_error_treer,train_error_svmr,train_error_svmrp,train_error_svmrf,train_error_knn,train_error_bay,train_error_bayb,train_error_tree,train_error_svm]
col={'Train Error':train_error,'Test Error':test_error}
models=['Ridge Regression','Knn Regressor','Bayesian Regression','Decision Tree Regressor','SVM Regressor Linear','SVM Regressor Polynomial','SVM Regressor RBF','Knn Classifier','GaussianNB Classifier','BernoulliNB Classifier','Decision Tree Classifier','SVM Classifier']
fdf=DataFrame(data=col,index=models)
fdf.plot(kind='bar')
plt.tight_layout()
plt.xlabel("Models")
plt.ylabel("Error Percentage")
plt.title("Comparison between models")
plt.show()
