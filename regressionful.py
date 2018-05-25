# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import heapq
#Read the analytics csv file and store our dataset into a dataframe called "df"
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,Ridge
from sklearn import tree,preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from math import sqrt
from sklearn.decomposition import PCA


if __name__ == '__main__':
	df = pd.read_csv('train.csv')
	y=df['SalePrice']
	df=df.drop('SalePrice',axis=1)
	df_test=pd.read_csv('test.csv', index_col=None)
	df=pd.concat([df,df_test],keys=['train','test'])
	'''
	#Investigate features
	temp=df.drop('SalePrice',axis=1)
	for index in temp.columns:
		print(temp[index].unique())
	'''

	df=df.drop(labels=['Id'],axis=1) #drop invalid features

	missing_data_percent=df.isnull().sum()/df.isnull().count()
	#drop features whose missing data is more than 50%
	print(missing_data_percent[missing_data_percent>0.5].index)
	df=df.drop(missing_data_percent[missing_data_percent>0.5].index,axis=1)

	'''
	#removing samples with null features
	tmp=df.isnull().sum()
	nullindex=tmp[tmp>0].index
	for index in nullindex:
		df=df[df[index].isnull()!=True]
	'''
	#Filling NAs
	tmp=df.isnull().sum()
	nullindex=tmp[tmp>0].index
	for index in nullindex:
		df[index]=df[index].fillna(df[index].mode()[0])
	
	#preprocessing
	#unordered categorical
	catindex=['MSZoning','Street','LandContour','LotConfig','Neighborhood','Condition1'\
	,'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd'\
	,'MasVnrType','Foundation','Heating','CentralAir','Electrical','Functional','GarageType'\
	,'SaleType','SaleCondition','Utilities']

	for index in catindex:
		#df[index]=df[index].astype('category').cat.codes
		tmp=pd.get_dummies(df[index])
		df.drop(labels=index,axis=1,inplace=True)
		df=pd.concat([tmp,df],axis=1)
	#ordered categorical
	tname="LotShape"
	df[tname]=df[tname].astype('category').cat.set_categories(['Reg','IR1','IR2','IR3'],ordered=True)
	df[tname]=df[tname].cat.codes
	tname="LandSlope"
	df[tname]=df[tname].astype('category').cat.set_categories(['Gtl','Mod','Sev'],ordered=True)
	df[tname]=df[tname].cat.codes
	tname="PavedDrive"
	df[tname]=df[tname].astype('category').cat.set_categories(['Y','P','N'],ordered=True)
	df[tname]=df[tname].cat.codes
	for tname in ["BsmtFinType1","BsmtFinType2"]:
		df[tname]=df[tname].astype('category').cat.set_categories(['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA'],ordered=True)
		df[tname]=df[tname].cat.codes
	for tname in ["ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","HeatingQC"\
	,"KitchenQual","FireplaceQu","GarageFinish","GarageQual","GarageCond"]:
		df[tname]=df[tname].astype('category').cat.set_categories(['Ex','Gd','TA','Fa','Po'],ordered=True)
		df[tname]=df[tname].cat.codes
	


	#y=np.log(df['SalePrice']) #logarithm of SalePrice refering to RMSE
	
	df.insert(0,'augmentation',1)
	
	X=df.loc['train']
	X_true=df.loc['test']
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10)
	'''
	#Feature Selection
	clf=tree.DecisionTreeClassifier()
	clf.fit(X,y)
	tmp=clf.feature_importances_
	index=heapq.nlargest(120,range(len(tmp)),tmp.take)
	index=sorted(index)
	print("Selected Features: "+str(X.columns[index]))

	X_new=X.ix[:,X.columns[index]]
	X_train,X_test,y_train,y_test=train_test_split(X_new,y,test_size=0.2) #stratify keep the ratio of classes
	
	X_true=X_true.ix[:,X.columns[index]]
	'''
	#preprocessing
	#PCA
	#pca=PCA(n_components=40)
	#pca.fit(X_train)
	#X_train=pca.fit_transform(X_train)
	#X_test=pca.fit_transform(X_test)
	#scaler=preprocessing.StandardScaler().fit(X_train) #Standardization
	#X_train_scaled=scaler.transform(X_train)
	#X_test_scaled=scaler.transform(X_test)

	'''
	print("Linear Regression:")
	model=LinearRegression()
	model.fit(X_train,y_train)
	ACU_train=model.score(X_train,y_train)
	ACU_test=model.score(X_test,y_test)
	y_train_pred=model.predict(X_train)
	RMSE_train=sqrt(mean_squared_error(np.log(y_train),np.log(y_train_pred)))
	y_test_pred=model.predict(X_test)
	RMSE_test=sqrt(mean_squared_error(np.log(y_test),np.log(y_test_pred)))
	print("Train COD:%.2f"%ACU_train)
	print("Test COD:%.2f"%ACU_test)
	print("Train RMSE:%.4f"%RMSE_train)
	print("Test RMSE:%.4f"%RMSE_test)
	'''
	
	
	print("GradientBoostingRegressor:")
	model=GradientBoostingRegressor(n_estimators=30000, learning_rate=0.05, max_depth=4, max_features='sqrt'\
		,min_samples_split=8, loss='huber',alpha=0.9)
	model.fit(X,y)
	y_test_pred=model.predict(X_test)
	RMSE_test=sqrt(mean_squared_error(np.log(y_test),np.log(y_test_pred)))
	print("Test RMSE:%.4f"%RMSE_test)
	ACU=model.score(X_test,y_test)
	print("COD on testing:%.4f"%ACU)

	y_true_pred=model.predict(X_true)
	np.savetxt('my.csv',y_true_pred,delimiter=',')
	