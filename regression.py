# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
import time
#Read the analytics csv file and store our dataset into a dataframe called "df"
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,Ridge
from sklearn import tree,preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from math import sqrt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

def printwt(*args):
	print(datetime.now(), args)

if __name__ == '__main__':
	df = pd.read_csv('train.csv')
	y=df['SalePrice']
	df=df.drop('SalePrice',axis=1)
	#df_test=pd.read_csv('test.csv', index_col=None)
	#df=pd.concat([df,df_test],keys=['train','test'])
	'''
	#Investigate features
	temp=df.drop('SalePrice',axis=1)
	for index in temp.columns:
		print(temp[index].unique())
	'''

	df=df.drop(labels=['Id'],axis=1) #drop invalid features

	missing_data_percent=df.isnull().sum()/df.isnull().count()
	#drop features whose missing data is more than 50%
	#print(missing_data_percent[missing_data_percent>0.5].index)
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
	X=df;
	y=np.log1p(y)
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15)
	#Feature Selection
	clf=tree.DecisionTreeRegressor()
	clf.fit(X,y)
	tmp=clf.feature_importances_
	index=heapq.nlargest(120,range(len(tmp)),tmp.take)
	index=sorted(index)
	print("Selected Features: "+str(X.columns[index]))

	X_new=X.ix[:,X.columns[index]]
	#X_new=X[['GrLivArea','OverallQual']]
	X_train,X_test,y_train,y_test=train_test_split(X_new,y,test_size=0.2) #stratify keep the ratio of classes
	
	#X_true=X_true.ix[:,X.columns[index]]
	'''
	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')
	ax.scatter(X['GrLivArea'],X['OverallQual'],np.exp(y))
	ax.set_ylabel('OverallQual')
	ax.set_xlabel('GrLivArea')
	ax.set_zlabel('SalePrice')
	'''

	printwt("GradientBoostingRegressor:")
	model=GradientBoostingRegressor()
	estimatros_r=np.logspace(1,4,20).astype(int) 
	features_r=['auto','sqrt','log2'] 
	depth_r=np.linspace(1,10,10).astype(int) 
	learning_r=np.logspace(-2,1,20)
	alpha_r=np.linspace(0,1,20);
	minsamp_r=np.linspace(2,16,7).astype(int);
	loss=['huber']

	param_grid=dict(n_estimators=estimatros_r,max_features=features_r,max_depth=depth_r,learning_rate=learning_r\
		,loss=loss,alpha=alpha_r,min_samples_split=minsamp_r)
	grid=GridSearchCV(model,param_grid=param_grid,cv=5,n_jobs=4,scoring='neg_mean_squared_error')
	grid.fit(X_train,y_train)
	printwt("The best parameters are %s with a score of %0.4f" % (grid.best_params_, grid.best_score_))
	print(grid.cv_results_)
	print("Training RMSE:%.4f"%np.sqrt((-grid.cv_results_['mean_train_score'][grid.best_index_])))
	print("Validation RMSE:%.4f"%np.sqrt((-grid.cv_results_['mean_test_score'][grid.best_index_])))
	cvs=np.sqrt(-cross_val_score(model,X,y,cv=5,scoring='neg_mean_squared_error'))
	print("Test RMSE:%.4f"%np.mean(cvs))

	'''
	
	x_min,x_max=X['GrLivArea'].min()-1,X['GrLivArea'].max()+1
	y_min,y_max=X['OverallQual'].min()-1,X['OverallQual'].max()+1
	xx,yy=np.meshgrid(np.arange(x_min,x_max,(x_max-x_min)/50),np.arange(y_min,y_max,(y_max-y_min)/50))
	Z=np.zeros((2500,2),dtype=np.float)
	for i in range(0,50):
		for j in range(0,50):
			Z[i*50+j,:]=[xx[i,j],yy[i,j]]
			
	Z=grid.predict(Z)
	Z=Z.reshape(50,50)
	print(Z)
	ax.plot_surface(xx,yy,np.exp(Z))
	ax.set_ylabel('OverallQual')
	ax.set_xlabel('GrLivArea')
	ax.set_zlabel('SalePrice')
	plt.show()
	'''
	
