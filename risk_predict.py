
# coding: utf-8

# In[1]:


'''    Importing Libraries    ''' 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# In[101]:


'''Reading Data'''
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[75]:


'''Drop Attributes

Passing the data and corresponding attribute to remove

'''
def drop_attrib(X,attrib):
    X.drop(attrib,inplace = True,axis = 1)


# In[76]:


'''
Splitting the target column "risk". Removing Text Columns
'''
X = train.copy()


def split_target(data):
    z = data["risk"]
    drop_attrib(data,"risk")
    return z

y = split_target(X)


# In[104]:


'''
Making copy for Regressor and Classifier

'''
X_reg__ = X.copy()
X_clf__ = X.copy()
y_reg__ = y.copy()
y_clf__ = y.copy()
def classify_my_y(data):
    data[data!=0] = 1   # For classifying Using One v/s Rest strategy
classify_my_y(y_clf__)


# In[99]:


'''
Dropping data for Regressor Calculated using feature_importances_ of GradientBoostingRegressor 
'''

def reg_remove_scale(dat):
    data = dat.copy()
    str_remove = ["batt_instance","batt_voltage","installed_count","event_country_code","batt_manufacturer"]
    for i in range(len(str_remove)):
        drop_attrib(data,str_remove[i])
    return scale_data(data)    


# In[100]:


'''
Dropping data for Classifier Calculated using feature_importances_ of GradientBoostingClassifier 
'''
def clf_remove_scale(dat):
    data = dat.copy()
    str_remove = ["batt_instance","batt_voltage","installed_count","design_voltage","event_country_code","batt_manufacturer"]
    for i in range(len(str_remove)):
        drop_attrib(data,str_remove[i])
    return scale_data(data)    



# In[87]:


'''

Scaling of Data

'''
def scale_data(data):
    scaler = StandardScaler()
    z = scaler.fit_transform(data)
    return z
    


# In[88]:


'''
Splitting Data for Training Cross Validation and Test
'''

X_reg = reg_remove_scale(X_reg__)
X_clf = clf_remove_scale(X_clf__)

X_real,X_test,y_real,y_test = train_test_split(X_reg,y_reg,test_size = 0.2,random_state = 20)
X_train,X_val,y_train,y_val = train_test_split(X_real,y_real,test_size = 0.2,random_state = 42)

cX_real,cX_test,cy_real,cy_test = train_test_split(X_clf,y_clf,test_size = 0.2,random_state = 20)
cX_train,cX_val,cy_train,cy_val = train_test_split(cX_real,cy_real,test_size = 0.2,random_state = 42)



# In[1]:


'''

Pre-calculated Parameter Declaration For Regressor ran over multiple iteration of Cross Validations

'''

reg_impurity = 2
reg_depth = 10
reg_estimator = 377
reg_split = 9
reg_leaf = 8
reg_learn = 0.2


# In[2]:


'''

Pre-calculated Parameter Declaration For Classifier ran over multiple iteration of Cross Validations

'''


clf_depth = 8
clf_estimator = 100
clf_split = 9
clf_leaf = 8
clf_learn = 0.09


# In[33]:


'''

Regressor Parameter Calculation : 

0.Keep n_estimators as ~1000 for calculation of the below parameters

1.max_depth: Fixing it as 10 because after this state it overfits 
             using GridSearch always gives the maximum among all
             the depths passed so chose this after lot of iterations
             and cross validation.

2.min_samples_split and min_samples_leaf : Do grid search for combination of [8,9] in both the parameters.

3.min_impurity_decrease : GridSearch for [0,1,2,3,4] gives mostly  2 

4.learning_rate: GridSearch for [0.05,0.1,0.2,0.3] mostly gives 0.2 for this dataset No need to GridCheck !

4.5 : Using the above parameters to calculate the optimal n_estimator

5.n_estimator : Using early stopping strategy to calculate no. of trees estimators.  


'''


# In[ ]:




#-------Regressor Parameter Calculation Code------#




'''
For  min_samples_split and min_samples_leaf
'''


gbrt_leaf_split = GradientBoostingRegressor(max_depth=10,n_estimators=1000,learning_rate=0.2)
param_grid = { "min_samples_split" : [8,9],
              "min_samples_leaf" : [8,9]
              
        
        }
CV_split_leaf = GridSearchCV(gbrt_leaf_split, param_grid=param_grid, cv= 3,scoring = 'neg_mean_squared_error')
CV_split_leaf.fit(X_real,y_real )
leaf_split = CV_split_leaf.best_params_
reg_split = leaf_split['min_samples_splits']
reg_leaf = leaf_split['min_samples_leaf']


# In[ ]:


'''
For  min_impurity_decrease
'''


gbrt_impure = GradientBoostingRegressor(max_depth=10,n_estimators=1000,learning_rate=0.2,min_samples_leaf=reg_leaf,min_samples_split=reg_split)
param_grid = { 
              "min_impurity_decrease" : [0,1,2,3,4]
              
        
        }
CV_impure = GridSearchCV(gbrt_impure, param_grid=param_grid, cv= 3,scoring = 'neg_mean_squared_error')
CV_impure.fit(X_real,y_real )
impure = CV_impure.best_params_
reg_impurity = impure['min_impurity_decrease']


# In[ ]:


'''
Using Early stopping strategy to calculate no. of estimators 
required in order to decrease computations and overfitting

'''

gbrt_est = GradientBoostingRegressor(min_impurity_decrease=reg_impurity,max_depth=reg_depth,min_samples_leaf = reg_leaf,min_samples_split=reg_split,warm_start = True,learning_rate = reg_learn)
min_val_error = float("inf")
error_going_up = 0
n_estimators = 0
for n_estimators in range(1,1000):
    gbrt_est.n_estimators = n_estimators
    gbrt_est.fit(X_real,y_real)
    y_pred = gbrt_est.predict(X_test)
    val_error = mean_squared_error(y_test,y_pred)
    if val_error<min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up +=1
        if error_going_up==10:
            break ;

# increasing estimator
n_estimators = (math.ceil(n_estimators/100))*100

reg_estimator = n_estimators            


# In[ ]:


'''

Classifier Parameter Calculation : 

0.Keep n_estimators as ~1000 for calculation of the below parameters

1.max_depth: Fixing it as 8 because after this state it overfits 
             using GridSearch always gives the maximum among all
             the depths passed so chose this after lot of iterations
             and cross validation.

2.min_samples_split and min_samples_leaf : Do grid search for combination of [8,9] in both the parameters.

3.learning_rate: GridSearch for [0.09,,0.1,0.2] mostly gives 0.09 for this dataset No need to GridCheck !

3.5 : Using the above parameters to calculate the optimal n_estimator

4.n_estimator : Using early stopping strategy to calculate no. of trees estimators.  


'''


# In[ ]:




#-------Classifier Parameter Calculation Code------#




'''
For  min_samples_split and min_samples_leaf
'''


c_gbrt_leaf_split = GradientBoostingClassifier(max_depth=8,n_estimators=1000,learning_rate=0.09)
param_grid = { "min_samples_split" : [8,9],
              "min_samples_leaf" : [8,9]
              
        
        }
c_CV_split_leaf = GridSearchCV(c_gbrt_leaf_split, param_grid=param_grid, cv= 3,scoring = 'neg_mean_squared_error')
c_CV_split_leaf.fit(cX_real,cy_real )
c_leaf_split = c_CV_split_leaf.best_params_
clf_split = c_leaf_split['min_samples_splits']
clf_leaf = c_leaf_split['min_samples_leaf']


# In[ ]:


'''
Using Early stopping strategy to calculate no. of estimators 
required in order to decrease computations and overfitting

'''

c_gbrt_est = GradientBoostingClassifier(max_depth=8,min_samples_split=9,min_samples_leaf=8,learning_rate=0.09,warm_start=True)
min_val_error = float("inf")
error_going_up = 0
n_estimators = 0
for n_estimators in range(1,1200):
    c_gbrt_est.n_estimators = n_estimators
    c_gbrt_est.fit(cX_real_drop,cy_real_drop)
    y_pred = c_gbrt_est.predict(cX_test_drop)
    val_error = mean_squared_error(cy_test_drop,y_pred)
    if val_error<min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up +=1
        if error_going_up==15: 
            break 
# A minimum of 100 estimators for the classifier            
if n_estimators<100:
    n_estimators = 100
clf_estimator = n_estimators


# In[5]:


'''

Function : 
            Sets the risk = 0 values calculated by classifier in the respective predictions made by the 
            regressor
            
            Also rounds off predictions
            
            Fills up predictions which are negative and classifier says risk!=0 with the average
            calculated in this case with diff combination and diff random seeds in train,val and 
            test set.

'''

def precise_pred(reg_pred,clf_pred,avg):
    reg =reg_pred.copy() 
    clf = clf_pred.copy()
    reg[clf==0] = 0
    reg[(reg>0) & (reg<=0.5)] = 1
    reg[(reg<0) & (clf!=0)] = avg # Filling the negative predictions made by regressor with average values in the case where my cross validation test used to give negative predictions instead of positive value
    for i in range(len(reg)):
        
        reg[i] = round(reg[i])
    
    return reg
    
    


# In[6]:


# Calculation of the average

def cal_avg(y_re,reg,clf):
    y_real = np.array(y_re)
    count = 0
    av = 0
    for i in range(len(y_real)):
        if (-y_real[i]<0) and (y_real[i]!=0) and (clf[i]!=0) and (round(reg[i])<=0):
            av = av + y_real[i]
            count+=1
    return round((av/count))        


# In[12]:


'''

Returns the Model of Regressor and Classifier after fitting data

NOTE: If new training data is feeded grid search for the best parameter first as 
      variables declared above will be used else default calculated paramter would
      be used
'''

def reg_clf_model_fit(reg_train,reg_y,clf_train,clf_y):
    gbrt_reg = GradientBoostingRegressor(n_estimators=reg_estimator,max_depth=reg_depth,min_impurity_decrease=reg_impurity,min_samples_leaf=reg_leaf,min_samples_split=reg_split,learning_rate=reg_learn)
    gbrt_clf = GradientBoostingClassifier(n_estimators=clf_estimator,max_depth=clf_depth,min_samples_leaf=clf_leaf,min_samples_split=clf_split,learning_rate=clf_learn)
    
    gbrt_reg.fit(reg_train,reg_y)
    gbrt_clf.fit(clf_train,clf_y)
    return gbrt_reg,gbrt_clf
    
def svc_model_fit(clf_train,clf_y):
    rbf = SVC(kernel = "rbf",gamma=0.096,C=100)
       
    rbf.fit(clf_train,clf_y)
    return rbf


# In[11]:


'''
Returns the Prediction of Regressor and Classifier

'''

def model_prediction(reg_model,clf_model,reg_data,clf_data):
    return reg_model.predict(reg_data),clf_model.predict(clf_data)

def svc_pred(rbf_model,rbf_data):
    return rbf_model.predict(rbf_data)


# In[96]:


'''
Average Calculation Process
'''
REG,CLF = reg_clf_model_fit(X_real,y_real,cX_real,cy_real)
reg_pred_1,clf_pred_1 = model_prediction(REG,CLF,X_test,cX_test)
av1 = cal_avg(y_test,reg_pred_1,clf_pred_1)
REG,CLF = reg_clf_model_fit(X_train,y_train,cX_train,cy_train)
reg_pred_1,clf_pred_1 = model_prediction(REG,CLF,X_val,cX_val)
av2 = cal_avg(y_val,reg_pred_1,clf_pred_1)
REG,CLF = reg_clf_model_fit(X_train,y_train,cX_train,cy_train)
reg_pred_1,clf_pred_1 = model_prediction(REG,CLF,X_test,cX_test)
av3 = cal_avg(y_test,reg_pred_1,clf_pred_1)
av = (av1+av2+av3)/3


# In[133]:


### FINAL  PREDICTIONS ###

test_reg = reg_remove_scale(test)
test_clf = clf_remove_scale(test)


# In[105]:


REG_test,CLF_test = reg_clf_model_fit(X_reg,y,X_clf,y_clf)
reg_test_pred,clf_test_pred = model_prediction(REG_test,CLF_test,test_reg,test_clf)
SVC_test = svc_model_fit(X_clf,y_clf)
svc_test_pred = svc_pred(SVC_test,test_clf)


# In[13]:


output = precise_pred(reg_test_pred,clf_test_pred,av)
svc_output = precise_pred(reg_test_pred,svc_test_pred)

