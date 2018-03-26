{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''    Importing Libraries    ''' \n",
    "import math\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.ensemble import GradientBoostingRegressor\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.svm import SVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''Reading Data'''\n",
    "train = pd.read_csv('train.csv')\n",
    "test = pd.read_csv('test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''Drop Attributes\n",
    "\n",
    "Passing the data and corresponding attribute to remove\n",
    "\n",
    "'''\n",
    "def drop_attrib(X,attrib):\n",
    "    X.drop(attrib,inplace = True,axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Splitting the target column \"risk\". Removing Text Columns\n",
    "'''\n",
    "X = train.copy()\n",
    "\n",
    "\n",
    "def split_target(data):\n",
    "    z = data[\"risk\"]\n",
    "    drop_attrib(data,\"risk\")\n",
    "    return z\n",
    "\n",
    "y = split_target(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Making copy for Regressor and Classifier\n",
    "\n",
    "'''\n",
    "X_reg__ = X.copy()\n",
    "X_clf__ = X.copy()\n",
    "y_reg__ = y.copy()\n",
    "y_clf__ = y.copy()\n",
    "def classify_my_y(data):\n",
    "    data[data!=0] = 1   # For classifying Using One v/s Rest strategy\n",
    "classify_my_y(y_clf__)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Dropping data for Regressor Calculated using feature_importances_ of GradientBoostingRegressor \n",
    "'''\n",
    "\n",
    "def reg_remove_scale(dat):\n",
    "    data = dat.copy()\n",
    "    str_remove = [\"batt_instance\",\"batt_voltage\",\"installed_count\",\"event_country_code\",\"batt_manufacturer\"]\n",
    "    for i in range(len(str_remove)):\n",
    "        drop_attrib(data,str_remove[i])\n",
    "    return scale_data(data)    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Dropping data for Classifier Calculated using feature_importances_ of GradientBoostingClassifier \n",
    "'''\n",
    "def clf_remove_scale(dat):\n",
    "    data = dat.copy()\n",
    "    str_remove = [\"batt_instance\",\"batt_voltage\",\"installed_count\",\"design_voltage\",\"event_country_code\",\"batt_manufacturer\"]\n",
    "    for i in range(len(str_remove)):\n",
    "        drop_attrib(data,str_remove[i])\n",
    "    return scale_data(data)    \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "\n",
    "Scaling of Data\n",
    "\n",
    "'''\n",
    "def scale_data(data):\n",
    "    scaler = StandardScaler()\n",
    "    z = scaler.fit_transform(data)\n",
    "    return z\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Splitting Data for Training Cross Validation and Test\n",
    "'''\n",
    "\n",
    "X_reg = reg_remove_scale(X_reg__)\n",
    "X_clf = clf_remove_scale(X_clf__)\n",
    "\n",
    "X_real,X_test,y_real,y_test = train_test_split(X_reg,y_reg,test_size = 0.2,random_state = 20)\n",
    "X_train,X_val,y_train,y_val = train_test_split(X_real,y_real,test_size = 0.2,random_state = 42)\n",
    "\n",
    "cX_real,cX_test,cy_real,cy_test = train_test_split(X_clf,y_clf,test_size = 0.2,random_state = 20)\n",
    "cX_train,cX_val,cy_train,cy_val = train_test_split(cX_real,cy_real,test_size = 0.2,random_state = 42)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "\n",
    "Pre-calculated Parameter Declaration For Regressor ran over multiple iteration of Cross Validations\n",
    "\n",
    "'''\n",
    "\n",
    "reg_impurity = 2\n",
    "reg_depth = 10\n",
    "reg_estimator = 377\n",
    "reg_split = 9\n",
    "reg_leaf = 8\n",
    "reg_learn = 0.2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "\n",
    "Pre-calculated Parameter Declaration For Classifier ran over multiple iteration of Cross Validations\n",
    "\n",
    "'''\n",
    "\n",
    "\n",
    "clf_depth = 8\n",
    "clf_estimator = 100\n",
    "clf_split = 9\n",
    "clf_leaf = 8\n",
    "clf_learn = 0.09"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\n\\nRegressor Parameter Calculation : \\n\\n0.Keep n_estimators as ~1000 for calculation of the below parameters\\n\\n1.max_depth: Fixing it as 10 because after this state it overfits \\n             using GridSearch always gives the maximum among all\\n             the depths passed so chose this after lot of iterations\\n             and cross validation.\\n\\n2.min_samples_split and min_samples_leaf : Do grid search for combination of [8,9] in both the parameters.\\n\\n3.min_impurity_decrease : GridSearch for [0,1,2,3,4] gives mostly  2 \\n\\n4.learning_rate: GridSearch for [0.]\\n\\n4.5 : Using the above parameters to calculate the optimal n_estimator\\n\\n5.n_estimator : Using early stopping strategy to calculate no. of trees estimators.  \\n\\n\\n'"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "\n",
    "Regressor Parameter Calculation : \n",
    "\n",
    "0.Keep n_estimators as ~1000 for calculation of the below parameters\n",
    "\n",
    "1.max_depth: Fixing it as 10 because after this state it overfits \n",
    "             using GridSearch always gives the maximum among all\n",
    "             the depths passed so chose this after lot of iterations\n",
    "             and cross validation.\n",
    "\n",
    "2.min_samples_split and min_samples_leaf : Do grid search for combination of [8,9] in both the parameters.\n",
    "\n",
    "3.min_impurity_decrease : GridSearch for [0,1,2,3,4] gives mostly  2 \n",
    "\n",
    "4.learning_rate: GridSearch for [0.05,0.1,0.2,0.3] mostly gives 0.2 for this dataset No need to GridCheck !\n",
    "\n",
    "4.5 : Using the above parameters to calculate the optimal n_estimator\n",
    "\n",
    "5.n_estimator : Using early stopping strategy to calculate no. of trees estimators.  \n",
    "\n",
    "\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "#-------Regressor Parameter Calculation Code------#\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "'''\n",
    "For  min_samples_split and min_samples_leaf\n",
    "'''\n",
    "\n",
    "\n",
    "gbrt_leaf_split = GradientBoostingRegressor(max_depth=10,n_estimators=1000,learning_rate=0.2)\n",
    "param_grid = { \"min_samples_split\" : [8,9],\n",
    "              \"min_samples_leaf\" : [8,9]\n",
    "              \n",
    "        \n",
    "        }\n",
    "CV_split_leaf = GridSearchCV(gbrt_leaf_split, param_grid=param_grid, cv= 3,scoring = 'neg_mean_squared_error')\n",
    "CV_split_leaf.fit(X_real,y_real )\n",
    "leaf_split = CV_split_leaf.best_params_\n",
    "reg_split = leaf_split['min_samples_splits']\n",
    "reg_leaf = leaf_split['min_samples_leaf']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "For  min_impurity_decrease\n",
    "'''\n",
    "\n",
    "\n",
    "gbrt_impure = GradientBoostingRegressor(max_depth=10,n_estimators=1000,learning_rate=0.2,min_samples_leaf=reg_leaf,min_samples_split=reg_split)\n",
    "param_grid = { \n",
    "              \"min_impurity_decrease\" : [0,1,2,3,4]\n",
    "              \n",
    "        \n",
    "        }\n",
    "CV_impure = GridSearchCV(gbrt_impure, param_grid=param_grid, cv= 3,scoring = 'neg_mean_squared_error')\n",
    "CV_impure.fit(X_real,y_real )\n",
    "impure = CV_impure.best_params_\n",
    "reg_impurity = impure['min_impurity_decrease']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Using Early stopping strategy to calculate no. of estimators \n",
    "required in order to decrease computations and overfitting\n",
    "\n",
    "'''\n",
    "\n",
    "gbrt_est = GradientBoostingRegressor(min_impurity_decrease=reg_impurity,max_depth=reg_depth,min_samples_leaf = reg_leaf,min_samples_split=reg_split,warm_start = True,learning_rate = reg_learn)\n",
    "min_val_error = float(\"inf\")\n",
    "error_going_up = 0\n",
    "n_estimators = 0\n",
    "for n_estimators in range(1,1000):\n",
    "    gbrt_est.n_estimators = n_estimators\n",
    "    gbrt_est.fit(X_real,y_real)\n",
    "    y_pred = gbrt_est.predict(X_test)\n",
    "    val_error = mean_squared_error(y_test,y_pred)\n",
    "    if val_error<min_val_error:\n",
    "        min_val_error = val_error\n",
    "        error_going_up = 0\n",
    "    else:\n",
    "        error_going_up +=1\n",
    "        if error_going_up==10:\n",
    "            break ;\n",
    "\n",
    "# increasing estimator\n",
    "n_estimators = (math.ceil(n_estimators/100))*100\n",
    "\n",
    "reg_estimator = n_estimators            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "\n",
    "Classifier Parameter Calculation : \n",
    "\n",
    "0.Keep n_estimators as ~1000 for calculation of the below parameters\n",
    "\n",
    "1.max_depth: Fixing it as 8 because after this state it overfits \n",
    "             using GridSearch always gives the maximum among all\n",
    "             the depths passed so chose this after lot of iterations\n",
    "             and cross validation.\n",
    "\n",
    "2.min_samples_split and min_samples_leaf : Do grid search for combination of [8,9] in both the parameters.\n",
    "\n",
    "3.learning_rate: GridSearch for [0.09,,0.1,0.2] mostly gives 0.09 for this dataset No need to GridCheck !\n",
    "\n",
    "3.5 : Using the above parameters to calculate the optimal n_estimator\n",
    "\n",
    "4.n_estimator : Using early stopping strategy to calculate no. of trees estimators.  \n",
    "\n",
    "\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "#-------Classifier Parameter Calculation Code------#\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "'''\n",
    "For  min_samples_split and min_samples_leaf\n",
    "'''\n",
    "\n",
    "\n",
    "c_gbrt_leaf_split = GradientBoostingClassifier(max_depth=8,n_estimators=1000,learning_rate=0.09)\n",
    "param_grid = { \"min_samples_split\" : [8,9],\n",
    "              \"min_samples_leaf\" : [8,9]\n",
    "              \n",
    "        \n",
    "        }\n",
    "c_CV_split_leaf = GridSearchCV(c_gbrt_leaf_split, param_grid=param_grid, cv= 3,scoring = 'neg_mean_squared_error')\n",
    "c_CV_split_leaf.fit(cX_real,cy_real )\n",
    "c_leaf_split = c_CV_split_leaf.best_params_\n",
    "clf_split = c_leaf_split['min_samples_splits']\n",
    "clf_leaf = c_leaf_split['min_samples_leaf']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Using Early stopping strategy to calculate no. of estimators \n",
    "required in order to decrease computations and overfitting\n",
    "\n",
    "'''\n",
    "\n",
    "c_gbrt_est = GradientBoostingClassifier(max_depth=8,min_samples_split=9,min_samples_leaf=8,learning_rate=0.09,warm_start=True)\n",
    "min_val_error = float(\"inf\")\n",
    "error_going_up = 0\n",
    "n_estimators = 0\n",
    "for n_estimators in range(1,1200):\n",
    "    c_gbrt_est.n_estimators = n_estimators\n",
    "    c_gbrt_est.fit(cX_real_drop,cy_real_drop)\n",
    "    y_pred = c_gbrt_est.predict(cX_test_drop)\n",
    "    val_error = mean_squared_error(cy_test_drop,y_pred)\n",
    "    if val_error<min_val_error:\n",
    "        min_val_error = val_error\n",
    "        error_going_up = 0\n",
    "    else:\n",
    "        error_going_up +=1\n",
    "        if error_going_up==15: \n",
    "            break \n",
    "# A minimum of 100 estimators for the classifier            \n",
    "if n_estimators<100:\n",
    "    n_estimators = 100\n",
    "clf_estimator = n_estimators"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "\n",
    "Function : \n",
    "            Sets the risk = 0 values calculated by classifier in the respective predictions made by the \n",
    "            regressor\n",
    "            \n",
    "            Also rounds off predictions\n",
    "            \n",
    "            Fills up predictions which are negative and classifier says risk!=0 with the average\n",
    "            calculated in this case with diff combination and diff random seeds in train,val and \n",
    "            test set.\n",
    "\n",
    "'''\n",
    "\n",
    "def precise_pred(reg_pred,clf_pred,avg):\n",
    "    reg =reg_pred.copy() \n",
    "    clf = clf_pred.copy()\n",
    "    reg[clf==0] = 0\n",
    "    reg[(reg>0) & (reg<=0.5)] = 1\n",
    "    reg[(reg<0) & (clf!=0)] = avg # Filling the negative predictions made by regressor with average values in the case where my cross validation test used to give negative predictions instead of positive value\n",
    "    for i in range(len(reg)):\n",
    "        \n",
    "        reg[i] = round(reg[i])\n",
    "    \n",
    "    return reg\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculation of the average\n",
    "\n",
    "def cal_avg(y_re,reg,clf):\n",
    "    y_real = np.array(y_re)\n",
    "    count = 0\n",
    "    av = 0\n",
    "    for i in range(len(y_real)):\n",
    "        if (-y_real[i]<0) and (y_real[i]!=0) and (clf[i]!=0) and (round(reg[i])<=0):\n",
    "            av = av + y_real[i]\n",
    "            count+=1\n",
    "    return round((av/count))        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "\n",
    "Returns the Model of Regressor and Classifier after fitting data\n",
    "\n",
    "NOTE: If new training data is feeded grid search for the best parameter first as \n",
    "      variables declared above will be used else default calculated paramter would\n",
    "      be used\n",
    "'''\n",
    "\n",
    "def reg_clf_model_fit(reg_train,reg_y,clf_train,clf_y):\n",
    "    gbrt_reg = GradientBoostingRegressor(n_estimators=reg_estimator,max_depth=reg_depth,min_impurity_decrease=reg_impurity,min_samples_leaf=reg_leaf,min_samples_split=reg_split,learning_rate=reg_learn)\n",
    "    gbrt_clf = GradientBoostingClassifier(n_estimators=clf_estimator,max_depth=clf_depth,min_samples_leaf=clf_leaf,min_samples_split=clf_split,learning_rate=clf_learn)\n",
    "    \n",
    "    gbrt_reg.fit(reg_train,reg_y)\n",
    "    gbrt_clf.fit(clf_train,clf_y)\n",
    "    return gbrt_reg,gbrt_clf\n",
    "    \n",
    "def svc_model_fit(clf_train,clf_y):\n",
    "    rbf = SVC(kernel = \"rbf\",gamma=0.096,C=100)\n",
    "       \n",
    "    rbf.fit(clf_train,clf_y)\n",
    "    return rbf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Returns the Prediction of Regressor and Classifier\n",
    "\n",
    "'''\n",
    "\n",
    "def model_prediction(reg_model,clf_model,reg_data,clf_data):\n",
    "    return reg_model.predict(reg_data),clf_model.predict(clf_data)\n",
    "\n",
    "def svc_pred(rbf_model,rbf_data):\n",
    "    return rbf_model.predict(rbf_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Average Calculation Process\n",
    "'''\n",
    "REG,CLF = reg_clf_model_fit(X_real,y_real,cX_real,cy_real)\n",
    "reg_pred_1,clf_pred_1 = model_prediction(REG,CLF,X_test,cX_test)\n",
    "av1 = cal_avg(y_test,reg_pred_1,clf_pred_1)\n",
    "REG,CLF = reg_clf_model_fit(X_train,y_train,cX_train,cy_train)\n",
    "reg_pred_1,clf_pred_1 = model_prediction(REG,CLF,X_val,cX_val)\n",
    "av2 = cal_avg(y_val,reg_pred_1,clf_pred_1)\n",
    "REG,CLF = reg_clf_model_fit(X_train,y_train,cX_train,cy_train)\n",
    "reg_pred_1,clf_pred_1 = model_prediction(REG,CLF,X_test,cX_test)\n",
    "av3 = cal_avg(y_test,reg_pred_1,clf_pred_1)\n",
    "av = (av1+av2+av3)/3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, ..., 1, 1, 1])"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "48.333333333333336"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [],
   "source": [
    "### FINAL  PREDICTIONS ###\n",
    "\n",
    "test_reg = reg_remove_scale(test)\n",
    "test_clf = clf_remove_scale(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [],
   "source": [
    "REG_test,CLF_test = reg_clf_model_fit(X_reg,y,X_clf,y_clf)\n",
    "reg_test_pred,clf_test_pred = model_prediction(REG_test,CLF_test,test_reg,test_clf)\n",
    "SVC_test = svc_model_fit(X_clf,y_clf)\n",
    "svc_test_pred = svc_pred(SVC_test,test_clf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "7.747061811050437   0\n",
      "1.2149084181990626   0\n",
      "1.3001291639153978   0\n",
      "2.1084081441663747   0\n",
      "0.5451988058669108   0\n",
      "44.20081777146426   0\n",
      "7.601293441314508   0\n",
      "3.129513271924518   0\n",
      "11.175346418951651   0\n",
      "3.584311637941277   0\n",
      "10.051686876742698   0\n",
      "6.132491842088077   0\n",
      "2.1084081441663747   0\n",
      "1.7693275509112958   0\n",
      "3.1937542944987953   0\n",
      "2.5345537102851403   0\n",
      "2.1084081441663747   0\n",
      "7.786715848682884   0\n",
      "5.614392044477237   0\n",
      "7.192767682872381   0\n",
      "2.1270068212566557   0\n",
      "6.104121166436553   0\n",
      "50.10780680966881   0\n",
      "5.06049067785108   0\n",
      "0.2855042252449778   0\n",
      "3.868617615125239   0\n",
      "2.078531745928417   0\n",
      "2.1084081441663747   0\n",
      "2.183987786055961   0\n",
      "10.051686876742698   0\n",
      "7.601293441314508   0\n",
      "3.617733162109447   0\n",
      "2.5485736746618026   0\n",
      "3.0595369105602184   0\n",
      "3.0595369105602184   0\n",
      "2.371044950449019   0\n",
      "6.104121166436553   0\n",
      "3.8248040505833276   0\n",
      "41.886726108134205   0\n",
      "2.5329413513118815   0\n",
      "13.363139027478015   0\n",
      "3.1937542944987953   0\n",
      "1.8450993105517153   0\n",
      "18.38569481594263   0\n",
      "2.967325470651004   0\n",
      "6.104121166436553   0\n",
      "3.0595369105602184   0\n",
      "9.005143681769008   0\n",
      "27.53638558473449   0\n",
      "4.012204082987685   0\n",
      "3.068440550834888   0\n",
      "6.34577965285024   0\n",
      "9.10923671192607   0\n",
      "24.786690945862585   0\n",
      "4.429196383425921   0\n",
      "3.1937542944987953   0\n",
      "2.9808273986067357   0\n",
      "2.6216008337992336   0\n",
      "7.682598900665037   0\n",
      "0.6425958586958073   0\n",
      "9.10923671192607   0\n",
      "2.464421641661721   0\n",
      "7.192767682872381   0\n",
      "2.509885501163129   0\n",
      "2.1140015012831346   0\n",
      "19.816973116084426   0\n",
      "13.580286259445119   0\n",
      "11.231345823506741   0\n",
      "2.1084081441663747   0\n",
      "3.0595369105602184   0\n",
      "3.1937542944987953   0\n",
      "3.1937542944987953   0\n",
      "2.465501979529716   0\n",
      "5.76351991862354   0\n",
      "0.9365375721306878   0\n",
      "2.1084081441663747   0\n",
      "9.156147342403433   0\n",
      "0.4811330520742074   0\n",
      "0.7226293297593965   0\n",
      "8.764091393312183   0\n",
      "1.2579986288954894   0\n",
      "11.640997071801076   0\n",
      "7.325601002334598   0\n",
      "8.185848043874548   0\n",
      "3.0595369105602184   0\n",
      "2.371044950449019   0\n",
      "56.33635206969938   0\n",
      "12.119973958416322   0\n",
      "4.435558314358492   0\n",
      "6.104121166436553   0\n",
      "52.84674467527987   0\n",
      "9.10923671192607   0\n",
      "52.72680113513702   0\n",
      "15.35201695931718   0\n",
      "2.1084081441663747   0\n",
      "0.9998552613031414   0\n",
      "8.007270999877262   0\n",
      "1.8768337637241994   0\n",
      "11.051120435831555   0\n",
      "1.3001291639153978   0\n",
      "1.4567390752964666   0\n",
      "3.382308749620265   0\n",
      "42.13267099986158   0\n",
      "1.335157830427958   0\n",
      "6.25428309263472   0\n",
      "2.1084081441663747   0\n",
      "7.503189374747244   0\n",
      "6.839447159847438   0\n",
      "5.06049067785108   0\n",
      "2.1270068212566557   0\n",
      "5.811237532874222   0\n",
      "3.1937542944987953   0\n",
      "3.0595369105602184   0\n",
      "2.4488957079012463   0\n",
      "7.50753295663196   0\n",
      "3.0595369105602184   0\n",
      "13.262748734154188   0\n",
      "7.557455316363669   0\n",
      "5.172957607696216   0\n",
      "1.9904793164103514   0\n",
      "0.2855042252449778   0\n",
      "6.104121166436553   0\n",
      "0.531164719966765   0\n",
      "16.5300275238004   0\n",
      "8.97370054764801   0\n",
      "0.3615048654509324   0\n",
      "2.1084081441663747   0\n",
      "4.372854873678659   0\n",
      "41.50357437599987   0\n",
      "2.7693819436857057   0\n",
      "1.3453918580574828   0\n",
      "13.262748734154188   0\n",
      "13.03449151400668   0\n",
      "0.9309640240921766   0\n",
      "0.04849063132731758   0\n",
      "3.5322210521894792   0\n",
      "7.747061811050437   0\n",
      "7.672266151620448   0\n",
      "3.4638646451059576   0\n",
      "2.5032224239590297   0\n",
      "0.7390890736248301   0\n",
      "2.7635523838697313   0\n",
      "2.464421641661721   0\n",
      "8.957782453217888   0\n",
      "2.8101740295419857   0\n",
      "14.506940234927013   0\n",
      "6.104121166436553   0\n",
      "9.10923671192607   0\n",
      "2.8101740295419857   0\n",
      "10.828829920672293   0\n",
      "6.104121166436553   0\n",
      "7.325601002334598   0\n",
      "6.104121166436553   0\n",
      "1.6397258152842749   0\n",
      "14.506940234927013   0\n",
      "1.3453918580574828   0\n",
      "2.2624150924933746   0\n",
      "20.941601708558196   0\n",
      "5.313568446710736   0\n",
      "14.900218820209037   0\n",
      "2.1084081441663747   0\n",
      "6.5044366080076355   0\n",
      "1.502122164272792   0\n",
      "5.660327806714   0\n",
      "2.5904942320177753   0\n",
      "6.104121166436553   0\n",
      "7.50753295663196   0\n",
      "11.087012734195913   0\n",
      "8.97370054764801   0\n",
      "3.0595369105602184   0\n",
      "4.754717166994674   0\n",
      "3.2651771151026967   0\n",
      "4.286874483031712   0\n",
      "42.31750460428378   0\n",
      "3.1937542944987953   0\n",
      "2.211040148245405   0\n",
      "6.749470830902052   0\n",
      "3.0595369105602184   0\n",
      "4.517673352181006   0\n",
      "9.10923671192607   0\n",
      "4.748175610284291   0\n",
      "3.0595369105602184   0\n",
      "8.983348826477112   0\n",
      "2.826927377698727   0\n",
      "9.237631192658093   0\n",
      "43.69767690043043   0\n",
      "3.4751391750136857   0\n",
      "4.752792781074712   0\n",
      "20.941601708558196   0\n",
      "3.1937542944987953   0\n",
      "8.007270999877262   0\n",
      "2.1084081441663747   0\n",
      "6.294387662989683   0\n",
      "5.493029796474701   0\n",
      "3.0595369105602184   0\n",
      "8.983348826477112   0\n",
      "3.1937542944987953   0\n",
      "3.0595369105602184   0\n",
      "5.07122941834224   0\n",
      "9.10923671192607   0\n",
      "2.3344707267174725   0\n",
      "9.10923671192607   0\n",
      "2.464421641661721   0\n",
      "2.2624150924933746   0\n",
      "6.104121166436553   0\n",
      "18.38569481594263   0\n",
      "2.1084081441663747   0\n",
      "1.7548357446925364   0\n",
      "1.9904793164103514   0\n",
      "11.647652860617962   0\n",
      "3.1937542944987953   0\n",
      "7.1185598786041595   0\n",
      "2.1084081441663747   0\n",
      "0.27453606833698185   0\n",
      "3.1937542944987953   0\n",
      "3.1937542944987953   0\n",
      "3.573522562526537   0\n",
      "2.1084081441663747   0\n",
      "6.677023553495048   0\n",
      "2.1084081441663747   0\n",
      "6.66459263374229   0\n",
      "3.486149507145077   0\n",
      "10.000948222030559   0\n",
      "0.6061702386852295   0\n",
      "9.547911887689692   0\n",
      "1.073396566826025   0\n",
      "1.5583802958492088   0\n",
      "5.660327806714   0\n",
      "3.0595369105602184   0\n",
      "0.5451988058669108   0\n",
      "8.876536376207609   0\n",
      "1.0585082403289918   0\n",
      "2.1084081441663747   0\n",
      "9.10923671192607   0\n",
      "1.335157830427958   0\n",
      "39.966171583422266   0\n",
      "10.300478281405008   0\n",
      "13.03449151400668   0\n",
      "9.10923671192607   0\n",
      "2.762121959899116   0\n",
      "6.809529270457641   0\n",
      "12.070773945369446   0\n",
      "2.3462939507812424   0\n",
      "3.094997734940736   0\n",
      "6.104121166436553   0\n",
      "5.4818129244720195   0\n",
      "5.813529761319272   0\n",
      "4.429196383425921   0\n",
      "1.7548357446925364   0\n",
      "0.038862649603523605   0\n",
      "8.876536376207609   0\n",
      "5.875974896380927   0\n",
      "8.021527731845731   0\n",
      "1.7548357446925364   0\n",
      "12.460934742581761   0\n",
      "1.7783032407609056   0\n",
      "9.10923671192607   0\n",
      "1.8450993105517153   0\n",
      "38.68761991255147   0\n",
      "6.410895958786624   0\n",
      "3.0595369105602184   0\n",
      "12.20298162175252   0\n",
      "4.587668268225384   0\n",
      "11.022165083502458   0\n",
      "6.104121166436553   0\n",
      "3.7375601358791584   0\n",
      "6.104121166436553   0\n",
      "8.185848043874548   0\n",
      "4.991869655525377   0\n",
      "5.24386877741637   0\n",
      "0.4811330520742074   0\n",
      "1.2579986288954894   0\n",
      "14.106504303390793   0\n",
      "2.7859779847665256   0\n",
      "3.8730143405698043   0\n",
      "52.07630189774415   0\n",
      "9.52218196575687   0\n",
      "9.10923671192607   0\n",
      "35.26095845696869   0\n",
      "6.104121166436553   0\n",
      "2.1084081441663747   0\n",
      "2.6731700071259756   0\n",
      "19.816973116084426   0\n",
      "9.10923671192607   0\n",
      "2.8101740295419857   0\n",
      "2.7635523838697313   0\n",
      "2.244410022880511   0\n",
      "3.0595369105602184   0\n",
      "0.3615048654509324   0\n",
      "3.1937542944987953   0\n",
      "59.84125253829478   0\n",
      "14.900218820209037   0\n",
      "3.1937542944987953   0\n",
      "6.104121166436553   0\n",
      "3.1937542944987953   0\n",
      "9.248668133794071   0\n",
      "3.1937542944987953   0\n",
      "9.10923671192607   0\n",
      "12.835109758146599   0\n",
      "0.9998552613031414   0\n",
      "2.429343511797847   0\n",
      "3.129513271924518   0\n",
      "1.6031139658301938   0\n",
      "8.97370054764801   0\n",
      "7.503189374747244   0\n",
      "17.620307547557296   0\n",
      "19.14865700129176   0\n",
      "0.20176221702608443   0\n",
      "45.772905225532575   0\n",
      "13.337468248771248   0\n",
      "38.793697012566106   0\n",
      "2.9749241352027087   0\n",
      "6.893247381147509   0\n",
      "6.809529270457641   0\n",
      "3.0595369105602184   0\n",
      "3.1937542944987953   0\n",
      "34.50831019897309   0\n",
      "4.432631228772124   0\n",
      "2.963508616761353   0\n",
      "57.268740389576344   0\n",
      "10.351373070383262   0\n",
      "11.231345823506741   0\n",
      "6.7440219953937826   0\n",
      "4.341451641645443   0\n",
      "2.429780844710539   0\n",
      "6.331438427999201   0\n",
      "3.979241815833997   0\n",
      "3.246490823582467   0\n",
      "5.172957607696216   0\n",
      "13.363139027478015   0\n",
      "61.24649326398687   0\n",
      "9.10923671192607   0\n",
      "0.531164719966765   0\n",
      "8.070425488857317   0\n",
      "6.294387662989683   0\n",
      "59.384016797025744   0\n",
      "1.0585082403289918   0\n",
      "4.621052953123276   0\n",
      "64.4468728489748   0\n",
      "3.1937542944987953   0\n",
      "9.10923671192607   0\n",
      "9.10923671192607   0\n",
      "0.7066817579706306   0\n",
      "1.9904793164103514   0\n",
      "0.4811330520742074   0\n",
      "0.5708638319328327   0\n",
      "14.036805618235837   0\n",
      "4.556256597000577   0\n",
      "9.42014862206769   0\n",
      "15.968203777862673   0\n",
      "12.119973958416322   0\n",
      "4.949017901001809   0\n",
      "3.4788976508707528   0\n",
      "1.4771815374972117   0\n",
      "9.138707906450072   0\n",
      "9.156147342403433   0\n",
      "9.10923671192607   0\n",
      "9.10923671192607   0\n",
      "4.3558763150365305   0\n",
      "2.1084081441663747   0\n",
      "10.412604974434972   0\n",
      "7.786715848682884   0\n",
      "9.509053351919231   0\n",
      "9.297268818219655   0\n",
      "13.363139027478015   0\n",
      "5.18929074831896   0\n",
      "7.532353390616823   0\n",
      "6.104121166436553   0\n",
      "11.022165083502458   0\n",
      "5.367077423682531   0\n",
      "2.3462939507812424   0\n",
      "2.95935785867785   0\n",
      "3.0595369105602184   0\n",
      "1.073396566826025   0\n",
      "2.1084081441663747   0\n",
      "6.617391803807879   0\n",
      "4.754717166994674   0\n",
      "47.59339829606031   0\n",
      "0.8247311206338729   0\n",
      "8.876536376207609   0\n",
      "3.0595369105602184   0\n",
      "3.1937542944987953   0\n",
      "4.202036565265654   0\n",
      "11.175346418951651   0\n",
      "17.57678943488275   0\n",
      "0.038862649603523605   0\n",
      "31.48500631121027   0\n",
      "3.0595369105602184   0\n",
      "9.10923671192607   0\n",
      "6.6096215066291295   0\n",
      "1.3881654569222568   0\n",
      "0.6425958586958073   0\n",
      "3.382308749620265   0\n",
      "0.9365375721306878   0\n",
      "5.897026701418664   0\n",
      "2.5485736746618026   0\n",
      "8.20154634959925   0\n",
      "3.1937542944987953   0\n",
      "2.1084081441663747   0\n",
      "5.2531923739789   0\n",
      "11.647652860617962   0\n",
      "5.100466874067223   0\n",
      "2.470079313216973   0\n",
      "9.10923671192607   0\n",
      "6.466359968589916   0\n",
      "50.882750919392954   0\n",
      "44.800597119012146   0\n",
      "0.13672678865765775   0\n",
      "6.104121166436553   0\n",
      "0.8247311206338729   0\n",
      "6.104121166436553   0\n",
      "9.929116618503834   0\n",
      "0.14773711451201066   0\n",
      "11.175346418951651   0\n",
      "1.12224211087729   0\n",
      "6.1294860889692835   0\n",
      "6.584092338352555   0\n",
      "1.4771815374972117   0\n",
      "6.104121166436553   0\n",
      "6.410895958786624   0\n",
      "3.0595369105602184   0\n",
      "0.27453606833698185   0\n",
      "7.672266151620448   0\n",
      "2.371044950449019   0\n",
      "9.509053351919231   0\n",
      "13.03449151400668   0\n",
      "6.104121166436553   0\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "427"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output = precise_pred(reg_test_pred,clf_test_pred,av)\n",
    "svc_output = precise_pred(reg_test_pred,svc_test_pred)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}