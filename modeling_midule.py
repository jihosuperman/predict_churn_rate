import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn import svm
from catboost import CatBoostClassifier, Pool
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import f_classif, chi2,SelectKBest

class ModelingModule():

    def __init__(self, path):
        self.df = None
        self.path = path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.model_name = None
        self.recall_train = None
        self.precision_train = None
        self.confusion_matrix_train = None
        self.recall_test = None
        self.precision_test = None
        self.confusion_matrix_test = None
        self.X = None
        self.y = None

    def getDataFrame(self, df):
        self.df = df

    def trainTestSplit(self):
        X = self.df.drop(columns=['msno_num','is_churn'])
        y = self.df[['is_churn']]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=13, stratify=self.y)
        print("Split Complete")

    def selectBest(self, num=None):
        selector = SelectKBest(score_func = chi2, k = num)
        self.X_train = selector.fit_transform(self.X_train, self.y_train)
        self.X_test = selector.transform(self.X_test)

    def overSampling(self):
        sm = SMOTE(random_state=13)
        self.X_train , self.y_train = sm.fit_resample(self.X_train, self.y_train)

    def minMaxScaling(self):
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)
        
    def standardScaling(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)
        
    def robustScaling(self):
        scaler = RobustScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)

    #의사결정나무
    def modelDecisionTree(self):
        out_tree = DecisionTreeClassifier()
        self.model = out_tree.fit(self.X_train, self.y_train)
        self.model_name = 'DecisionTree'

    #랜덤포레스트
    def modelRandomForest(self, max_depth=None):
        
        out_ran = RandomForestClassifier(max_depth=max_depth)
        self.model = out_ran.fit(self.X_train, self.y_train)
        self.model_name = 'RandomForest'

    #xgboost
    def modelXGBoost(self):

        out_xgb = XGBClassifier()
        self.model = out_xgb.fit(self.X_train, self.y_train)
        self.model_name = 'XGBoost'

    #lightgbm
    def modelLightGBM(self):

        out_lgbm = lgb.LGBMClassifier()
        self.model = out_lgbm.fit(self.X_train, self.y_train)
        self.model_name = 'LGBM'

    #logistic regression
    def modelLogisticRegression(self):

        out_lgr = LogisticRegression()
        self.model = out_lgr.fit(self.X_train, self.y_train)
        self.model_name = 'LogisticRegression'
    
    #svc
    def modelSVC(self):

        out_svc = svm.SVC()
        self.model = out_svc.fit(self.X_train, self.y_train)
        self.model_name = 'SVC'

    #catboost Classifier
    def modelCatboost(self):

        out_cat = CatBoostClassifier()
        self.model = out_cat.fit(self.X_train, self.y_train)
        self.model_name = 'Catboost'
        
    #나이브베이지안 - 가우시안 네이브베이즈
    def modelGaussianNB(self):

        out_gnb = GaussianNB()
        self.model = out_gnb.fit(self.X_train, self.y_train)
        self.model_name='GaussianNB'

    #나이브베이지안 - 베르누이 네이브베이즈 : 독립변수가 0 또는 1의 값을 가지는 경우
    def modelBernoulliNB(self):

        out_bnb = BernoulliNB()
        self.model = out_bnb.fit(self.X_train, self.y_train)
        self.model_name='BernoulliNB'

    #나이브베이지안 - 다항분포 네이브베이즈
    def modelMultinomialNB(self):
        
        out_mnb = MultinomialNB()
        self.model = out_mnb.fit(self.X_train, self.y_train)
        self.model_name='MultinomialNB'
    
    def saveModel(self, num):
        joblib.dump(self.model, self.path+f'/{self.model_name}_test_{num}.pkl')
        
    def loadModel(self, filename):
        self.model = joblib.load(self.path+filename)

    def modelResult(self):
        y_pred_tr = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        precision_train = precision_score(self.y_train, y_pred_tr)
        precision_test = precision_score(self.y_test, y_pred_test)
        recall_train = recall_score(self.y_train, y_pred_tr)
        recall_test = recall_score(self.y_test, y_pred_test)
        train_f1 =  (2 *precision_train *recall_train) / (precision_train + recall_train)
        test_f1 =  (2 *precision_test *recall_test) / (precision_test + recall_test)
        confusion_matrix_train = confusion_matrix(self.y_train, y_pred_tr)
        confusion_matrix_test = confusion_matrix(self.y_test, y_pred_test)
        return precision_train , precision_test , recall_train,  recall_test, train_f1, test_f1, confusion_matrix_train, confusion_matrix_test
        
        