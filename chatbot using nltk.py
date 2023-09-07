# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:44:29 2023

@author: my pc
"""
import cv2
import nltk
from nltk.chat.util import Chat,reflections

pairs=[[
        "(.*) my name is (.*)",
        ["Hii %2,how are you"]],
       ["(.*) help(.*)",
        ["How may i help you"]],
       ["(.*)regression alogrithms(.*)",
        ["These are regression alogrithms :linear regression,multiple linear regression,polynomial regression,support vector regressor,kneareast neighbors regressor,Decision tree regression,random forest regression"]],
       ["(.*)classification alogrithms(.*)",
        ["logistic regression,support vector classification,naive_bayes classification,knearest neigbors classifier,decisiontree classifier,random forest classifier"]],
       ["(.*)clustering alogrithms(.*)",
        ["we have two clustering those are: K_means Clustering and Agglomerative clustering"]],
       ["(.*)most important thing in clustering(.*)",
        ["we have draw first elbow graph by using the elbow graph we can find the kvalues by using these values we can build the models"]],
       ["(.*)define the dataset belong with alogrithms(.*)",
        ["Based on the depedent varibule (Target varibules(y)) we can decide the dataset belong which category.if the depedent varibule is continous there we can use regression alogrithms ,if the depedent varibules is binary there we can use classification alogrithms ,if there is no depedent varibule we can use clustering alogrithms"]],
       ["(.*)find regression model is perfect(.*)",
        ["If the model is regression model we can calculate the mse and r2_score for the model.Based on the mse(mean_squared_error) and r2_score we can find the perfect fit model .Mse value should be high and r2_score should be low"]],
       ["(.*)find classification model is perfect(.*)",
        ["We can calculate the best fit model for classification by using the accuracy_score and confusion_matrix and classification_report"]],
       ["(.*)package used in machine learnings(.*)",
        ["Sklearn(scikit-learn) is package used in the meachine learnings"]],
       ["(.*)code for svm regression(.*)",
        ["""from sklearn.svm import SVR
         svr=SVR()
         svc.fit(X_train,y_train)
         y_pred=svc.predict(X_train)
         by using these we first install the sklearn by uisng pip (pip install sklearn),In place of X_train,and y_train,you can assign your own train data Inplace of SVC you can use the any other alogrithms as well like knn,decsiontree,randomforest,boosting alogrithms"""]],
         ["(.*)code for knn regression(.*)",
          ["""from sklearn.neighbors import KNeighborsRegressor
           knn=KNeighborsRegressor()
           knn.fit(X_train,y_train)
           y_pred=knn.predict(X_train)
           by using these we first install the sklearn by uisng pip (pip install sklearn),In place of X_train,and y_train,you can assign your own train data Inplace of knn you can use the any other alogrithms as well like svr,decsiontree,randomforest,boosting alogrithms"""]],
           ["(.*)code for decisiontree regression(.*)",
            ["""from sklearn.tree import DecionTreeRegressor
             tree=DecisionTreeRegressor()
             tree.fit(X_train,y_train)
             y_pred=tree.predict(X_train)
             by using these we first install the sklearn by uisng pip (pip install sklearn),In place of X_train,and y_train,you can assign your own train data Inplace of decisiontree you can use the any other alogrithms as well like svr,knn,randomforest,boosting alogrithms"""]],
             ["(.*)code for randomforest regression(.*)",
              ["""from sklearn.ensemble import RandomForestRegressor
               rndf=RandomForestRegressor()
               rndf.fit(X_train,y_train)
               y_pred=rndf.predict(X_train)
               by using these we first install the sklearn by uisng pip (pip install sklearn),In place of X_train,and y_train,you can assign your own train data Inplace of RandomForest you can use the any other alogrithms as well like svr,knn,decsiontree,boosting alogrithms"""]],
               ["(.*)code for xgboost regression(.*)",
                ["""from xgboost import XGBRegressor
                 xgb=XGBRegressor()
                 xgb.fit(X_train,y_train)
                 y_pred=xgb.predict(X_train)"""]],
                 ["(.*)code for svm classification(.*)",
                  ["""from sklearn.svm import SVC
                   svc=SVC()
                   svc.fit(X_train,y_train)
                   y_pred=svc.predict(X_train)
                   by using these we first install the sklearn by uisng pip (pip install sklearn),In place of X_train,and y_train,you can assign your own train data Inplace of svc you can use the any other alogrithms as well like knn,decisiontree,randomforest,boosting alogrithms"""]],
                   ["(.*)code for knn classification(.*)",
                    ["""from sklearn.neighbors import KNeighborsClassifier
                     knn=KNeighborsClassifier()
                     knn.fit(X_train,y_train),
                     y_pred=knn.predict(X_train)
                     by using these we first install the sklearn by uisng pip (pip install sklearn),In place of X_train,and y_train,you can assign your own train data Inplace of knn you can use the any other alogrithms as well like svc,decsiontree,randomforest,boosting alogrithms"""]],
                     ["(.*)code for decisiontree classification(.*)",
                      ["""from sklearn.tree import DecisionTreeClassifier
                       tree=DecisionTreeClassifier()
                       tree.fit(X_train,y_train),
                       y_pred=tree.predict()
                       by using these we first install the sklearn by uisng pip (pip install sklearn),In place of X_train,and y_train,you can assign your own train data Inplace of decisiontree you can use the any other alogrithms as well like svr,knn,randomforest,boosting alogrithms"""]],
                       ["(.*)code for randomforest classification",
                        ["""from sklearn.ensemble import RandomForestClassifier
                         rndf=RandomForestClassifier()
                         rndf.fit(X_train,y_train)
                         y_pred=rndf.predict(X_train)
                         by using these we first install the sklearn by uisng pip (pip install sklearn),In place of X_train,and y_train,you can assign your own train data Inplace of randomforest you can use the any other alogrithms as well like svr,knn,decisiontree,boosting alogrithms"""]],
                         ["(.*)code for xgboost classification",
                          ["""from xgboost import XGBClassifier
                           xgb=XGBClassifier()
                           xgb.fit(X_train,y_train)
                           y_pred=xgb.predict(X_train)
                           by using these we first install the sklearn by uisng pip (pip install sklearn),In place of X_train,and y_train,you can assign your own train data Inplace of xgboost you can use the any other alogrithms as well like svr,knn,randomforest,decisiontree alogrithms"""]],
                           ["(.*)calculate meansquarederror(.*) r2_score(.*)",
                            ["""from sklearn.metrics import mean_squared_error ,r2_score
                             mse=mean_squared_error(y_train,y_pred)
                             r2=r2_score(y_train,y_pred)
                             print(mse)
                             print(r2)
                             we must install first sklearn(scikit-learn) package by using pip install sckitlearn we can calculate mse and r2_score for regression model here y_true and y_pred based on the model you build"""]],
                             ["(.*)calcualate accuracy(.*)confusionmatrix(.*)classificationreport(.*)",
                              ["""from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
                               acc=accuracy_score(y_train,y_pred)
                               print(acc)
                               cm=confusion_matrix(y_train,y_pred)
                               print(cm)
                               report=classification_report(y_train,y_pred)
                               print(report)
                               we must install skitlearn by using pip here in place y_train,y_pred you can use your own model data"""]],
                               ["r(.*)",
                                ["Sorry i am unable to respond can you check it and try again with right command"]]]
chatbot=Chat(pairs,(reflections))

print("Hii i am chabot how may i help you")
chatbot.converse()
