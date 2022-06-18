# tune parameters for classifiers
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2,SelectFpr,f_classif,mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import recall_score,accuracy_score,roc_auc_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix,classification_report, roc_curve
from collections import Counter
from tqdm import tqdm
from scipy import stats
from math import sqrt
import os
import sys

classifier = sys.argv[1]
transformer = sys.argv[2]
dataset1 = sys.argv[3]
dataset2 = sys.argv[4]
# classifier type: 20K_LR, 80K_LR, 20K_knn, 80K_kNN, 20K_RF, 80K_RF
# transformer type: esm
# esm 20K dataset /projects/qbio/bifo/ProtProduction/khld071/esm_output/datasets/cdhit25_esm.csv
# esm 80K dataset /projects/qbio/bifo/ProtProduction/khld071/esm_output/datasets/esm_reps.csv
def type_transformer(transformer_name,classifier_name,file1,file2):
    if '20' in classifier_name:
        esm_df = pd.read_csv(file1,low_memory=False,index_col = 'id')
    if '80' in classifier_name:
        esm_df = pd.read_csv(file2,low_memory=False,index_col = 'id')
    if transformer_name == 'esm':
        output_file1 = '/projects/qbio/bifo/ProtProduction/khld071/esm_output/para_tune/score4trainset' + classifier_name + '.csv'
        output_file2 = '/projects/qbio/bifo/ProtProduction/khld071/esm_output/para_tune/score4testset' + classifier_name + '.csv'

    return(esm_df,output_file1,output_file2)

def type_classifier(classifiername):
    if classifiername == 'rf':
        param = {"n_estimators": [100, 500, 1000, 3000, 5000],"max_features": [1, 10, 50, 100, 500, 1280],"min_samples_split": [1, 5, 10, 20, 50, 100],"min_samples_leaf": [1, 5, 10, 20, 50, 100],"bootstrap": [True, False]}
        downstream = RandomForestClassifier()
    if classifiername == 'lr20' or classifiername == 'lr80':
        param = {"penalty": ['l1','l2','elasticnet','none'],"fit_intercept": [True,False],"solver": ['saga'],"max_iter": [100, 200, 500, 1000, 3000, 5000],"multi_class": ['ovr'],"warm_start": [True,False]}
        downstream = LogisticRegression()
    if classifiername == 'knn20' or classifiername == 'knn80':
        param = {'n_neighbors': [5, 50, 100, 200, 500, 1000, 1500, 3000, 5000],'weights': ['distance','uniform'],"leaf_size": [30, 50, 100, 200, 500, 1000, 3000],"p": [1, 2, 10],"n_jobs": [-1]}
        downstream = KNeighborsClassifier()
    return(downstream, param)

data_df, output_path1, output_path2 = type_transformer(transformer,classifier,dataset)
clf, para_grid = type_classifier(classifier)

df = esm_df[esm_df.symbol != 'na']
df_0 = df[df.symbol == '0.0'].reset_index(drop=True)
df_1 = df[df.symbol == '1.0'].reset_index(drop=True)
sub_df = df_1.append(df_0)
df = sub_df.sample(frac=1).reset_index(drop=True)
for ind,row in df.iterrows():
    df['symbol'][ind] = int(float(row['symbol']))
X = df.drop(['symbol'], axis = 1)
Y = df.iloc[:,-1]
X,Y = X.to_numpy(),Y.tolist()
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.3,stratify=Y)
test_y = test_y.tolist()
train_y = train_y.tolist()

df1 = pd.DataFrame(columns=['para','acc','recall','precision','f1','mcc','auc','tp','fn','threshold','fpr','tpr'])
df2 = pd.DataFrame(columns=['para','acc','recall','precision','f1','mcc','auc','tp','fn','threshold','fpr','tpr'])

int_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state= 42)
inner_para = []
acc_inner,acc_l = [],[]
auc_inner,auc_l =[],[]
mcc_inner,mcc_l = [],[]
recall_inner,recall_l = [],[]
f1_inner,f1_l = [],[]
precision_inner,precision_l = [],[]
auc_inner,acc_l = [],[]
fpr_inner,fpr_l = [],[]
tpr_inner,tpr_l = [],[]
tp_inner,tp_l = [],[]
fn_inner,fn_l = [],[]
threshold_inner,threshold_l = [],[]

for train_i_index, test_i_index in int_cv.split(train_x, train_y):
    X_train_i, X_test_i = train_x[train_i_index], train_x[test_i_index]
    y_train_i, y_test_i = [],[]
    for ele in range(len(train_i_index)):
        y_train_i.append(train_y[train_i_index[ele]])
    for ele in range(len(test_i_index)):
        y_test_i.append(train_y[test_i_index[ele]])
    c = clf
    grid_search = RandomizedSearchCV(estimator = c, param_distributions = param_grid, cv = 5, n_jobs = -1, verbose = 2)
    grid_search.fit(X_train_i, y_train_i)
    clf = clf(**grid_search.best_params_)
    clf.fit(X_train_i, y_train_i)
    inner_para.append(grid_search.best_params_)
            
    ypre = clf.predict(X_test_i)
    acc = accuracy_score(y_test_i, ypre) 
    acc_inner.append(acc)
    matrix = confusion_matrix(y_test_i, ypre)
    TP = matrix[1][1]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TN = matrix[0][0]
    numerator = (TP * TN) - (FP * FN)
    denominator = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    MCC = numerator/denominator
    mcc_inner.append(MCC)
    fn_inner.append(FN)
    tp_inner.append(TP)
    recall_inner.append(recall_score(y_test_i, ypre))
    f1_inner.append(f1_score(y_test_i, ypre))
    precision_inner.append(precision_score(y_test_i, ypre))
    fpr,tpr,threshold_keras = roc_curve(y_test_i, ypre)
    fpr_inner.append(fpr)
    tpr_inner.append(tpr)
    auc_inner.append(roc_auc_score(y_test_i, ypre))
    threshold_inner.append(threshold_keras)
    
    #clf_final = KNeighborsClassifier(**grid_search.best_params_)
    #clf_final.fit(train_x, train_y)
    ypre_f = clf.predict(test_x)
    acc_f = accuracy_score(test_y, ypre_f) 
    acc_l.append(acc_f)
    matrix_f = confusion_matrix(test_y, ypre_f)
    TP_f = matrix_f[1][1]
    FP_f = matrix_f[0][1]
    FN_f = matrix_f[1][0]
    TN_f = matrix_f[0][0]
    numerator = (TP_f * TN_f) - (FP_f * FN_f)
    denominator = sqrt((TP_f+FP_f)*(TP_f+FN_f)*(TN_f+FP_f)*(TN_f+FN_f))
    MCC_f = numerator/denominator
    mcc_l.append(MCC_f)
    fn_l.append(FN_f)
    tp_l.append(TP_f)
    recall_l.append(recall_score(test_y, ypre_f))
    f1_l.append(f1_score(test_y, ypre_f))
    precision_l.append(precision_score(test_y, ypre_f))
    fpr_f,tpr_f,threshold_keras_f = roc_curve(test_y, ypre_f)
    fpr_l.append(fpr_f)
    tpr_l.append(tpr_f)
    auc_l.append(roc_auc_score(test_y, ypre_f))
    threshold_l.append(threshold_keras_f)
    print('LR:')
    print('\n')
    print(confusion_matrix(test_y, ypre_f))
    print('\n')
    print(classification_report(test_y, ypre_f))
    print('\n')

    
df1['acc'] = acc_inner
df1['f1'] = f1_inner
df1['auc'] = auc_inner
df1['recall'] = recall_inner
df1['mcc'] = mcc_inner
df1['precision'] = precision_inner
df1['para'] = inner_para
df1['fpr'] = fpr_inner
df1['tpr'] = tpr_inner
df1['threshold'] = threshold_inner
df1['fn'] = fn_inner
df1['tp'] = tp_inner

df2['acc'] = acc_l
df2['f1'] = f1_l
df2['auc'] = auc_l
df2['recall'] = recall_l
df2['mcc'] = mcc_l
df2['precision'] = precision_l
df2['para'] = inner_para
df2['fpr'] = fpr_l
df2['tpr'] = tpr_l
df2['threshold'] = threshold_l
df2['fn'] = fn_l
df2['tp'] = tp_l

df1.to_csv(output_path1,sep=',',index=False,header=True)
df2.to_csv(output_path2,sep=',',index=False,header=True)





    

