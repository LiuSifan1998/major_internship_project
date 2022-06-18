# train classifiers on oversampling processed class-balanced dataset
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
import imblearn
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from scipy import stats
from math import sqrt
import os
import sys

classifier = sys.argv[1]
transformer = sys.argv[2]
transformer_train = sys.argv[3]
transformer_test = sys.argv[4]
# classifier type: lr20, lr80, knn20, kNN80, rf
# transformer type: baseline, esm, bert, xl, xknet
# baseline train set /projects/qbio/bifo/ProtProduction/khld071/order_encode_output/final_model/cdhit.csv
# baseline test set /projects/qbio/bifo/ProtProduction/khld071/order_encode_output/final_model/cdhit_remove.csv
# esm train set /projects/qbio/bifo/ProtProduction/khld071/esm_output/datasets/cdhit25_esm.csv
# esm test set /projects/qbio/bifo/ProtProduction/khld071/esm_output/cdhit_removed.csv
def type_transformer(transformer_name,classifier_name,file_train,file_test):
    df1 = pd.read_csv(file_train,low_memory=False,index_col = 'id')
    df2 = pd.read_csv(file_test,low_memory=False,index_col = 'id')
    if transformer_name == 'esm':
        output_file = '/projects/qbio/bifo/ProtProduction/khld071/esm_output/final_model/scores/score4final' + classifier_name + '_balance_ovs.csv'
    if transformer_name == 'bert':
        output_file = '/projects/qbio/bifo/ProtProduction/khld071/protransbert_output/final_model/score/score4final' + classifier_name + '_balance_ovs.csv'
    if transformer_name == 'xl':
        output_file = '/projects/qbio/bifo/ProtProduction/khld071/protransxl_output/final_model/score/score4final' + classifier_name + '_balance_ovs.csv'
    if transformer_name == 'xlnet':
        output_file = '/projects/qbio/bifo/ProtProduction/khld071/protransxlnet_output/final_model/score/score4final' + classifier_name + '_balance_ovs.csv'
    return(df1,df2,output_file)

def type_classifier(classifiername):
    if classifiername == 'rf':
        downstream = RandomForestClassifier()
    if classifiername == 'lr20':
        downstream = LogisticRegression(warm_start = True, solver = 'saga', penalty = 'l2', multi_class = 'ovr', max_iter = 500, fit_intercept = True)
    if classifiername == 'lr80':
        downstream = LogisticRegression(warm_start = False, solver = 'saga', penalty = 'none', multi_class = 'ovr', max_iter = 5000, fit_intercept = False)
    if classifiername == 'knn20':
        downstream = KNeighborsClassifier(weights='uniform',p=1,n_neighbors=5,leaf_size=50)
    if classifiername == 'knn80':
        downstream = KNeighborsClassifier(weights='distance',p=1,n_neighbors=50,leaf_size=3000)
    return(downstream)

train_df, test_df, output_path = type_transformer(transformer,classifier,transformer_train,transformer_test)
clf = type_classifier(classifier)

df = train_df[train_df.symbol != 'na']
for ind,row in df.iterrows():
    df['symbol'][ind] = int(float(row['symbol']))
X = df.drop(['symbol'], axis = 1)
Y = df.iloc[:,-1]
#oversample = RandomOverSampler(sampling_strategy='minority')
#X, Y = oversample.fit_resample(X, Y)
oversample = SMOTE()
X,Y = X.to_numpy(),Y.tolist()
X, Y = oversample.fit_resample(X, Y)


test_df = test_df[test_df.symbol != 'na']
test_df = test_df.sample(frac=1).reset_index(drop=True)
for ind,row in test_df.iterrows():
    test_df['symbol'][ind] = int(float(row['symbol']))
test_X = test_df.drop(['symbol'], axis = 1)
test_Y = test_df.iloc[:,-1]
test_X,test_Y = test_X.to_numpy(),test_Y.tolist()

df1 = pd.DataFrame(columns=['acc','recall','precision','f1','mcc','auc','tp','fn','fpr','tpr','threshold'])
acc_inner = []
auc_inner =[]
mcc_inner = []
recall_inner = []
f1_inner = []
precision_inner = []
auc_inner = []
fpr_inner = []
tpr_inner = []
tp_inner = []
fn_inner = []
threshold_inner = []

clf.fit(X, Y)
    
ypre_f = clf.predict(test_X)
acc_f = accuracy_score(test_Y, ypre_f) 
acc_inner.append(acc_f)
matrix_f = confusion_matrix(test_Y, ypre_f)
TP_f = matrix_f[1][1]
FP_f = matrix_f[0][1]
FN_f = matrix_f[1][0]
TN_f = matrix_f[0][0]
numerator = (TP_f * TN_f) - (FP_f * FN_f)
denominator = sqrt((TP_f+FP_f)*(TP_f+FN_f)*(TN_f+FP_f)*(TN_f+FN_f))
MCC_f = numerator/denominator
mcc_inner.append(MCC_f)
fn_inner.append(FN_f)
tp_inner.append(TP_f)
recall_inner.append(recall_score(test_Y, ypre_f))
f1_inner.append(f1_score(test_Y, ypre_f))
precision_inner.append(precision_score(test_Y, ypre_f))
fpr_f,tpr_f,threshold_keras_f = roc_curve(test_Y, ypre_f)
fpr_inner.append(fpr_f)
tpr_inner.append(tpr_f)
auc_inner.append(roc_auc_score(test_Y, ypre_f))
threshold_inner.append(threshold_keras_f)   
df1['acc'] = acc_inner
df1['f1'] = f1_inner
df1['auc'] = auc_inner
df1['recall'] = recall_inner
df1['mcc'] = mcc_inner
df1['precision'] = precision_inner
df1['fpr'] = fpr_inner
df1['tpr'] = tpr_inner
df1['threshold'] = threshold_inner
df1['fn'] = fn_inner
df1['tp'] = tp_inner

df1.to_csv(output_path,sep=',',index=False,header=True)




    

