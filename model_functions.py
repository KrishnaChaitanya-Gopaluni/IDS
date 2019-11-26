from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from scipy import stats
import numpy as np
import random

def accuracy(test,y_pred):

    return accuracy_score(test[['classes']], y_pred, normalize=True)
    

def pres(test, y_pred,posLabel = 'normal.'):
    if posLabel == 'normal.':
        posLabel = 0
    if posLabel == 'attack':
        posLabel = 1        
    
    pres_wt = precision_score(test[['classes']], y_pred, labels=[posLabel], pos_label = posLabel,average ='weighted')
    pres_micro = precision_score(test[['classes']], y_pred, labels=[ posLabel], pos_label = posLabel,average ='micro')
    pres_macro = precision_score(test[['classes']], y_pred, labels=[posLabel], pos_label = posLabel,average ='macro')
    pres_binary = precision_score(test[['classes']], y_pred, labels=[posLabel], pos_label = posLabel,average ='binary')

    return [pres_wt,pres_micro, pres_macro , pres_binary]

    
def entropy_sampling(unlabeled_pairs, xgboost_model, sample_size = 48984):
    temp = xgboost_model.predict_proba(unlabeled_pairs).tolist()
    # print("now retriving uncertain samples using entropy sampling")
    uncertain = [stats.entropy(x, qk= None, base = 10) for x in temp]
    return  (-np.asarray(uncertain)).argsort()[:sample_size].tolist()
    
def get_last2max(temp):
    return (-np.asarray(temp)).argsort()[:2].tolist()

# modified margin sampling to support binary classification
def margin(temp,temp2):
    
    return temp2[temp[0]]- temp2[temp[1]]

      
def margin_sampling(unlabeled_pairs, xgboost_model, sample_size = 48984):

    temp = xgboost_model.predict_proba(unlabeled_pairs).tolist()
    # print("now retriving uncertain samples using margin sampling")
    uncertain = [margin(get_last2max(i),i) for i in temp]

    return (np.asarray(uncertain)).argsort()[:sample_size].tolist() # get least margin value


def random_sampling(unlabeled_pairs, xgboost_model, sample_size = 48984):
    
    return random.sample(range(len(unlabeled_pairs)),sample_size)
    
#upper confidance bound, a solution to the multiarmed bandit problem
def get_ucb(mean, n, c, t ):
    return [mean[i] + np.sqrt(np.log(t)/n[i])*c for i in range(len(mean)) ]