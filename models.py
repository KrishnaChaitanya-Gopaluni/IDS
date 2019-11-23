import xgboost
import multiprocessing as mp
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score




def model_1(data, test):
        
                                                                                
    classifier = xgboost.XGBClassifier(n_jobs=mp.cpu_count())

    classifier.fit(data.iloc[:,0:60], data[['classes']])

    y_pred = classifier.predict(test.iloc[:,0:60])

    #accuracy of the model
    accuracy = accuracy_score(test[['classes']], y_pred, normalize=True)
    print(accuracy) #0.9273891739650449

    '''
    ----
    as this is a binary classification, 
    precision calculation of this commented code and the uncomented code are the same
    -----
    pres_normal_wt = precision_score(test[['classes']], y_pred, labels=['attack', 'normal.'], pos_label = 'normal.',average ='weighted')
    pres_normal_micro = precision_score(test[['classes']], y_pred, labels=['attack', 'normal.'], pos_label = 'normal.',average ='micro')
    pres_normal_macro = precision_score(test[['classes']], y_pred, labels=['attack', 'normal.'], pos_label = 'normal.',average ='macro')
    pres_normal_binary = precision_score(test[['classes']], y_pred, labels=['attack', 'normal.'], pos_label = 'normal.',average ='binary')
    '''

    #0.7299779751676065
    pres_normal_wt = precision_score(test[['classes']], y_pred, labels=['normal.'], pos_label = 'normal.',average ='weighted')
    pres_normal_micro = precision_score(test[['classes']], y_pred, labels=[ 'normal.'], pos_label = 'normal.',average ='micro')
    pres_normal_macro = precision_score(test[['classes']], y_pred, labels=['normal.'], pos_label = 'normal.',average ='macro')
    pres_normal_binary = precision_score(test[['classes']], y_pred, labels=['normal.'], pos_label = 'normal.',average ='binary')


    #0.9988134539436238
    pres_attack_wt = precision_score(test[['classes']], y_pred, labels=['attack'], pos_label = 'attack',average ='weighted')
    pres_attack_micro = precision_score(test[['classes']], y_pred, labels=['attack'], pos_label = 'attack',average ='micro')
    pres_attack_macro = precision_score(test[['classes']], y_pred, labels=['attack'], pos_label = 'attack',average ='macro')
    pres_attack_binary = precision_score(test[['classes']], y_pred, labels=['attack'], pos_label = 'attack',average ='binary')



def model_2(data, test):
    





