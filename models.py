import xgboost
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from pickle import dump,load
from model_functions import *
import os



def model_1(data, test):#xgboost
                                                                            
    classifier = xgboost.XGBClassifier(n_jobs=mp.cpu_count())
    classifier.fit(data.iloc[:,0:60], data[['classes']])
    y_pred = classifier.predict(test.iloc[:,0:60])
    #accuracy of the model #0.9273891739650449
    print(accuracy(test,y_pred)) 
    #precision of the model #0.7299779751676065 for 'normal.'  #0.9988134539436238 for 'attack' # posLabel is the positive class
    # returns weighted, micro, macro, binary precision values
    print(pres(test, y_pred,posLabel = 0))
    print(pres(test, y_pred,posLabel = 1))  

    return classifier
  

def model_2(data, test,experiment,method = 'entropy', initialCheck_point = 49014, samples_peround  = 48984): #this model supports three active learning
    acc = []# on whole test data
    acc_unlabeled = []
    acc_labeled = []
    pres_normal = []
    pres_attack = []
    i = 0
    
    
    classifier = xgboost.XGBClassifier(n_jobs=mp.cpu_count(), objective='binary:logistic')
    
    classifier.fit(data.iloc[0:initialCheck_point,0:60], data[['classes']].iloc[0:initialCheck_point])
    
    acc.append(accuracy(test,classifier.predict(test.iloc[:,0:60])))
    acc_labeled.append(accuracy(data.iloc[0:initialCheck_point,0:61],classifier.predict(data.iloc[0:initialCheck_point,0:60])))
    acc_unlabeled.append(accuracy(data.iloc[initialCheck_point:data.shape[0],0:61],classifier.predict(data.iloc[initialCheck_point:data.shape[0],0:60])))

    pres_normal.append(pres(test,classifier.predict(test.iloc[:,0:60]),posLabel = 'normal.'))
    pres_attack.append(pres(test,classifier.predict(test.iloc[:,0:60]),posLabel = 'attack'))
    os.system('clear')

    #prepare the data for active learning
    labeled_data = data.iloc[0:initialCheck_point,0:61].reset_index(drop = True)
    unlabeled_data = data.iloc[initialCheck_point:data.shape[0],0:61].reset_index(drop = True)

    del data
    # print("Total:"+str(acc[i])+" Labeled:"+str(acc_labeled[i])+" Unlabeled Accuracies:"+str(acc_unlabeled[i]))
    
    for itr in tqdm(range(99), desc=method+" experiment "+str(experiment)+" training Progress:"):
        if unlabeled_data.shape[0] ==0 and len(acc)==100:
            break
        
        if method =='margin':
            idx = margin_sampling(unlabeled_data.iloc[:,0:60], classifier, sample_size = samples_peround)
        if method == 'entropy':
            idx = entropy_sampling(unlabeled_data.iloc[:,0:60], classifier, sample_size = samples_peround)
        if method == 'random':
            idx = random_sampling(unlabeled_data.iloc[:,0:60], classifier, sample_size = samples_peround)

        #update the labels
        labeled_data = pd.concat([labeled_data, unlabeled_data.loc[idx]], axis =0).reset_index(drop = True)
        unlabeled_data = unlabeled_data.drop(idx).reset_index(drop = True)
       
        del classifier # trash the old model to build the new one
       
        classifier = xgboost.XGBClassifier(n_jobs=mp.cpu_count(), objective='binary:logistic')
        classifier.fit(labeled_data.iloc[:,0:60], labeled_data[['classes']])
        
        #claculate accuracy
        acc.append(accuracy(test,classifier.predict(test.iloc[:,0:60])))
        acc_labeled.append(accuracy(labeled_data,classifier.predict(labeled_data.iloc[:,0:60])))
        if unlabeled_data.shape[0]!=0:
            acc_unlabeled.append(accuracy(unlabeled_data,classifier.predict(unlabeled_data.iloc[:,0:60])))
        else:
            acc_unlabeled.append(acc_unlabeled[len(acc_unlabeled)-1])

        #claculate precision
        pres_normal.append(pres(test,classifier.predict(test.iloc[:,0:60]),posLabel = 'normal.'))
        pres_attack.append(pres(test,classifier.predict(test.iloc[:,0:60]),posLabel = 'attack'))
        os.system('clear')
        dump([acc,acc_labeled, acc_unlabeled,pres_normal,pres_attack],open('../'+method+'/acc_pres_'+str(experiment)+'.data', 'wb'))
        

        
        i = i + 1
    # print("Total:"+str(acc[i])+" Labeled:"+str(acc_labeled[i])+" Unlabeled Accuracies:"+str(acc_unlabeled[i]))    



def model_3(data, test,experiment,initialCheck_point = 49014, samples_peround  = 48984): #Run Upper Confidance bound on top of Active learning
    acc = []# on whole test data
    acc_unlabeled = []
    acc_labeled = []
    pres_normal = []
    pres_attack = []
    i = 0
    
    #UCB 
    method = ''
    st = ['margin','entropy','random']
    d = 3
    model_selected = []
    numbers_of_selections = [0] * d
    running_means = [0] * d
    alpha = 0.1#A #0.1
    c = 0.1#100 const
    running_mean_values_perIter = []
    ucb_values = []
    deltas = []

    classifier = xgboost.XGBClassifier(n_jobs=mp.cpu_count(), objective='binary:logistic')
    
    classifier.fit(data.iloc[0:initialCheck_point,0:60], data[['classes']].iloc[0:initialCheck_point])
    
    acc.append(accuracy(test,classifier.predict(test.iloc[:,0:60])))
    acc_labeled.append(accuracy(data.iloc[0:initialCheck_point,0:61],classifier.predict(data.iloc[0:initialCheck_point,0:60])))
    acc_unlabeled.append(accuracy(data.iloc[initialCheck_point:data.shape[0],0:61],classifier.predict(data.iloc[initialCheck_point:data.shape[0],0:60])))

    pres_normal.append(pres(test,classifier.predict(test.iloc[:,0:60]),posLabel = 'normal.'))
    pres_attack.append(pres(test,classifier.predict(test.iloc[:,0:60]),posLabel = 'attack'))
    os.system('clear')
    
    #for UCB
    acc3 = accuracy(data.iloc[initialCheck_point:data.shape[0],0:61],classifier.predict(data.iloc[initialCheck_point:data.shape[0],0:60]))
    last_acc =  acc3
    delta = acc3 - last_acc
    deltas.append(delta)

    #prepare the data for active learning
    labeled_data = data.iloc[0:initialCheck_point,0:61].reset_index(drop = True)
    unlabeled_data = data.iloc[initialCheck_point:data.shape[0],0:61].reset_index(drop = True)

    del data
    # print("Total:"+str(acc[i])+" Labeled:"+str(acc_labeled[i])+" Unlabeled Accuracies:"+str(acc_unlabeled[i]))
    
    for itr in tqdm(range(99), desc=method+" experiment "+str(experiment)+" training Progress:"):
        
        if itr <= 2:

            strgy = st[i]
            
            model_selected.append(i)
            numbers_of_selections[i] += 1
            print(" (without ucb) selected Strategy" +strgy )
        else:

            ucbs = get_ucb(running_means, numbers_of_selections, c, i+1)
            ucb_values.append(ucbs)
            arm = np.argmax(np.array(ucbs))
            strgy = st[arm]
            model_selected.append(arm)
            numbers_of_selections[arm] += 1
            print(" UCB selected Strategy" +strgy )
        
        method = strgy
        if unlabeled_data.shape[0] ==0 and len(acc)==100:
            break
        
        if method =='margin':
            idx = margin_sampling(unlabeled_data.iloc[:,0:60], classifier, sample_size = samples_peround)
        if method == 'entropy':
            idx = entropy_sampling(unlabeled_data.iloc[:,0:60], classifier, sample_size = samples_peround)
        if method == 'random':
            idx = random_sampling(unlabeled_data.iloc[:,0:60], classifier, sample_size = samples_peround)

        #update the labels
        labeled_data = pd.concat([labeled_data, unlabeled_data.loc[idx]], axis =0).reset_index(drop = True)
        unlabeled_data = unlabeled_data.drop(idx).reset_index(drop = True)
       
        del classifier # trash the old model to build the new one
       
        classifier = xgboost.XGBClassifier(n_jobs=mp.cpu_count(), objective='binary:logistic')
        classifier.fit(labeled_data.iloc[:,0:60], labeled_data[['classes']])
        
        #claculate accuracy
        acc.append(accuracy(test,classifier.predict(test.iloc[:,0:60])))
        acc_labeled.append(accuracy(labeled_data,classifier.predict(labeled_data.iloc[:,0:60])))
        if unlabeled_data.shape[0]!=0:
            acc3 = accuracy(unlabeled_data,classifier.predict(unlabeled_data.iloc[:,0:60]))
            acc_unlabeled.append(acc3)
        else:
            acc3 = last_acc
            acc_unlabeled.append(acc_unlabeled[len(acc_unlabeled)-1])


        #claculate precision
        pres_normal.append(pres(test,classifier.predict(test.iloc[:,0:60]),posLabel = 'normal.'))
        pres_attack.append(pres(test,classifier.predict(test.iloc[:,0:60]),posLabel = 'attack'))
        os.system('clear')
        dump([acc,acc_labeled, acc_unlabeled,pres_normal,pres_attack],open('../ucb/acc_pres.data', 'wb'))
        dump([model_selected,numbers_of_selections,running_means,deltas,ucb_values],open('../ucb/ucb.data', 'wb'))
        

        delta = acc3 - last_acc
        last_acc =  acc3
        deltas.append(delta)
        

        running_means[st.index(strgy)] = alpha*running_means[st.index(strgy)] + (1-alpha)*delta
        running_mean_values_perIter.append(running_means)
        
        i = i + 1





    
    






    





