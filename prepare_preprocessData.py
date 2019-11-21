import pandas as pd 
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import xgboost
from sklearn.model_selection import train_test_split



'''
read data and name the columns
'''
def read_data(filename):
    data = pd.read_csv('../'+filename+'.csv')
    data.columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "classes"] 
    return data


'''
naming intrusions
'''
def className_replace(x):
    if x != 'normal.':
        return 'attack'
    return x        

'''
Use feature hasing rather than one hot encoding for the features  ["protocol_type","service","flag"] . Because, these Categorical feature have many
unique values. It's feasible to use one hot here.
'''
def encodeFeatures(data):
            #feature hashing         
    fh = FeatureHasher(n_features=6, input_type='string')
    ef1 = pd.concat([pd.DataFrame(fh.fit_transform(data[i]).toarray()) for i in cat_vars if i in ["protocol_type","service","flag"]], axis = 1)
    hash_features_names = ["protocol_type","service","flag"]
    hash_features_names = [hash_features_names[i]+'_'+str(j) for i in range(3) for j in range(6)]
    ef1.columns = hash_features_names

            #one hot encoding for other 4 Categorical features (before encoding each value is either 0 or 1 )
    ef2 = [pd.get_dummies(data[i], prefix=i) for i in cat_vars if i not in ["protocol_type","service","flag"]]
    ef2.append(ef1)
    return pd.concat(ef2,axis=1)

#preparing train data
data=  read_data('train')
columns = data.columns.tolist()
[columns.remove(i) for i in ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]]# get continuous column names
cat_vars = ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]
data["classes"] = data["classes"].apply(className_replace)
data =  pd.concat([encodeFeatures(data), data[columns]], axis = 1) # encode Categorical and concat with continuos 


#preparing test data
test = read_data('test')
test["classes"] = test["classes"].apply(className_replace)
test =  pd.concat([encodeFeatures(test), test[columns]], axis = 1) # encode Categorical and concat with continuos 




#label encoding

# y_pred = classifier.predict(test2)


#--------mock model--------------------#
import xgboost
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data2.iloc[:,0:126], data2.iloc[:,126], test_size = 0.2, random_state = 10)
classifier = xgboost.XGBClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred, normalize=True)
print(accuracy) #0.9998019773682588

# train the remaining data
classifier.fit(X_test, y_test)
del X_train
del X_test
del y_train
del y_test
del y_pred




