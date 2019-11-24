import pandas as pd 
import numpy as np
from sklearn.feature_extraction import FeatureHasher


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
        return 1
    return 0        

'''
Use feature hasing rather than one hot encoding for the features  ["protocol_type","service","flag"] . Because, these Categorical feature have many
unique values. It's feasible to use one hot here.
'''
def encodeFeatures(data,cat_vars):
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

#preparing train/test data
def prepare_data(fileNmae):
    data=  read_data(fileNmae)
    columns = data.columns.tolist()
    [columns.remove(i) for i in ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]]# get continuous column names
    cat_vars = ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]
    data["classes"] = data["classes"].apply(className_replace)
    data =  pd.concat([encodeFeatures(data,cat_vars), data[columns]], axis = 1) # encode Categorical and concat with continuos 
    return data
'''
redundant code
#preparing test data
def prepare_test():

    test = read_data('test')
    columns = test.columns.tolist()
    [columns.remove(i) for i in ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]]# get continuous column names
    cat_vars = ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]
    test["classes"] = test["classes"].apply(className_replace)
    test =  pd.concat([encodeFeatures(test, cat_vars), test[columns]], axis = 1) # encode Categorical and concat with continuos 
'''





