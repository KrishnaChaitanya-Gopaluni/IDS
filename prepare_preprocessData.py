import pandas as pd 
import numpy as np


data = pd.read_csv("../kdd_encodes.csv")

columns= ["duration",
"protocol_type",
"service",
"flag",
"src_bytes",
"dst_bytes",
"land",
"wrong_fragment",
"urgent",
"hot",
"num_failed_logins",
"logged_in",
"num_compromised",
"root_shell",
"su_attempted",
"num_root",
"num_file_creations",
"num_shells",
"num_access_files",
"num_outbound_cmds",
"is_host_login",
"is_guest_login",
"count",
"srv_count",
"serror_rate",
"srv_serror_rate",
"rerror_rate",
"srv_rerror_rate",
"same_srv_rate",
"diff_srv_rate",
"srv_diff_host_rate",
"dst_host_count",
"dst_host_srv_count",
"dst_host_same_srv_rate",
"dst_host_diff_srv_rate",
"dst_host_same_src_port_rate",
"dst_host_srv_diff_host_rate",
"dst_host_serror_rate",
"dst_host_srv_serror_rate",
"dst_host_rerror_rate",
"dst_host_srv_rerror_rate",
"classes"] 

#now columns has only continuous variables
[columns.remove(i) for i in ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]]

cat_vars = ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]

#name all the intrusions as "attack"
def classname_replace(x):
    if x != 'normal.':
        return 'attack'
    return x        
data["classes"] = data["classes"].apply(classname_replace)


#one hot encoding
cat_conti_data = [pd.get_dummies(data[i], prefix=i) for i in cat_vars]
cat_conti_data.append(data[columns])
data2 = pd.concat(cat_conti_data,axis=1)

del cat_conti_data
del data

data2.to_csv("../kdd_encodes_2.csv")

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

#---------------------------------------------------------------#
#preparing test data

test = pd.read_csv("../corrected.csv")

#rerun the line 7 and run the below code
test.columns = columns

test["classes"] = test["classes"].apply(classname_replace)

from sklearn.preprocessing import LabelEncoder

#label encoding
cat_cols = ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]
labelencoder = LabelEncoder()
categorical_var = [test["protocol_type"],test["service"],test["flag"],test["land"],test["logged_in"], test["is_host_login"],test["is_guest_login"]]
cat_var_encoded = list(map(labelencoder.fit_transform, categorical_var))
cat_var_encoded = [i.tolist() for i in cat_var_encoded]
del categorical_var

#transposing
def transpoese_list(j):
    return [cat_var_encoded[i][j] for i in range(7)]
cat_var_encoded = list(map(transpoese_list,range(311028)))

#create DF of cat vars
cat_var_encoded_df = pd.DataFrame(cat_var_encoded,columns=["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"])
del cat_var_encoded

#replace original cols with encoded cols
def replace_cols(i):
    test[i] = cat_var_encoded_df[i]
[ replace_cols(i) for i in cat_cols]
del cat_var_encoded_df


#rerun line 51 before code below executes
#one hot encoding
cat_conti_data = [pd.get_dummies(test[i], prefix=i) for i in cat_vars]
cat_conti_data.append(test[columns])
test2 = pd.concat(cat_conti_data,axis=1)

[i for i in data2.columns.tolist() if i not in test2.columns.tolist()]
#['service_65', 'service_66', 'service_67', 'service_68', 'service_69'] 
''' these are the 5 missing columns in the testing dataset'''



del cat_conti_data
del test

test2.to_csv("../test_encodes.csv")

