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

data2.to_csv("./kdd_encodes_2.csv")


import xgboost
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data2.iloc[:,0:126], data2.iloc[:,126], test_size = 0.2, random_state = 10)
classifier = xgboost.XGBClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred, normalize=True)
print(accuracy)

