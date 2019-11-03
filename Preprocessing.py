import os
import pandas as pd
import numpy as np
import glob
import multiprocessing as mp
from joblib import Parallel, delayed
import pickle 
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#read data
files = glob.glob('*.csv')
df = pd.read_csv(files[0])
                                                                            # df = Parallel(n_jobs=mp.cpu_count())(delayed(pd.read_csv)(i, low_memory=False) for i in files)
                                                                            # df = pd.concat(df)
'''
#catogorical columns
classes = np.unique(list(map(lambda x:x[41],df.values.tolist())))
protocol_type =np.unique(list(map(lambda x:x[1],df.values.tolist())))
service =np.unique(list(map(lambda x:x[2],df.values.tolist())))
flag =np.unique(list(map(lambda x:x[3],df.values.tolist())))
land = np.unique(list(map(lambda x:x[6],df.values.tolist())))
logged_in = np.unique(list(map(lambda x:x[11],df.values.tolist())))
is_host_login = np.unique(list(map(lambda x:x[20],df.values.tolist())))
is_guest_login = np.unique(list(map(lambda x:x[21],df.values.tolist())))
'''
#add column names
df.columns = ["duration",
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

#label encoding
cat_cols = ["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"]
labelencoder = LabelEncoder()
categorical_var = [df["protocol_type"],df["service"],df["flag"],df["land"],df["logged_in"], df["is_host_login"],df["is_guest_login"]]
cat_var_encoded = list(map(labelencoder.fit_transform, categorical_var))
cat_var_encoded = [i.tolist() for i in cat_var_encoded]
del categorical_var

#transposing
def transpoese_list(j):
    return [cat_var_encoded[i][j] for i in range(7)]
cat_var_encoded = list(map(transpoese_list,range(4898430)))

#create DF of cat vars
cat_var_encoded_df = pd.DataFrame(cat_var_encoded,columns=["protocol_type","service","flag","land","logged_in", "is_host_login","is_guest_login"])
del cat_var_encoded

#replace original cols with encoded cols
def replace_cols(i):
    df[i] = cat_var_encoded_df[i]
[ replace_cols(i) for i in cat_cols]
del cat_var_encoded_df # df.to_csv("kdd_encodes.csv")

#verify index of the col names 
df.iloc[:,1:4].columns
df.iloc[:,6:8].columns
df.iloc[:,11:13].columns
df.iloc[:,20:22].columns
    # or
df.iloc[:,[1,2,3, 6,11,20,21]]  


#one hot encoder
onehotencoder = OneHotEncoder(categorical_features = [1,2,3, 6,11,20,21])
df = onehotencoder.fit_transform(df).toarray()


