import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import preprocessing

#read the dataset to viz
df = pd.read_csv('../kddcup_10_percent_corrected.csv')

#name the columns 
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

#name all the intrusions as "attack"
def classname_replace(x):
    if x != 'normal.':
        return 'attack'
    return x        
df["classes"] = df["classes"].apply(classname_replace)

#plot the graphs 
plt.style.use('classic')

g = sns.PairGrid(df[["duration",
#"protocol_type",
# "service",
"src_bytes",
"dst_bytes",
# "flag",	
# "land",	
"wrong_fragment",
"urgent", "classes"]], hue = "classes")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g.add_legend()
g.savefig("pair_plots_with_label.png")


#Scaling the features for better viz
mm_scaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_scaler.fit_transform(df[["duration",
"src_bytes",
"dst_bytes",
"wrong_fragment",
"urgent"]])# mm_scaler.transform(X_test)

basic_featues = pd.DataFrame(X_train_minmax, columns=["duration",
"src_bytes",
"dst_bytes",
"wrong_fragment",
"urgent"] )

basic_featues["classes"] = df["classes"]

g = sns.PairGrid(basic_featues, hue = "classes")
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g.add_legend()
g.savefig("pair_plots_with_label_basic_STDfeatures.png")
