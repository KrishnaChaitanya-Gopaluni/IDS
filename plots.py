import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

df = pd.read_csv('../kddcup_10_percent_corrected.csv')


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

def classname_replace(x):
    if x != 'normal.':
        return 'attack'
    return x        

df["classes"] = df["classes"].apply(classname_replace)
# g = sns.PairGrid(df)
# g = g.map_diag(plt.hist)
# g = g.map_offdiag(plt.scatter)
# g.savefig("pair_plots.png")


plt.style.use('classic')
# %matplotlib inline
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