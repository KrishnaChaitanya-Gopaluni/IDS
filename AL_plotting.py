#%%
import matplotlib.pyplot as plt
from pickle import load

#%%
def load_values(method):

    dump1 = load(open("./"+method+"/dump/acc_pres_0.data", 'rb'))
    # dump2 = load(open("/"+method+"/dump/acc_press_0.data", 'rb'))
    # dump3 = load(open("/"+method+"/dump/acc_press_0.data", 'rb'))

    dump2 = load(open("./"+method+"/acc_pres_0.data", 'rb'))
    dump3 = load(open("./"+method+"/acc_pres_1.data", 'rb'))
    dump4 = load(open("./"+method+"/acc_pres_2.data", 'rb'))
    dump5 = load(open("./"+method+"/acc_pres_3.data", 'rb'))
    

    return (dump1,dump2,dump3,dump4,dump5) 
# acc,acc_labeled, acc_unlabeled,pres_normal,pres_attack    


#%% plotting for the different scores
def plots(method,score_type,score_name):
    # names = ['acc','acc_labeled', 'acc_unlabeled','pres_normal','pres_attack' ]
    r1, r2, r3, r4, r5 = load_values(method)
    plt.figure(figsize=(20,10))
    plt.grid(fillstyle = 'full')
    if score_type <3:

        plt.plot(range(99),r1[score_type])
        plt.plot(range(100),r2[score_type])
        plt.plot(range(100),r3[score_type])
        plt.plot(range(100),r4[score_type])
        plt.plot(range(100),r5[score_type])
    else:
        plt.plot(range(99),[r1[score_type][i][0] for i in range(len(r1[score_type]))])
        plt.plot(range(100),[r2[score_type][i][0] for i in range(len(r2[score_type]))])
        plt.plot(range(100),[r3[score_type][i][0] for i in range(len(r3[score_type]))])
        plt.plot(range(100),[r4[score_type][i][0] for i in range(len(r4[score_type]))])
        plt.plot(range(100),[r5[score_type][i][0] for i in range(len(r5[score_type]))])


    plt.title(score_name+" for " +method+ " sampling")
    plt.ylabel(score_name)
    plt.xlabel("100 Iterations- 50k samples per itr")
    plt.legend([method+"_model1", method+"_model2", method+"_model3",method+"_model4", method+"_model5"])
    plt.savefig(score_name+"_"+method+'.png')
    

# %%
plots("random",0,'Accuracy')
plots("entropy",0,'Accuracy')
plots("margin",0,'Accuracy')
# %%
plots("random",1,'Training_Accuracy')
plots("entropy",1,'Training_accuracy')
plots("margin",1,'Training_accuracy')
#%%
plots("random",2,'Unlabeled_accuracy')
plots("entropy",2,'Unlabeled_accuracy')
plots("margin",2,'Unlabeled_accuracy')
# %%
plots("random",3,'precesion_Class-Normal')
plots("entropy",3,'precesion_Class-Normal')
plots("margin",3,'precesion_Class-Normal')
#%%
plots("random",4,'precesion_Class-Attack')
plots("entropy",4,'precesion_Class-Attack')
plots("margin",4,'precesion_Class-Attack')

#%%Plot all the three trainng accuracies in a single plot for 15 models
r1, r2, r3, r4, r5 = load_values('random')
e1, e2, e3, e4, e5 = load_values('entropy')
m1, m2, m3, m4, m5 = load_values('margin')
plt.figure(figsize=(20,10))
plt.grid(fillstyle = 'full')
score_type = 1 # unlabeled accuracy
plt.plot(range(99),r1[score_type])
plt.plot(range(100),r2[score_type])
plt.plot(range(100),r3[score_type])
plt.plot(range(100),r4[score_type])
plt.plot(range(100),r5[score_type])

plt.plot(range(99),e1[score_type])
plt.plot(range(100),e2[score_type])
plt.plot(range(100),e3[score_type])
plt.plot(range(100),e4[score_type])
plt.plot(range(100),e5[score_type])

plt.plot(range(99),m1[score_type])
plt.plot(range(100),m2[score_type])
plt.plot(range(100),m3[score_type])
plt.plot(range(100),m4[score_type])
plt.plot(range(100),m5[score_type])


score_name ='Training_Accuracy'
plt.title(score_name+" for all three sampling techniques")
plt.ylabel(score_name)
plt.xlabel("100 Iterations- 50k samples per itr")

method = 'random'
method2 = 'entropy'
method3='margin'
plt.legend([method+"_model1", method+"_model2", method+"_model3",method+"_model4", method+"_model5",
method2+"_model1", method2+"_model2", method2+"_model3",method2+"_model4", method2+"_model5",
method3+"_model1", method3+"_model2", method3+"_model3",method3+"_model4", method3+"_model5"])
plt.savefig(score_name+'_for all.png')


