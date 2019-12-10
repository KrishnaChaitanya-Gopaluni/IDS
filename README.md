# Building Intrusion Detection System (IDS) using Machine Learning


Need for Machine learning in IDS: 

Traditionally intrusion detection systems (IDS) are signature/rule-based which means they excel at detecting known attacks. Since the vulnerabilities and their exploits are evolving at a breakneck pace, it may not always be possible to have all known exploits/signatures as a part of the database. Hence there is a need for a more intelligent, proactive and flexible system that automatically learns the characteristics of malicious access from a historical database. The machine learning algorithms learn from the existing instances and extract valuable and discriminative features from the data set to help the learning algorithm differentiate the intrusions.

Machine Learning methods :  
Upper Confidance Bound, a classic solution to reinforcement learning problem called multi armed bandit is implemented.It can maximize the model performance in XGBoost active learning by exploring/exploiting 3 alternative choices(uncertainity sampling techniques). More details are in the report.


#Code Execution instructions

    0) Running baseline model

         python playground.py --type 0

    1) Running Active learnig techniques

        python playground.py --type 1 --method entropy

        or 
        python playground.py --type 1 --method margin

        or 
        python playground.py --type 1 --method random

    2) Running UCB technique
        
        python playground.py --type 2



place the data in parent folder with names train.csv and test.csv and code in sub folder. Code automatically cleans the data before the models starts.
