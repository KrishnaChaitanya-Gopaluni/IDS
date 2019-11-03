# IDS
Intrusion Detection using Machine Learning


Need for Machine learning in IDS: 

Traditionally intrusion detection systems (IDS) are signature/rule-based which means they excel at detecting known attacks. Since the vulnerabilities and their exploits are evolving at a breakneck pace, it may not always be possible to have all known exploits/signatures as a part of the database. Hence there is a need for a more intelligent, proactive and flexible system that automatically learns the characteristics of malicious access from a historical database. The machine learning algorithms learn from the existing instances and extract valuable and discriminative features from the data set to help the learning algorithm differentiate the intrusions.

Machine Learning methods :  

We aim to implement and benchmark various machine learning techniques used in different research papers. As a best practice, to compare different machine learning models, we will be using the same standard dataset across all methods. We will be aiming to use KDD Cup 1999: Computer network intrusion detection dataset and explore more. In the recent past research papers for ML-based IDS, classical ML models were used as well as advanced techniques like deep neural networks. We specifically wanted to concentrate on the state of the art techniques like Extreme gradient boosting (XGBoost), deep neural networks from the papers specified. Also, we try to re-use and enhance the existing methods using active learning techniques, which is a good fit when there are limited labeled data points, a scenario not uncommon in the real world. 

