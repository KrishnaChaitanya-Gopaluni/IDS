from prepare_functions import *
from model_functions import *
from models import *
import argparse
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser(description='pass sampling method and number of experiments')
parser.add_argument("--type", required=True, default=1, type=int, help="Enter number of experiments to repeat with shuffled data")
parser.add_argument("--method", required=False, type=str, help="sampling name:entropy/margin/random")

#parser.add_argument("--mth", required=True, type=str, help="sampling name:entropy/margin/random")


args = parser.parse_args()

data = prepare_data('train')
test = prepare_data('test')

if args.type ==0:
    model_1(data, test)

if args.type ==1:

    model_2(data, test,1,method = args.method, initialCheck_point = 49014, samples_peround  = 48984) #this model supports three active learning techniques

if args.type ==2:
    #0th  experiment(3rd parameter)
    model_3(data, test,0, initialCheck_point = 49014, samples_peround  = 48984) #UCB Active learning Experiment





'''
#uncommet to run active learning,  multile experiments with shuffled data
rands = np.random.randint(5,100, size = args.exp)
for i in range(args.exp):


    #shuffle the records for the random experiments
    if i != 0:
        data = data.sample(frac = 1, random_state = rands[i]).reset_index(drop=True)

    model_2(data, test,i, method = args.mth, initialCheck_point = 49014, samples_peround  = 48984) #Active learning

'''
