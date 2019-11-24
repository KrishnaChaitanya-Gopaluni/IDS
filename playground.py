from prepare_functions import *
from model_functions import *
from models import *
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='pass sampling method and number of experiments')
parser.add_argument("--exp", required=True, default=1, type=int, help="Enter number of experiments to repeat with shuffled data")
parser.add_argument("--mth", required=True, type=str, help="sampling name:entropy/margin/random")

args = parser.parse_args()

data = prepare_data('train')
test = prepare_data('test')

for i in tqdm(range(args.exp), desc= "All experiments Progress"):


    #shuffle the records for the random experiments
    data = data.sample(frac = 1).reset_index(drop=True)

    model_2(data, test,i, method = args.mth, initialCheck_point = 49014, samples_peround  = 48984) #Active learning


