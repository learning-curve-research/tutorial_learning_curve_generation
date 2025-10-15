import pandas as pd
import numpy as np
import os
import itertools
from tqdm import tqdm
# local
import sys
sys.path.append('lcdb_function')
from lcdb_function.lcdb import get_dataset, get_inner_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

anchor_list = np.ceil(16 * 2 ** ((np.arange(137)) / 8)).astype(int)

dataset_ids = [  11,     13 ]

seed_list = range(5)

learner_zoo = [ 'sklearn.tree.DecisionTreeClassifier',
                'sklearn.tree.ExtraTreeClassifier',
              ]


def get_real_anchor_list(openml_id):
    X, y = get_dataset(openml_id)
    X_train, _, _, _, _, _ = get_inner_split(X, y, outer_seed=0, inner_seed=0)
    real_anchor_list = anchor_list[anchor_list <= X_train.shape[0]] 
    return real_anchor_list


timelimit = 1 # hour
num_splits = 10
counter = 0


data = []
first_dataset = True

one_seed = False
if one_seed:
    print('only doing 1 seed for testing purpose!')
    seed_list = [0]
else:
    seed_list = range(5)

for dataset_id in tqdm(dataset_ids):
    data = []
    
    real_anchor_list = get_real_anchor_list(dataset_id)

    print(f"Learner Zoo {len(learner_zoo)}, Real Anchor List {len(real_anchor_list)}")

    combinations = len(learner_zoo) * len(real_anchor_list) * len(seed_list) * len(seed_list) * 2 
    print(f"we have here {combinations} combinations.")

    param_combinations = itertools.product(
        [dataset_id],
        learner_zoo,
        real_anchor_list,
        seed_list,
        seed_list,
        [timelimit]
    )

    for combination in param_combinations:
        row = list(combination)
        row.insert(0, counter)
        counter = counter+1
        data.append(row)

    df = pd.DataFrame(data, columns=['jobid', 'openmlid', 'learner', 'size_train', 'outer_seed', 'inner_seed', 'timelimit'])
    df.to_csv('jobs_dataset%d.csv' % dataset_id)
    df = df.sample(frac=1).reset_index(drop=True)
    
    splits = np.array_split(df, num_splits)

    for index, value in enumerate(splits):
        # print('working on job %d that consists of %d tasks...\n' % (index, len(value)))
        if first_dataset:
            value.to_csv('jobs/experiments_job%d.csv' % index, index=False)
        else: 
            value.to_csv('jobs/experiments_job%d.csv' % index, mode='a', index=False, header=False)

    first_dataset = False
    print("-----------------------------------")

