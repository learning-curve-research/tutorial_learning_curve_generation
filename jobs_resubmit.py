import pandas as pd 
from tqdm import tqdm
import numpy as np 

file_index = 1
target_split = 10

# Load jobs files
dfs = []
num = 0

for i in range(0, 10):
    try:
        fn = f'/mnt/jobs{file_index}/experiments_job{i}.csv'
        df_temp = pd.read_csv(fn)
        dfs.append(df_temp)
        num += len(df_temp)
    except:
        print('file ' + fn + ' is missing...')
        pass

df_jobs = pd.concat(dfs)
print("Total jobs:", len(df_jobs))


dfs = []
num = 0
empty_files = []  # To store filenames with only headers

for i in range(0, 10):
    try:
        fn = f'/mnt/results{file_index}/status_{i}.csv'
        df_temp = pd.read_csv(fn)
        # Check if the DataFrame only contains headers (no rows)
        if df_temp.empty:
            empty_files.append(fn)
        else:
            dfs.append(df_temp)
            num += len(df_temp)
    except:
        print('file ' + fn + ' is missing...')
        pass

# Display files that only contain headers (jobid and status)
if empty_files:
    print("Files with only headers and no data:")
    for empty_file in empty_files:
        print(empty_file)
else:
    print("No empty files with only headers found.")

df_status = pd.concat(dfs)

# Filter finished jobs
df_done = df_status[
    (df_status['status'] == 'ok') | 
    (df_status['status'] == 'timeout') | 
    (df_status['status'] == 'error')    # could be commanded out to avoid random error
]
print("finished jobs:", len(df_done))

done_jobids = set(df_done['jobid'])

# Use the set for checking membership in df_jobs and exclude jobid 
df_filtered_jobs = df_jobs[~df_jobs['jobid'].isin(done_jobids)]

# Split the filtered DataFrame and save
splits = np.array_split(df_filtered_jobs, target_split)

for index, value in enumerate(splits):
    print('working on job %d that consists of %d tasks...\n' % (index, len(value)))
    value.to_csv(f'/mnt/jobs/experiments_job{index}.csv', index=False)  # Disable index