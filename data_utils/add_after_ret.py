'''
convert oth_ret to after_ret format for a single dataset and seed
'''

import pandas as pd
import numpy as np

results = 'aggregated_results/learned_opt_main_data_stats.csv'

df = pd.read_csv(results)
df['after_ret'] = ''
df['after_ret_mean'] = np.nan
print("Adding after_ret...")
print(df.shape)
for row_num in range(df.shape[0]):
    oth_ret = df.iloc[row_num]['oth_ret']
    if type(oth_ret) is str:
        for elem in oth_ret.split():
            data_id, stat = elem.split('-')
            data_id, stat = int(data_id), float(stat)
            at = df.at[row_num, 'after_ret']
            if at == '':
                df.at[row_num, 'after_ret'] = f"{stat}"
            else:
                df.at[row_num, 'after_ret'] += f" {stat}"
            
print("Adding after_ret_mean...")
for row_num in range(df.shape[0]):
    at = df.at[row_num, 'after_ret']
    if at != '':
        after_ret = [float(x) for x in at.split()]
        df.at[row_num, 'after_ret_mean'] = np.nanmean(after_ret)

print("Adding after_acc_mean...")      
for row_num in range(df.shape[0]):
    at = df.at[row_num, 'after_acc']
    if type(at) is str:
        after_acc = [float(x) for x in at.split()]
        df.at[row_num, 'after_acc_mean'] = np.nanmean(after_acc)

df.to_csv(results, index=False)



