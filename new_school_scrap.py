"""
Scrap processing scripts for new_schools results
using for MLRS paper
"""

import pandas as pd

root_dir = '/Users/kelseydoerksen/Desktop/Giga/isa_new_schools/BWA'
models = ['gb', 'rf', 'svm', 'lr', 'mlp']

def add_pred(data):
    final_pred = []

    for i in range(len(data)):
        if data.loc[i]['connected'] > data.loc[i]['unconnected']:
            final_pred.append(1)
        else:
            final_pred.append(0)

    return final_pred


for m in models:
    print('Running for model: {}'.format(m))
    df = pd.read_csv('{}/BWA_results_new_schools_corrected_{}_model.csv'.format(root_dir, m))
    preds = add_pred(df)
    df['final_pred'] = preds
    print('Number of connected schools from {} model is: {}'.format(m, sum(df['final_pred'])))
    print('Number of non-connected schools from {} model is: {}'.format(m, len(df)-sum(df['final_pred'])))
    df.to_csv('{}/BWA_results_new_schools_{}_model.csv'.format(root_dir, m))