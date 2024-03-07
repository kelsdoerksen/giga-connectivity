"""
Script for generating results from classifier
"""

import pandas as pd
from sklearn.model_selection import cross_validate


def cross_validate_scoring(classifier, X_train, y_train, scoring, cv, results_dir, prefix_name):
    cv_scoring = cross_validate(classifier, X_train, y_train, scoring=scoring, cv=cv)
    with open('{}/{}_model_training_results.txt'.format(results_dir, prefix_name), 'w') as f:
        f.write('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
        f.write('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
        f.write('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
        f.write('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))

    return cv_scoring

def results_for_plotting(labels, preds, latitudes, longitudes, results_dir, classifier_type):
    """
    Generating df for further plotting later
    :return:
    """
    # Save the preds with labels and location to df for further plotting
    df_results = pd.DataFrame(columns=['label', 'prediction', 'lat', 'lon'])
    df_results['label'] = labels
    df_results['prediction'] = preds[:, 1] >= 0.5
    df_results['lat'] = latitudes
    df_results['lon'] = longitudes
    df_results.to_csv('{}/{}_results_for_plotting.csv'.format(results_dir, classifier_type))