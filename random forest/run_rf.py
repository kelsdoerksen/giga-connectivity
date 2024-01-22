import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import os
from sklearn.metrics import f1_score, make_scorer
import argparse

parser = argparse.ArgumentParser(description='Random Forest',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--aoi", help="Country of interest")
parser.add_argument("--buffer", help="Buffer extent around school to extract data from")
parser.add_argument("--experiment", help="Feature Experiment type. Must be one of full or limited at the moment")
args = parser.parse_args()

root_dir = '/Users/kelseydoerksen/Desktop/Giga'

def calc_importance(model, X, save_dir):
  importances = model.feature_importances_
  feature_names = X.columns
  df = pd.DataFrame(np.array([importances]), columns=feature_names, index=['importance'])
  df = df.sort_values(by='importance', axis=1, ascending=False)
  df.to_csv('{}/FI.csv'.format(save_dir))
  importances = pd.Series(importances, index=feature_names)
  fig, ax = plt.subplots()
  importances.plot.bar()
  ax.set_title("Feature importances using Gini Index")
  ax.set_ylabel("Feature importance score")
  plt.savefig('{}/FeatureImportances.png'.format(save_dir), bbox_inches='tight')

  return df


def calc_confusion_matrix(y_test, y_pred, savedir):
    """
    Calculates confusion matrix
    """
    predictions = (y_pred >= 0.5)

    CM = confusion_matrix(y_test, predictions)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    print('True Positive is {}'.format(TP))
    print('True Negative is {}'.format(TN))
    print('False Positive is {}'.format(FP))
    print('False Negative is {}'.format(FN))

    FP_Rate = FP / (FP + TN)
    TP_Rate = TP / (TP + FN)
    FN_Rate = FN / (FN + TP)
    TN_Rate = TN / (TN + FP)

    print('False positive rate is {}'.format(FP_Rate))
    print('True positive rate is {}'.format(TP_Rate))
    print('False negative rate is {}'.format(FN_Rate))
    print('True negative rate is {}'.format(TN_Rate))

    with open('{}/confusionmatrix.txt'.format(savedir), 'w') as f:
        f.write('False positive rate is {}'.format(FP_Rate))
        f.write('True positive rate is {}'.format(TP_Rate))
        f.write('False negative rate is {}'.format(FN_Rate))
        f.write('True negative rate is {}'.format(TN_Rate))


def get_class_balance(train_df, test_df, savedir):
    """
    Calculate class balance for total, train and test set to save
    :param: train_df: training data
    :param: test_df: testing data
    :param: savedir: savedirectory
    :return:
    """
    positive_samples_total = sum(train_df['connectivity']) + sum(test_df['connectivity'])
    negative_sampels_total = len(train_df) + len(test_df) - positive_samples_total

    with open('{}/classbalance.txt'.format(savedir), 'w') as f:
        f.write('Number of positive samples total is: {}'.format(positive_samples_total))
        f.write('Number of negative samples total is: {}'.format(negative_sampels_total))
        f.write('Number of positive training samples is: {}'.format(sum(train_df['connectivity'])))
        f.write('Number of positive training samples is: {}'.format(len(train_df) - sum(train_df['connectivity'])))
        f.write('Number of positive testing samples is: {}'.format(sum(test_df['connectivity'])))
        f.write('Number of positive testing samples is: {}'.format(len(test_df) - sum(test_df['connectivity'])))


def load_data(aoi, buffer_extent):
    """
    Loading data
    :param aoi:
    :param buffer_extent:
    :return:
    """
    train_df = pd.read_csv('{}/{}/{}m_buffer/TrainingData.csv'.format(root_dir, aoi, buffer_extent))
    test_df = pd.read_csv('{}/{}/{}m_buffer/TestingData.csv'.format(root_dir, aoi, buffer_extent))

    training_data = train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'connectivity.1'])
    testing_data = test_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'connectivity.1'])

    training_data = shuffle(training_data)
    testing_data = shuffle(testing_data)

    # Drop rows if lat or lon are NaN
    training_dataset = training_data[training_data['lat'].notna()]
    training_dataset = training_data[training_data['lon'].notna()]
    testing_dataset = testing_data[testing_data['lat'].notna()]
    testing_dataset = testing_data[testing_data['lon'].notna()]

    return training_dataset, testing_dataset


def run_rf(aoi, buffer_extent, exp_type):
    """
    Experiment for train and test set from all years, all regions
    """
    results = '{}/{}/{}_features_results_{}m'.format(root_dir, aoi, exp_type, buffer_extent)
    if not os.path.isdir(results):
        os.makedirs(results)

    print('Loading data...')
    train_df, test_df = load_data(aoi, buffer_extent)

    # Save lat, lon for X test and then drop label and location data
    test_latitudes = test_df['lat']
    test_longitudes = test_df['lon']

    # Calculate the class balance accordingly and save
    get_class_balance(train_df, test_df, results)

    # currently dropping education_level because there are nans in a lot of samples
    if exp_type == 'full':
        cols_to_drop = ['connectivity', 'lat', 'lon', 'school_locations', 'giga_id_school']

    # Drop education level first
    train_df = train_df.drop(columns=['education_level'])
    test_df = test_df.drop(columns=['education_level'])

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    y_train = train_df['connectivity']
    y_test = test_df['connectivity']

    X_test = test_df.drop(columns=cols_to_drop)
    X_train = train_df.drop(columns=cols_to_drop)

    print('Number of positive testing samples: {}'.format(sum(y_test)))
    print('Number of negative testing samples: {}'.format(len(y_test) - sum(y_test)))

    # Create an instance of Random Forest
    forest = RandomForestClassifier(criterion='gini',
                                    random_state=87,
                                    n_estimators=100,
                                    n_jobs=-1)

    # Fit the model
    print('Fitting model...')
    forest.fit(X_train, y_train)

    # Measure model performance on test set
    print('Evaluating model...')
    probs = forest.predict_proba(X_test)

    accuracy = forest.score(X_test, y_test)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    # Save the preds with labels and location to df for further plotting
    df_results = pd.DataFrame(columns=['label', 'prediction', 'lat', 'lon'])
    df_results['label'] = y_test
    df_results['prediction'] = probs[:,1] >= 0.5
    df_results['lat'] = test_latitudes
    df_results['lon'] = test_longitudes
    df_results.to_csv('{}/results_for_plotting.csv'.format(results))

    # Cross validation score
    cv_scoring = cross_validate(forest, X_test, y_test, scoring=['accuracy', 'f1'], cv=5)
    print('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
    print('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
    print('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
    print('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))

    with open('{}/results.txt'.format(results), 'w') as f:
        f.write(f'The hard predictions were right on the test set {100 * accuracy:5.2f}% of the time')
        f.write('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
        f.write('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
        f.write('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
        f.write('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    rfc_auc = roc_auc_score(y_test, probs[:,1])
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('RFC: ROC AUC=%.3f' % rfc_auc)
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    rfc_fpr, rfc_tpr, _ = roc_curve(y_test, probs[:,1])
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(rfc_fpr, rfc_tpr, marker='.', label='RFC')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('{}/roc_auc_curve.png'.format(results))

    calc_importance(forest, X_test, results)
    calc_confusion_matrix(y_test, probs[:,1], results)

# Run it
run_rf(args.aoi, args.buffer, args.experiment)