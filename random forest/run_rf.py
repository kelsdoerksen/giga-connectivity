import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
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
  fig.tight_layout()
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


def get_class_balance(df, train, test, savedir):
    """
    Calculate class balance for total, train and test set to save
    :param: df: dataframe of input
    :param: train: labels from training set
    :param: test: labels from testing set
    :param: savedir: savedirectory
    :return:
    """

    with open('{}/classbalance.txt'.format(savedir), 'w') as f:
        f.write('Number of positive samples total is: {}'.format(sum(df['connectivity'])))
        f.write('Number of negative samples total is: {}'.format(len(df) - sum(df['connectivity'])))
        f.write('Number of positive training samples is: {}'.format(sum(train)))
        f.write('Number of positive training samples is: {}'.format(len(train) - sum(train)))
        f.write('Number of positive testing samples is: {}'.format(sum(test)))
        f.write('Number of positive testing samples is: {}'.format(len(test) - sum(test)))


def run_rf(aoi, buffer_extent, exp_type):
    """
    Experiment for train and test set from all years, all regions
    """
    results = '{}/{}/{}_features_results_{}m'.format(root_dir, aoi, exp_type, buffer_extent)
    if not os.path.isdir(results):
        os.makedirs(results)

    print('Loading data...')
    dataset = pd.read_csv('{}/{}/{}m_buffer/full_feature_space.csv'.format(root_dir, aoi, buffer_extent))

    dataset = dataset.drop(columns=['Unnamed: 0'])
    dataset = shuffle(dataset)

    # Drop rows if lat or lon are NaN
    dataset = dataset[dataset['lat'].notna()]
    dataset = dataset[dataset['lon'].notna()]

    X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['connectivity'], test_size = 0.2, random_state = 42)

    # Save lat, lon for X test and then drop label and location data
    test_latitudes = X_test['lat']
    test_longitudes = X_test['lon']

    # Dropping additional features to have a paired down feature space
    if exp_type == 'limited':
        feats_to_drop = ['connectivity', 'lat', 'lon', 'school_locations', 'modis.evg_conif', 'modis.evg_broad',
           'modis.dcd_needle', 'modis.dcd_broad', 'modis.mix_forest',
           'modis.cls_shrub', 'modis.open_shrub', 'modis.woody_savanna',
           'modis.savanna', 'modis.grassland', 'modis.perm_wetland',
           'modis.cropland', 'modis.urban', 'modis.crop_nat_veg',
           'modis.perm_snow', 'modis.barren', 'modis.water_bds', 'nightlight.avg_rad.var', 'nightlight.cf_cvg.var']

    if exp_type == 'full':
        feats_to_drop = ['connectivity', 'lat', 'lon', 'school_locations']

    X_test = X_test.drop(columns=feats_to_drop)
    X_train = X_train.drop(columns=feats_to_drop)

    print('Number of positive testing samples: {}'.format(sum(y_test)))
    print('Number of negative testing samples: {}'.format(len(y_test) - sum(y_test)))

    # Calculate the class balance accordingly and save
    get_class_balance(dataset, y_train, y_test, results)

    f1 = make_scorer(f1_score, average='macro')

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
    cv_accuracy = cross_val_score(forest, X_test, y_test, cv=5, scoring='accuracy')
    print('Scores for each fold are: {}'.format(cv_accuracy))
    print('Average accuracy: {}'.format(cv_accuracy.mean()))

    with open('{}/results.txt'.format(results), 'w') as f:
        f.write(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')
        f.write('Scores for each fold are: {}'.format(cv_accuracy))
        f.write('Average accuracy: {}'.format(cv_accuracy.mean()))

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