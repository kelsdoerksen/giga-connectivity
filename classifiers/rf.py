"""
Random Forest ML call for pipeline
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import wandb
import pickle


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



def run_rf(X_train,
           y_train,
           X_test,
           y_test,
           test_latitudes,
           test_longitudes,
           wandb_exp,
           results_dir):
    """
    Experiment for train and test set
    """

    # Create an instance of Random Forest
    print('Creating instance of Random Forest model...')
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

    # Save model using pickle
    with open('{}/rfc_model.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(forest, f)

    # Save the preds with labels and location to df for further plotting
    df_results = pd.DataFrame(columns=['label', 'prediction', 'lat', 'lon'])
    df_results['label'] = y_test
    df_results['prediction'] = probs[:,1] >= 0.5
    df_results['lat'] = test_latitudes
    df_results['lon'] = test_longitudes
    df_results.to_csv('{}/rf_results_for_plotting.csv'.format(results_dir))

    # Cross validation score
    cv_scoring = cross_validate(forest, X_test, y_test, scoring=['accuracy', 'f1'], cv=5)
    print('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
    print('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
    print('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
    print('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))

    with open('{}/rf_results.txt'.format(results_dir), 'w') as f:
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
    plt.savefig('{}/roc_auc_curve.png'.format(results_dir))

    calc_importance(forest, X_test, results_dir)
    calc_confusion_matrix(y_test, probs[:,1], results_dir)

    # Logging results to wandb
    # --- Logging metrics
    wandb_exp.log({
        'Test set CV accuracies': cv_scoring['test_accuracy'],
        'Average test set accuracy': cv_scoring['test_accuracy'].mean(),
        'Average test set F1': cv_scoring['test_f1'].mean()
    })

    # --- Logging plots
    wandb_exp.log({
        'roc': wandb.plot.roc_curve(y_test, probs)
    })

