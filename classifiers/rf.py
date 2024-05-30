"""
Random Forest ML call for pipeline
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             fbeta_score, classification_report, precision_recall_curve, auc,
                             roc_auc_score, recall_score, precision_score)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import wandb
import pickle
from analysis.generating_results import results_for_plotting
import seaborn as sn


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

    classes = ['0','1']
    df_cfm = pd.DataFrame(CM, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("{}/cfm.png".format(savedir))


def run_rf(X_train,
           y_train,
           X_test,
           y_test,
           test_latitudes,
           test_longitudes,
           wandb_exp,
           results_dir,
           tuning):
    """
    Experiment for train and test set
    """

    model_name = 'rf'

    # Create an instance of Random Forest
    print('Creating instance of Random Forest model...')
    forest = RandomForestClassifier(criterion='gini',
                                    random_state=87,
                                    n_estimators=200,
                                    n_jobs=-1)

    # Fit the model
    print('Fitting model...')
    forest.fit(X_train, y_train)

    # Measure model performance on test set
    print('Evaluating model...')
    probs = forest.predict_proba(X_test)

    model_score = forest.score(X_test, y_test)
    print('Model accuracy before any fine-tuning is: {}'.format(model_score))

    if not eval(tuning):
        print('No model hyperparameter tuning')

        with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
            pickle.dump(forest, f)

        predictions = (probs[:, 1] >= 0.5)
        predictions = predictions * 1
        f1 = f1_score(y_test, predictions)

        # Saving results for further plotting
        results_for_plotting(y_test, probs, test_latitudes, test_longitudes, results_dir, model_name)
        calc_importance(forest, X_test, results_dir)
        calc_confusion_matrix(y_test, probs[:, 1], results_dir)

        wandb_exp.log({
            'roc': wandb.plot.roc_curve(y_test, probs),
            'accuracy': accuracy_score(y_test, predictions),
            'f1': f1,
            'recall': recall_score(y_test, predictions, zero_division=0),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'fbeta': fbeta_score(y_test, predictions, beta=0.5, zero_division=0)
        })

    else:
        model_setup = 'tuned'
        # Tune the model
        param_grid = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100],
            'max_features': [2, 3, 4],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [4, 6, 8],
            'n_estimators': [100, 200, 300, 500]
        }
        # grid search cv
        rf_cv = GridSearchCV(estimator=forest,
                                   param_grid=param_grid,
                                   scoring={'accuracy', 'f1'},
                                   refit='f1',
                                   cv=5,
                                   n_jobs=-1,
                             )

        # Fit the grid search to the data
        print('Running grid search cv...')
        rf_cv.fit(X_train, y_train)

        # Set model to best estimator from grid search
        best_forest = rf_cv.best_estimator_

        # Prediction with best forest
        tuned_probs = best_forest.predict_proba(X_test)

        with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
            pickle.dump(best_forest, f)

        # Saving results for further plotting
        results_for_plotting(y_test, tuned_probs, test_latitudes, test_longitudes, results_dir, model_name)

        predictions = (tuned_probs[:, 1] >= 0.5)*1
        predictions = predictions.tolist()
        f1 = f1_score(y_test, predictions, zero_division=0)
        calc_importance(best_forest, X_test, results_dir)
        calc_confusion_matrix(y_test, tuned_probs[:, 1], results_dir)

        wandb_exp.log({
            'Best Model Params': rf_cv.best_params_,
            'roc': wandb.plot.roc_curve(y_test, tuned_probs),
            'accuracy': accuracy_score(y_test, predictions),
            'f1': f1,
            'recall': recall_score(y_test, predictions, zero_division=0),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'fbeta': fbeta_score(y_test, predictions, beta=0.5, zero_division=0)
        })
