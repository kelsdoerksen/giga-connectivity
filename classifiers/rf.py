"""
Random Forest ML call for pipeline
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score,
                             fbeta_score, recall_score, precision_score)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import wandb
import pickle
from analysis.generating_results import results_for_plotting
from analysis.confusion_matrix import calc_confusion_matrix
import random


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


def run_rf(X_train,
           y_train,
           X_test,
           y_test,
           X_val,
           y_val,
           test_latitudes,
           test_longitudes,
           wandb_exp,
           results_dir,
           tuning):
    """
    Experiment for train and test set
    """

    model_name = 'rf'
    seed = random.randint(0, 1000)

    # Create an instance of Random Forest
    print('Creating instance of Random Forest model...')
    forest = RandomForestClassifier(criterion='gini',
                                    random_state=seed,
                                    n_estimators=200,
                                    n_jobs=-1)

    # Fit the model
    print('Fitting model...')
    forest.fit(X_train, y_train)

    # Measure model performance on test set
    print('Evaluating model...')
    probs = forest.predict_proba(X_test)

    model_score = forest.score(X_test, y_test)
    print('Model accuracy on test set before any fine-tuning on validation is: {}'.format(model_score))

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
                                   scoring=['accuracy', 'f1'],
                                   refit='f1',
                                   cv=5,
                                   n_jobs=-1,
                             )

        # Fit the grid search to the data
        print('Running grid search cv on validation set...')
        rf_cv.fit(X_val, y_val)

        # Set model to best estimator from grid search
        best_forest = rf_cv.best_estimator_

        # Fit best estimator to our training set
        best_forest.fit(X_train, y_train)

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
