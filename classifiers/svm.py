"""
SVM pipeline for ML call
"""

from sklearn.svm import SVC
import pickle
from sklearn.model_selection import GridSearchCV
import wandb
from analysis.generating_results import cross_validate_scoring, results_for_plotting
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def run_svm(X_train,
           y_train,
           X_test,
           y_test,
           test_latitudes,
           test_longitudes,
           wandb_exp,
           results_dir,
            tuning):
    """
    Run svm model
    """
    model_name = 'svm'

    # Create instance of SVM model
    print('Creating instance of SVM model...')
    clf = SVC(random_state=48, probability=True)

    # Fit to training data
    print('Fitting data...')
    clf.fit(X_train, y_train)

    # Measure model performance on test set
    probs = clf.predict_proba(X_test)

    accuracy = clf.score(X_test, y_test)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    if not eval(tuning):
        print('No model hyperparameter tuning')
        model_setup = 'non-tuned'
        cv_scoring = cross_validate_scoring(clf, X_train, y_train, ['accuracy', 'f1'], cv=5, results_dir=results_dir,
                                                   prefix_name=model_setup)
        # Save model using pickle
        with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
            pickle.dump(clf, f)
        predictions = (probs[:, 1] >= 0.5)
        predictions = predictions * 1
        f1 = f1_score(y_test, predictions)
        calc_confusion_matrix(y_test, probs[:, 1], results_dir)

        # Saving results for further plotting
        results_for_plotting(y_test, probs, test_latitudes, test_longitudes, results_dir, model_name)

        wandb_exp.log({
            'roc': wandb.plot.roc_curve(y_test, probs)
        })

    else:
        model_setup = 'tuned'
        # Tune the model
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [1, 2, 3, 4],
            'gamma': ['scale', 'auto']
        }
        # grid search cv
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)

        # Fit the grid search to the data
        print('Running grid search cv...')
        grid_search.fit(X_train, y_train)
        # grid_search.best_params_
        best_clf = grid_search.best_estimator_

        tuned_probs = best_clf.predict_proba(X_test)
        calc_confusion_matrix(y_test, tuned_probs[:, 1], results_dir)

        # CV scoring
        cv_scoring = cross_validate_scoring(best_clf, X_train, y_train, ['accuracy', 'f1'], cv=5,
                                            results_dir=results_dir, prefix_name=model_setup)

        # Saving results for further plotting
        results_for_plotting(y_test, tuned_probs, test_latitudes, test_longitudes, results_dir, model_name)

        # Save model using pickle
        with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
            pickle.dump(best_clf, f)

        predictions = (tuned_probs[:, 1] >= 0.5)
        predictions = predictions * 1
        f1 = f1_score(y_test, predictions)

        wandb_exp.log({
            'Best Model Params': grid_search.best_params_,
            'roc': wandb.plot.roc_curve(y_test, tuned_probs)
        })


    # Logging results to wandb
    # --- Logging metrics
    wandb_exp.log({
        'CV accuracies': cv_scoring['test_accuracy'],
        'Average CV accuracy': cv_scoring['test_accuracy'].mean(),
        'Average CV F1': cv_scoring['test_f1'].mean(),
        'Test set F1': f1,
        'Test set accuracy': accuracy_score(y_test, predictions)
    })