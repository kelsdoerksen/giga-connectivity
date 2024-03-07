"""
Logistic Regression pipeline for ML call
"""

from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV
import wandb
from analysis.generating_results import cross_validate_scoring, results_for_plotting
from sklearn.metrics import f1_score


def run_lr(X_train,
           y_train,
           X_test,
           y_test,
           test_latitudes,
           test_longitudes,
           wandb_exp,
           results_dir):
    """
    Run logistic regression model
    """

    model_name = 'lr'

    # Create instance of Logistic Regression model
    print('Creating instance of LR model...')
    clf = LogisticRegression(max_iter=7000, random_state=48)

    # Fit to training data
    print('Fitting data...')
    clf.fit(X_train, y_train)

    # Measure model performance on test set
    probs = clf.predict_proba(X_test)

    accuracy = clf.score(X_test, y_test)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    # -----------------

    # Tune the model
    param_grid = {
        'penalty':  ['l2', None],
        'C': [0.01, 0.1, 1.0]
    }
    # grid search cv
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)

    # Fit the grid search to the data
    print('Running grid search cv...')
    grid_search.fit(X_train, y_train)
    # grid_search.best_params_
    best_clf = grid_search.best_estimator_

    tuned_probs = best_clf.predict_proba(X_test)

    # CV scoring
    cv_scoring = cross_validate_scoring(best_clf, X_train, y_train, ['accuracy', 'f1'], cv=5, results_dir=results_dir,
                                        prefix_name='tuned')

    # Saving results for further plotting
    results_for_plotting(y_test, tuned_probs, test_latitudes, test_longitudes, results_dir, model_name)

    # Save model using pickle
    with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
        pickle.dump(best_clf, f)

    predictions = (tuned_probs[:, 1] >= 0.5)
    predictions = predictions * 1
    f1 = f1_score(y_test, predictions)

    # Logging results to wandb
    # --- Logging metrics
    wandb_exp.log({
        'CV accuracies': cv_scoring['test_accuracy'],
        'Average CV accuracy': cv_scoring['test_accuracy'].mean(),
        'Average CV F1': cv_scoring['test_f1'].mean(),
        'Test set F1': f1,
        'Best Model Params': grid_search.best_params_
    })

    # --- Logging plots
    wandb_exp.log({
        'roc': wandb.plot.roc_curve(y_test, tuned_probs)
    })