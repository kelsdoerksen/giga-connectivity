"""
MLP pipeline for ML call
"""

from sklearn.neural_network import MLPClassifier
import pickle
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import wandb
from analysis.generating_results import cross_validate_scoring, results_for_plotting

def run_mlp(X_train,
           y_train,
           X_test,
           y_test,
           test_latitudes,
           test_longitudes,
           wandb_exp,
           results_dir):
    """
    Run mlp model
    """

    model_name = 'mlp'

    # Create instance of MLP model
    print('Creating instance of MLP model...')
    clf = MLPClassifier(random_state=48, max_iter=15000)

    # Fit to training data
    print('Fitting data...')
    clf.fit(X_train, y_train)

    # Measure model performance on test set
    probs = clf.predict_proba(X_test)

    accuracy = clf.score(X_test, y_test)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    # Tune the model
    param_grid = {
        'hidden_layer_sizes': [(100,), (150,), (200,)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.005, 0.001],
        'learning_rate': ['constant', 'invscaling',  'adaptive']
    }
    # grid search cv
    grid_search = GridSearchCV(estimator=clf,
                               param_grid=param_grid,
                               scoring={"Accuracy": make_scorer(accuracy_score)},
                               refit='Accuracy',
                               cv=5,
                               n_jobs=-1)

    # Fit the grid search to the data
    print('Running grid search cv...')
    grid_search.fit(X_train, y_train)
    # grid_search.best_params_
    best_clf = grid_search.best_estimator_

    probs = best_clf.predict_proba(X_test)

    # CV scoring
    cv_scoring = cross_validate_scoring(best_clf, X_test, y_test, ['accuracy', 'f1'], cv=5, results_dir=results_dir)


    # Save model using pickle
    with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
        pickle.dump(best_clf, f)

    # Logging results to wandb
    # --- Logging metrics
    wandb_exp.log({
        'Test set CV accuracies': cv_scoring['test_accuracy'],
        'Average test set accuracy': cv_scoring['test_accuracy'].mean(),
        'Average test set F1': cv_scoring['test_f1'].mean(),
        'Best Model Params': grid_search.best_params_
    })

    # --- Logging plots
    wandb_exp.log({
        'roc': wandb.plot.roc_curve(y_test, probs)
    })

    # Saving results for further plotting ignore for now
    results_for_plotting(y_test, probs, test_latitudes, test_longitudes, results_dir, model_name)