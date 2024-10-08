"""
MLP pipeline for ML call
"""

from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import wandb
from analysis.generating_results import cross_validate_scoring, results_for_plotting
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from analysis.confusion_matrix import calc_confusion_matrix
import random


def run_mlp(X_train,
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
    Run mlp model
    """

    model_name = 'mlp'
    seed = random.randint(0, 1000)

    # Create instance of MLP model
    print('Creating instance of MLP model...')
    clf = MLPClassifier(random_state=seed, max_iter=15000)

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
        # CV scoring
        cv_scoring = cross_validate_scoring(clf, X_train, y_train, ['accuracy', 'f1'], cv=5,
                                                   results_dir=results_dir, prefix_name=model_setup)

        # Save model using pickle
        with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
            pickle.dump(clf, f)

        predictions = (probs[:, 1] >= 0.5)
        predictions = predictions * 1
        f1 = f1_score(y_test, predictions)
        #confusion_matrix(y_test, probs[:, 1], results_dir)

        # Saving results for further plotting
        results_for_plotting(y_test, probs, test_latitudes, test_longitudes, results_dir, model_name)


        # --- Logging metrics
        wandb_exp.log({
            'CV accuracies': cv_scoring['test_accuracy'],
            'Average CV accuracy': cv_scoring['test_accuracy'].mean(),
            'Average CV F1': cv_scoring['test_f1'].mean(),
            'Test set F1': f1,
            'Test set accuracy': accuracy_score(y_test, predictions),
            'roc': wandb.plot.roc_curve(y_test, probs)
        })

    else:
        model_setup = 'tuned'
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
        grid_search.fit(X_val, y_val)
        # grid_search.best_params_
        best_clf = grid_search.best_estimator_

        # Fit our best model with training set
        best_clf.fit(X_train, y_train)

        tuned_probs = best_clf.predict_proba(X_test)
        calc_confusion_matrix(y_test, tuned_probs[:, 1], results_dir)

        predictions = (tuned_probs[:, 1] >= 0.5)
        predictions = predictions * 1
        f1 = f1_score(y_test, predictions)

        with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
            pickle.dump(clf, f)

        # Saving results for further plotting
        results_for_plotting(y_test, tuned_probs, test_latitudes, test_longitudes, results_dir, model_name)

        # --- Logging metrics
        wandb_exp.log({
            'Best Model Params': grid_search.best_params_,
            'roc': wandb.plot.roc_curve(y_test, tuned_probs),
            'Test set F1': f1,
            'Test set accuracy': accuracy_score(y_test, predictions)
        })

