"""
gb pipeline for ML call
"""

from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.model_selection import GridSearchCV
import wandb
from analysis.generating_results import cross_validate_scoring, results_for_plotting
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from analysis import confusion_matrix


def run_gb(X_train,
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
    Run gb model
    """

    model_name = 'gb'

    # Create instance of GB model
    print('Creating instance of GB model...')
    clf = GradientBoostingClassifier(random_state=48)

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
        cv_scoring = cross_validate_scoring(clf, X_train, y_train, ['accuracy', 'f1'], cv=5,
                                            results_dir=results_dir, prefix_name=model_setup)
        with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
            pickle.dump(clf, f)

        predictions = (probs[:, 1] >= 0.5)
        predictions = predictions * 1
        f1 = f1_score(y_test, predictions)
        confusion_matrix(y_test, probs[:, 1], results_dir)

        # Saving results for further plotting
        results_for_plotting(y_test, probs, test_latitudes, test_longitudes, results_dir, model_name)

        wandb_exp.log({
            'roc': wandb.plot.roc_curve(y_test, probs)
        })

    else:
        model_setup = 'tuned'
        # Tune the model
        param_grid = {
            'loss': ['log_loss', 'exponential'],
            'learning_rate': [0.05, 0.1, 0.5, 1],
            'n_estimators': [100, 200, 300],
            'criterion': ['friedman_mse', 'squared_error'],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 3, 5],
            'max_features': ['sqrt', 'log2', None]
        }
        # grid search cv
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)

        # Fit the grid search to the data
        # Turning off grid search manually because it is taking so long and I just want to compare
        print('Running grid search cv...')
        grid_search.fit(X_train, y_train)
        # grid_search.best_params_
        best_clf = grid_search.best_estimator_

        # CV scoring
        cv_scoring = cross_validate_scoring(best_clf, X_train, y_train, ['accuracy', 'f1'], cv=5,
                                            results_dir=results_dir, prefix_name=model_setup)

        tuned_probs = best_clf.predict_proba(X_test)
        confusion_matrix(y_test, tuned_probs[:, 1], results_dir)
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