"""
Logistic Regression pipeline for ML call
"""

from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.model_selection import cross_validate
import wandb


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

    # Create instance of Logistic Regression model
    print('Creating instance of LR model...')
    clf = LogisticRegression(solver='lbfgs', max_iter=200, random_state=48)

    # Fit to training data
    print('Fitting data...')
    clf.fit(X_train, y_train)

    # Measure model performance on test set
    probs = clf.predict_proba(X_test)

    accuracy = clf.score(X_test, y_test)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')
    import ipdb
    ipdb.set_trace()

    # CV scoring
    cv_scoring = cross_validate(clf, X_test, y_test, scoring=['accuracy', 'f1'], cv=5)
    print('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
    print('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
    print('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
    print('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))

    with open('{}/lr_results.txt'.format(results_dir), 'w') as f:
        f.write(f'The hard predictions were right on the test set {100 * accuracy:5.2f}% of the time')
        f.write('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
        f.write('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
        f.write('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
        f.write('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))

    # Save model using pickle
    with open('{}/lr_model.pkl'.format(results_dir), 'wb') as f:
        pickle.dump(clf, f)

    # Save the preds with labels and location to df for further plotting
    df_results = pd.DataFrame(columns=['label', 'prediction', 'lat', 'lon'])
    df_results['label'] = y_test
    df_results['prediction'] = probs[:, 1] >= 0.5
    df_results['lat'] = test_latitudes
    df_results['lon'] = test_longitudes
    df_results.to_csv('{}/lr_results_for_plotting.csv'.format(results_dir))

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


