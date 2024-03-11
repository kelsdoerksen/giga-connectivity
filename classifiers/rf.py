"""
Random Forest ML call for pipeline
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay, make_scorer, accuracy_score, f1_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import wandb
import pickle
from analysis.generating_results import cross_validate_scoring, results_for_plotting
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
    print('Model accuracy is: {}'.format(model_score))

    if not eval(tuning):
        print('No model hyperparameter tuning')
        model_setup = 'non-tuned'
        cv_scoring = cross_validate_scoring(forest, X_train, y_train, ['accuracy', 'f1'], cv=5, results_dir=results_dir,
                                                   prefix_name=model_setup)

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
            'roc': wandb.plot.roc_curve(y_test, probs)
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
        grid_search = GridSearchCV(estimator=forest,
                                   param_grid=param_grid,
                                   scoring={"Accuracy": make_scorer(accuracy_score)},
                                   refit='Accuracy',
                                   cv=5,
                                   n_jobs=-1)

        # Fit the grid search to the data
        print('Running grid search cv...')
        grid_search.fit(X_train, y_train)

        #grid_search.best_params_
        best_forest = grid_search.best_estimator_

        # CV scoring
        cv_scoring = cross_validate_scoring(best_forest, X_train, y_train, ['accuracy', 'f1'], cv=5,
                                            results_dir=results_dir, prefix_name=model_setup)

        # Prediction with best forest
        tuned_probs = best_forest.predict_proba(X_test)

        with open('{}/{}_model.pkl'.format(results_dir, model_name), 'wb') as f:
            pickle.dump(best_forest, f)

        # Saving results for further plotting
        results_for_plotting(y_test, tuned_probs, test_latitudes, test_longitudes, results_dir, model_name)

        predictions = (tuned_probs[:, 1] >= 0.5)
        predictions = predictions*1
        f1 = f1_score(y_test, predictions)
        calc_importance(best_forest, X_test, results_dir)
        calc_confusion_matrix(y_test, tuned_probs[:, 1], results_dir)

        wandb_exp.log({
            'Best Model Params': grid_search.best_params_,
            'roc': wandb.plot.roc_curve(y_test, tuned_probs)
        })


    '''
    # Old - Wandb will generate
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
    '''

    # Logging results to wandb
    # --- Logging metrics
    wandb_exp.log({
        'CV accuracies': cv_scoring['test_accuracy'],
        'Average CV accuracy': cv_scoring['test_accuracy'].mean(),
        'Average CV F1': cv_scoring['test_f1'].mean(),
        'Test set F1': f1,
        'Test set accuracy': accuracy_score(y_test, predictions)
    })
