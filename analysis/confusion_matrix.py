"""
Code to quickly generate confusion matrix from df of saved results
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

def calc_confusion_matrix(dataframe, savedir):
    """
    Calculates confusion matrix
    """
    ytest = dataframe['label']
    ypreds = dataframe['prediction']*1

    CM = confusion_matrix(ytest, ypreds)
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

aoi = 'RWA'
root_dir = '/Users/kelseydoerksen/Desktop/Giga/{}/results_1000m'.format(aoi)
BWA_folders = ['frosty-vortex-524/lr_', 'earnest-night-525/svm_', 'floral-aardvark-526/rf_',
               'silver-dew-527/mlp_', 'apricot-music-528/gb_',
               'hopeful-rain-498/rf_', 'devout-firefly-509/mlp_', 'jumping-lake-529/svm_',
               'colorful-glade-530/lr_', 'silvery_capybara-542/gb_']

RWA_folders = ['sweet-snowflake-469/gb_', 'divine-eon-477/rf_', 'brisk-universe-511/mlp_', 'honest-hill-536/lr_',
               'gentle-fog-537/svm_', 'comic-glade-531/lr_', 'sandy-smoke-532/svm_', 'effortless-night-533/rf_',
               'amber-yogurt-534/mlp_', 'balmy-bush-535/gb_']

IJCAI_folders = ['glamorous-smoke-513/mlp_']

for folder in IJCAI_folders:
    df = pd.read_csv('{}/{}results_for_plotting.csv'.format(root_dir, folder))
    head, sep, tail = folder.partition('/')
    print('Running for folder: {}'.format(head))
    save_directory = root_dir + '/' + head
    calc_confusion_matrix(df, save_directory)














