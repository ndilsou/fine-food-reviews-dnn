from typing import Sequence, Union
from pathlib import Path
import json
import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as ms, metrics

def evaluate_prediction(y_true: Sequence[int], y_pred: Sequence[int], label: str = None, show: bool = True) -> dict:
    '''
    Generates a report of evaluation metric for the predictions of an estimator.
    Some vizualisation can also be generated if the option show is set to True.
    Metrics:
    - AUROC
    - Matthews Correlation Coeficient
    - f1 score
    - normalized confusion matrix
    '''

    auroc = metrics.roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred))
    f1score = metrics.f1_score(y_true, y_pred, average='macro')
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    normalized_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    report = {
        'auroc': auroc,
        'f1score': f1score,
        'normalized_confusion_matrix': normalized_confusion_matrix.tolist(),
        'matthew_corrcoef': mcc
    }

    if show:
        print(f'AUROC: {auroc:.3f}')
        print(f'MCC: {mcc:.3f}')

        print(metrics.classification_report(y_true, y_pred))

        fig, ax = plt.subplots(1, 1, figsize = (10, 10))

        title = 'confusion matrix'

        if label:
            title +=  f' for {label}'
        ax.set_title(title)

        sns.heatmap(normalized_confusion_matrix,
                    annot=True,
                    vmin=0,
                    vmax=1,
                    square=True,
                    xticklabels=range(1, 6),
                    yticklabels=range(1, 6));
        ax.set_ylabel('True Score');
        ax.set_xlabel('Predicted Score');
        plt.show()

    return report


def save_report(report: dict, dirpath: Union[str, Path],label: str) -> None:
    '''
    Saves an evaluation report in json.
    '''

    with open(os.path.join(dirpath, f'{label}.json'), 'wt') as f:
        json.dump(report, f)
