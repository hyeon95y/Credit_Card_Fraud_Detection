import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

def auroc_plot(y_valid, y_valid_pred, y_test, y_test_pred) : 
    plt.figure(figsize=(20, 10))
    
    # valid
    plt.subplot(1, 2, 1)
    fpr, tpr, threshold = roc_curve(y_valid, y_valid_pred)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic (Valid Set)')
    plt.plot(fpr, tpr, label='AUROC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    # test
    plt.subplot(1, 2, 2)
    fpr, tpr, threshold = roc_curve(y_test, y_test_pred)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic (Test Set)')
    plt.plot(fpr, tpr, label='AUROC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    
    plt.show()
    return

def auprc_plot(y_valid, y_valid_pred, y_test, y_test_pred) :
    plt.figure(figsize=(20, 10))
    
    # valid
    plt.subplot(1, 2, 1)
    precision, recall, threshold = precision_recall_curve(y_valid, y_valid_pred)
    prc_auc = auc(recall, precision)
    plt.plot(recall, precision, 'b', label='AUPRC = %0.4f' % prc_auc)
    plt.legend(loc='upper right')
    plt.title('Recall vs Precision (Valid Set)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # test
    plt.subplot(1, 2, 2)
    precision, recall, threshold = precision_recall_curve(y_test, y_test_pred)
    prc_auc = auc(recall, precision)
    plt.plot(recall, precision, 'b', label='AUPRC = %0.4f' % prc_auc)
    plt.legend(loc='upper right')
    plt.title('Recall vs Precision (Test Set)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.show()
