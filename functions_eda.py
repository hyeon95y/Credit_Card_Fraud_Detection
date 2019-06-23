import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def mutual_info_table(df) : 
    temp = pd.DataFrame(columns=df.columns)
    for index in range(len(df.columns)) : 
        temp.loc[index] = np.zeros(len(df.columns))
    temp.index = df.columns
    
    for i, column in enumerate(df.columns) : 
        for j, index in enumerate(df.columns) :
            temp.iloc[i, j] = normalized_mutual_info_score(df[df.columns[i]], df[df.columns[j]])
            
    return temp

def input_distribution(df) :
    for column in df.drop(['Class'], axis=1).columns :
        plt.figure(figsize=(20, 4))

        plt.subplot(1, 3, 1)
        sns.boxplot(df[column])
        plt.title(column + ' boxplot')
        plt.xlabel('value')

        plt.subplot(1, 3, 2)
        sns.distplot(df[column])
        plt.title(column + ' distribution plot')
        plt.xlabel('value')
        plt.ylabel('probability')


        plt.subplot(1, 3, 3)
        sns.scatterplot(list(range(df.shape[0])), df[column], s=5, alpha=0.7)
        plt.title(column + ' scatter plot')
        plt.xlabel('index')
        plt.ylabel('value')

        plt.show()
    return

def input_distribution_upon_class(df) :
    df_normal = df[df['Class']==0]
    df_abnormal = df[df['Class']==1]
    
    for column in df.drop(['Class'], axis=1).columns :
        plt.figure(figsize=(20, 4))

        plt.subplot(1, 3, 1)
        sns.boxplot(df_normal[column], color='skyblue', boxprops=dict(alpha=.3))
        sns.boxplot(df_abnormal[column], color='orange', boxprops=dict(alpha=.3))
        plt.title(column + ' boxplot')
        plt.xlabel('value')
        skyblue_patch = mpatches.Patch(color='skyblue', label='normal')
        orange_patch = mpatches.Patch(color='orange', label='abnormal')
        plt.legend(handles=[skyblue_patch, orange_patch])

        plt.subplot(1, 3, 2)
        sns.distplot(df_normal[column], color='skyblue', label='normal')
        sns.distplot(df_abnormal[column], color='orange', label='abnormal')
        plt.title(column + ' distribution plot')
        plt.xlabel('value')
        plt.ylabel('probability')
        plt.legend()

        plt.subplot(1, 3, 3)
        sns.scatterplot(df_normal.index, df_normal[column], s=20, alpha=0.3, color='skyblue', label='normal')
        sns.scatterplot(df_abnormal.index, df_abnormal[column], s=20, alpha=0.3, color='orange', label='abnormal')
        plt.title(column + ' scatter plot')
        plt.xlabel('index')
        plt.ylabel('value')
        plt.legend()

        plt.show()
    return