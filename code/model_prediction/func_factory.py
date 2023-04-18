from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
import import_data.find_paths as find_paths

def get_scores(sub, cond, cam, task, side, block):

    read_scores = pd.read_excel(
        os.path.join(
            find_paths.find_onedrive_path('patientdata'),
            'scores_JB_JH_JR.xlsx',
        ),
        usecols='A:I',
    )

    read_scores.set_index('sub_cond_cam', inplace = True)

    if side == 'left': side='lh'
    elif side == 'right': side='rh'

    # read scores for all blocks of a subject in the same cond, cam per side
    ext_scores = read_scores.loc[f'{sub}_{cond}_{cam}'][f'{task}_{side}']

    if type(ext_scores) != float:

        if isinstance(ext_scores, int):
            ls_int_sc = [ext_scores,]
        else:
            ls_int_sc = [int(s) for s in ext_scores if s in ['0', '1', '2', '3', '4']]


        if block == 'b1':
            score = ls_int_sc[0]
        elif block == 'b2':
            try:
                score = ls_int_sc[1]
            except IndexError:
                score = ls_int_sc[0]

        elif block == 'b3':
            score = ls_int_sc[2]
        else:
            print(f'no scores for block {block} or block does not exist')

        return score


def load_labels_and_adjust_feature_df(ft_df):
    y = []  # list to store labels
    updated_df = ft_df.copy()  # create a copy of the original DataFrame

    ids = ft_df['filename']
    if ids[0].startswith('feat'):
        ids = [x[5:-5] for x in ids]
    else:
        ids = [x[:-5] for x in ids]

    ids = [x.split('_') for x in ids]

    drop_indexes = []  # list to store indexes of rows to be dropped

    for i, id_list in enumerate(ids):
        block, sub, cond, cam, task, side = id_list
        value = get_scores(sub, cond, cam, task, side, block)
        if value is not None:
            y.append(value)

        else:
            drop_indexes.append(i)

    # Drop the rows from the updated DataFrame based on the indexes
    updated_df = updated_df.drop(updated_df.index[drop_indexes])

    return updated_df, y
def split_dataframe_by_scores(df, score_column):
    '''
    Split a dataframe into four dataframes based on unique values in a specified column.

    Args:
        df (pd.DataFrame): The input dataframe.
        score_column (str): The column name in the dataframe containing the scores.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame: Four separate dataframes, each containing
        the rows with scores 0, 1, 2, and 3 respectively.
    '''
    # Filter the dataframe based on each score
    df_0 = df[df[score_column] == 0]
    df_1 = df[df[score_column] == 1]
    df_2 = df[df[score_column] == 2]
    df_3 = df[df[score_column] == 3]
    df_4= df[df[score_column] == 4]

    return df_0, df_1, df_2, df_3, df_4


def boxplot_by_dataframe(dataframes, column_name):
    '''
    Create a box plot for a specified column in multiple dataframes in a single figure,
    with all dataframes plotted on the x-axis.

    Args:
        dataframes (list): A list of input dataframes.
        column_name (str): The column name to create the box plot for.

    Returns:
        None
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a list of dataframes' column values for the specified column
    data = [df[column_name] for df in dataframes]

    labels = ['UPDRS Score 0', 'UPDRS Score 1', 'UPDRS Score 2', 'UPDRS Score 3', 'UPDRS Score 4']
    ax.boxplot(data, labels=labels)
    
    ax.set_xlabel('Dataframes')
    ax.set_ylabel(column_name)
    
    ax.set_title(f'Box plot for {column_name} in {len(dataframes)} Dataframes', fontsize=16)

    plt.show()

    return


def delete_feat_col(df, feat_to_drop = list):

    columns_to_delete = feat_to_drop

    df = df.drop(columns=columns_to_delete)

    return df


def feat_selection_RFE(df, y, groups, min_features= int, scoring= 'r2', step=1, n_jobs=-1):
    # create a Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # LOOCV
    loocv = GroupKFold(n_splits=len(set(groups)))

    # create an RFECV object for recursive feature selection
    selector = RFECV(estimator=rf, min_features_to_select=min_features, cv=loocv, scoring=scoring, step=step, n_jobs=n_jobs)

    # fit to the data
    selector.fit(df, y, groups=groups)

    # get the R-squared score
    r2_score = selector.score(df, y)
    
    feature_names = df.columns[selector.support_].tolist()

    return r2_score, feature_names


def feat_selection_importance(df, y, groups, k):

    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    importances = np.zeros((len(df.columns), len(groups)))

    loocv = GroupKFold(n_splits=len(set(groups)))

    for i, (train_index, test_index) in enumerate(loocv.split(df, y, groups)):
        X_train, y_train = df.iloc[train_index], y[train_index]

        rf.fit(X_train, y_train)

        importances[:, i] = rf.feature_importances_

    # average the feature importances across all folds
    mean_importances = np.mean(importances, axis=1)

    # sort feature importances (most to least)
    indices = np.argsort(mean_importances)[::-1]

    # select the top K features
    top_k_indices = indices[:k]
    top_k_features = df.iloc[:, top_k_indices]
    top_k_importances = mean_importances[top_k_indices]

    return pd.DataFrame(
        {'feature': top_k_features.columns, 'importance': top_k_importances}
    )


def define_groups(df, name = str):

    filenames = df['filename']
    name = name.lower()

    if name in ['sub', 'subs', 'subject', 'subjects', 'sub_id', 'sub_ids']:
        if filenames[0].startswith('feat'): filename = [x[8:13] for x in filenames]

        sub_per_sample = dict(enumerate(filename))

        groups = [sub_per_sample[sample] for sample in range(len(filename))]
    
    elif name in ['cond', 'conds', 'condition', 'conditions']:
        if filenames[0].startswith('feat'): filename= [x[14:18] for x in filenames]

        file = []
        for i in filename:
            if '_' in i:
                file.append(i[:-2])
            else:
                file.append(i)

        cond_per_sample = dict(enumerate(file))

        groups = [cond_per_sample[sample] for sample in range(len(file))]
    
    return groups

def predictions(X, y, classifier_list=list, groups=list):

    Accs = []
    Precisions = []
    Recalls = []
    F1_scores = []
    # AUC_ROCs = []

    loo = GroupKFold(n_splits=len(set(groups)))

    for c in classifier_list:
        c = c.lower()
        if c == 'linear':
            classifier = SVC(kernel='linear')
        elif c == 'nonlinear':
            classifier = SVC(kernel='rbf')
        elif c == 'logregression':
            classifier = LogisticRegression(max_iter= 1000)
        elif c == 'random forest':
            classifier = RandomForestClassifier(random_state=42)
        elif c == 'knearest':
            classifier = KNeighborsClassifier()
        elif c == 'gaussian naive bayes':
            classifier = GaussianNB()
        else:
            raise ValueError(f'Invalid classifier type. Check spelling for {c}')

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        # auc_roc_scores = []

        for i, (train_index, test_index) in enumerate(loo.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # # Enable for tracking of fold characteristics
            # test_sub = {groups[index] for index in test_index}
            # train_subs = {groups[index] for index in train_index}

            # print(f'Fold number {i} for {c}')
            # print(f'  Train: {len(train_index)} files for {train_subs}')
            # print(f'  Test: {len(test_index)} files for {test_sub}')
            # print(f'train scores: {y_train}')
            # print(f'test scores: {y_test}')
            
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            # auc_roc = roc_auc_score(y_test, y_pred)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            # # auc_roc_scores.append(auc_roc)

        mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        mean_precision = sum(precision_scores) / len(precision_scores)
        mean_recall = sum(recall_scores) / len(recall_scores)
        mean_f1 = sum(f1_scores) / len(f1_scores)
        # # # mean_auc_roc = sum(auc_roc_scores) / len(auc_roc_scores)

        Accs.append(mean_accuracy)
        Precisions.append(mean_precision)
        Recalls.append(mean_recall)
        F1_scores.append(mean_f1)
        # # AUC_ROCs.append(mean_auc_roc)
        
    x_ticks = np.arange(len(classifier_list)) - 0.1
    plt.figure(figsize=(20, 14))
    plt.title('Preliminary comparison of model performance metrcis FT \n selected features \n  cross validation: subs',  pad=15, fontsize = 25)
    plt.plot(x_ticks + 0.05, Accs, '-o', markersize = 10, label='Accuracy')
    plt.plot(x_ticks - 0.05, Precisions, '-o', markersize = 10, label='Precision')
    plt.plot(x_ticks + 0.05, Recalls, '-o', markersize = 10, label='Recall')
    plt.plot(x_ticks - 0.05, F1_scores, '-o', markersize = 10, label='F1-score')
    # plt.plot(x_ticks + 0.05, AUC_ROCs, '-o', markersize = 10, label='AUC-ROC')
    plt.xticks(x_ticks, classifier_list, fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.vlines(x_ticks, ymin = 0, ymax = 0.7, linestyles='dashed', alpha = 0.4, color = 'black')
    plt.ylabel('Value',fontsize = 20, labelpad = 10)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    # # Enable for automatic saving
    # fname = 'selected_features_pred_subs'
    # plt.tight_layout( pad = 5)
    # plt.savefig(
    #     os.path.join(
    #         '/Users/arianm/Documents/GitHub/ultraleap_analysis/figures/', fname
    #     ),
    #     dpi=300,
    #     facecolor='w',
    # )
    plt.show()

    return