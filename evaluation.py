import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import sys
import xlrd
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import read_csv_file
import os


# read the ground truth from xlsx file and output case id and eight labels
def importGT(filepath):
    data = xlrd.open_workbook(filepath)
    table = data.sheets()[0]
    data = [[int(table.row_values(i, 0, 1)[0])] + table.row_values(i, -8) for i in range(1, table.nrows)]
    return np.array(data)


# read the submitted predictions in csv format and output case id and eight labels 
def importPR(gt_data, filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        pr_data = [[int(row[0])] + list(map(float, row[1:])) for row in reader]
    pr_data = np.array(pr_data)

    # Sort columns if they are not in predefined order
    order = ['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    order_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    order_dict = {item: ind for ind, item in enumerate(order)}
    sort_index = [order_dict[item] for ind, item in enumerate(header) if item in order_dict]
    wrong_col_order = 0
    if (sort_index != order_index):
        wrong_col_order = 1
        pr_data[:, order_index] = pr_data[:, sort_index]

        # Sort rows if they are not in predefined order
    wrong_row_order = 0
    order_dict = {item: ind for ind, item in enumerate(gt_data[:, 0])}
    order_index = [v for v in order_dict.values()]
    sort_index = [order_dict[item] for ind, item in enumerate(pr_data[:, 0]) if item in order_dict]
    if (sort_index != order_index):
        wrong_row_order = 1
        pr_data[order_index, :] = pr_data[sort_index, :]

    # If have missing results
    missing_results = 0
    if (gt_data.shape != pr_data.shape):
        missing_results = 1
    return pr_data, wrong_col_order, wrong_row_order, missing_results


# calculate kappa, F-1 socre and AUC value
def ODIR_Metrics_for_classes(gt_data, pr_data, name):
    classes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    th = 0.5
    # gt = gt_data.flatten()
    # pr = pr_data.flatten()
    result_list = []
    labels = ["accuracy", "precision", "sensitivity", "specifity", "kappa", "f1", "auc", "final score"]
    result_list.append(labels)
    for i, classification in enumerate(classes):
        gt = gt_data[:, i]
        pr = pr_data[:, i]
        classified_pr = [1 if i >= th else 0 for i in pr]
        cm = confusion_matrix(gt, classified_pr)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure()
        disp.plot()
        title_name = name.replace('_', ' ')
        plt.title(f'{title_name} {classes[i]}')
        plt.savefig(f'{name}/{name}_{classes[i]}.png')
        tn, fp, fn, tp = confusion_matrix(gt, classified_pr).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        precision = (tp) / (fp + tp)
        sensitivity = (tp) / (tp + fn)
        specifity = (tn) / (tn + fp)
        kappa = metrics.cohen_kappa_score(gt, pr > th)
        f1 = metrics.f1_score(gt, pr > th, average='micro')
        auc = metrics.roc_auc_score(gt, pr)
        final_score = (kappa + f1 + auc) / 3.0

        result = [accuracy, precision, sensitivity, specifity, kappa, f1, auc, final_score]
        result_list.append(result)

    with open(f'{name}/{name}_class_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_list)
    # return kappa, f1, auc, final_score


# calculate kappa, F-1 score and AUC value
def ODIR_Metrics(gt_data, pr_data, name):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    classified_pr = [1 if i >= th else 0 for i in pr]
    tn, fp, fn, tp = confusion_matrix(gt, classified_pr).ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = (tp) / (fp + tp)
    sensitivity = (tp) / (tp + fn)
    specifity = (tn) / (tn + fp)
    kappa = metrics.cohen_kappa_score(gt, pr > th)
    f1 = metrics.f1_score(gt, pr > th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0

    result_list = []
    labels = ["accuracy", "precision", "sensitivity", "specifity", "kappa", "f1", "auc", "final score"]
    result_list.append(labels)
    result = [accuracy, precision, sensitivity, specifity, kappa, f1, auc, final_score]
    result_list.append(result)
    with open(f'{name}/{name}_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_list)


# calculate kappa, F-1 socre and AUC value
def confusion_matrix_for_all(gt_data, pr_data, name, matrix_name, title_name):
    gt_data = np.argmax(gt_data, axis=-1)
    pr_data = np.argmax(pr_data, axis=-1)
    cm = confusion_matrix(gt_data, pr_data)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    plt.figure()
    disp.plot()
    title_name = title_name.replace('_', ' ')
    plt.title(f'{title_name}')
    plt.savefig(f'{name}/{matrix_name}.png')
    # np.savetxt('inception_overall_processed_test_confusion_matrix.csv', cm, delimiter=',', fmt='%i')

    # return kappa, f1, auc, final_score


def calculate_score(GT_filepath, PR_filepath, name):
    try:
        os.mkdir(f"{name}")
        print(f"Directory {name} Created ")
    except FileExistsError:
        print(f"Directory {name} already exists")
    val_annotations = read_csv_file(GT_filepath)
    val_annotations = val_annotations[1:]
    gt_data = []
    for ann in val_annotations:
        gt_data.append([int(ann[0]), float(ann[-8]), float(ann[-7]), float(ann[-6]), float(ann[-5]), float(ann[-4]),
                        float(ann[-3]), float(ann[-2]), float(ann[-1])])
    gt_data = np.array(gt_data)
    pr_data, wrong_col_order, wrong_row_order, missing_results = importPR(gt_data, PR_filepath)

    ODIR_Metrics(gt_data[:, 1:], pr_data[:, 1:], name=name)
    ODIR_Metrics_for_classes(gt_data[:, 1:], pr_data[:, 1:], name=name)
    confusion_matrix_for_all(gt_data=gt_data[:, 1:], pr_data=pr_data[:, 1:], name=name, matrix_name=name, title_name=name)


calculate_score('Annotations/test.csv', 'inception_test_result.csv', name='inception_test')
# Run in terminal python evaluation.py on_site_test_annotation.xlsx *_result.csv
