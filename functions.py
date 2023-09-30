import numpy as np
from sklearn.metrics import auc

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def validate_ijb(labels, distances, ts, fmr_p):
    true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc = \
    evaluate_lfw_sf(
        distances=distances,
        labels=labels,
        far_target=1e-3
    )

    global_h = find_nearest(false_positive_rate, fmr_p)

    tpr_1e3 = true_positive_rate[np.argmin(np.abs(false_positive_rate - 1e-03))]
    tpr_1e4 = true_positive_rate[np.argmin(np.abs(false_positive_rate - 1e-04))]
    fpr_95 = false_positive_rate[np.argmin(np.abs(true_positive_rate - 0.95))]
    fnr = false_negative_rate
    fpr = false_positive_rate
#     print("fnr",fnr,"fpr",fpr)
    sub = np.abs(fnr - fpr)
    h = np.min(sub[np.nonzero(sub)])
    h = np.where(sub == h)[0][0]


    print("------------------------------------------------Single fold---------------------------------------")

    # Print statistics and add to log
    print("Accuracy on CC: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
          "ROC Area Under Curve: {:.4f}\t".format(
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc
            )
    )


    print('fpr at tpr 0.95: {},  tpr at fpr 0.001: {}, tpr at fpr 0.0001: {}'.format(fpr_95,tpr_1e3,tpr_1e4))
    print('At FNR = FPR: FNR = {}, FPR = {}'.format(fnr[h],fpr[h]))
# with open('logs/cc_tpr_fpr_{}_{}.txt'.format(logfname, ts), 'a') as f:
#             f.writelines(''.format()

#     with open('logs/log_stats_{}.txt'.format(ts), 'a') as f:

#         f.writelines("--------------------------single fold-------------------------------"
#           "Accuracy on CC: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
#           "ROC Area Under Curve: {:.4f}\t"
#           "fpr at tpr 0.95: {},  tpr at fpr 0.001: {} | At FNR = FPR: FNR = {}, FPR = {}".format(
#                 np.mean(accuracy),
#                 np.std(accuracy),
#                 np.mean(precision),
#                 np.std(precision),
#                 np.mean(recall),
#                 np.std(recall),
#                 roc_auc,
#                 fpr_95,
#                 tpr_1e3,
#                 fnr[h],
#                 fpr[h]
#             ) + '\n'
#         )

    return tpr_1e3,tpr_1e4,fpr_95, h, global_h, (fnr[h]+fpr[h])/2

def calculate_FNR_FPR_thresh(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)
#     print(predict_issame)
    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)
        
    false_negative_rate = 0 if (false_negatives + true_positives == 0) else \
        float(false_negatives) / float(false_negatives + true_positives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy

def evaluate_lfw_sf(distances, labels, far_target=1e-3):
    """Evaluates on the Labeled Faces in the Wild dataset using KFold cross validation based on the Euclidean
    distance as a metric.
    Note: "TAR@FAR=0.001" means the rate that faces are successfully accepted (True Acceptance Rate) (TP/(TP+FN)) when
    the rate that faces are incorrectly accepted (False Acceptance Rate) (FP/(TN+FP)) is 0.001 (The less the FAR value
    the mode difficult it is for the model). i.e: 'What is the True Positive Rate of the model when only one false image
    in 1000 images is allowed?'.
        https://github.com/davidsandberg/facenet/issues/288#issuecomment-305961018
    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds.
        far_target (float): The False Acceptance Rate to calculate the True Acceptance Rate (TAR) at,
                             defaults to 1e-3.
    Returns:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds for plotting
                             the Receiver operating characteristic (ROC) curve.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds for plotting
                              the Receiver operating characteristic (ROC) curve.
        accuracy: Array of accuracy values per each fold in cross validation set.
        precision: Array of precision values per each fold in cross validation set.
        recall: Array of recall values per each fold in cross validation set.
        roc_auc: Area Under the Receiver operating characteristic (AUROC) metric.
        best_distances: Array of Euclidean distance values that had the best performing accuracy on the LFW dataset
                         per each fold in cross validation set.
        tar: Array that contains True Acceptance Rate values per each fold in cross validation set
              when far (False Accept Rate) is set to a specific value.
        far: Array that contains False Acceptance Rate values per each fold in cross validation set.
    """

    # Calculate ROC metrics
    thresholds_roc = np.arange(0, 3.0, 0.001)
    true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy = \
        calculate_roc_values_sf(
            thresholds=thresholds_roc, distances=distances, labels=labels
        )

    roc_auc = auc(false_positive_rate, true_positive_rate)

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc


def calculate_roc_values_sf(thresholds, distances, labels):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)

    true_positive_rates = np.zeros((num_thresholds))
    false_positive_rates = np.zeros((num_thresholds))
    false_negative_rates = np.zeros((num_thresholds))

    test_set = np.arange(num_pairs)
    
#     print(test_set)
    

        # Test on test set using the best distance threshold
    for threshold_index, threshold in enumerate(thresholds):
#         print("threshold_index: ",threshold_index)
#         print("threshold: ", threshold)
        true_positive_rates[threshold_index], false_positive_rates[threshold_index], false_negative_rates[threshold_index], _, _,\
            _ = calculate_metrics_sf(
                threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set])

    _, _, _, precision, recall, accuracy = calculate_metrics_sf(
        threshold=thresholds[threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
    )


    return true_positive_rates, false_positive_rates, false_negative_rates, precision, recall, accuracy


def calculate_metrics_sf(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)
        
    false_negative_rate = 0 if (false_negatives + true_positives == 0) else \
        float(false_negatives) / float(false_negatives + true_positives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy







def evaluate_lfw(distances, labels, num_folds=5, far_target=1e-3):
    """Evaluates on the Labeled Faces in the Wild dataset using KFold cross validation based on the Euclidean
    distance as a metric.
    Note: "TAR@FAR=0.001" means the rate that faces are successfully accepted (True Acceptance Rate) (TP/(TP+FN)) when
    the rate that faces are incorrectly accepted (False Acceptance Rate) (FP/(TN+FP)) is 0.001 (The less the FAR value
    the mode difficult it is for the model). i.e: 'What is the True Positive Rate of the model when only one false image
    in 1000 images is allowed?'.
        https://github.com/davidsandberg/facenet/issues/288#issuecomment-305961018
    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds.
        far_target (float): The False Acceptance Rate to calculate the True Acceptance Rate (TAR) at,
                             defaults to 1e-3.
    Returns:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds for plotting
                             the Receiver operating characteristic (ROC) curve.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds for plotting
                              the Receiver operating characteristic (ROC) curve.
        accuracy: Array of accuracy values per each fold in cross validation set.
        precision: Array of precision values per each fold in cross validation set.
        recall: Array of recall values per each fold in cross validation set.
        roc_auc: Area Under the Receiver operating characteristic (AUROC) metric.
        best_distances: Array of Euclidean distance values that had the best performing accuracy on the LFW dataset
                         per each fold in cross validation set.
        tar: Array that contains True Acceptance Rate values per each fold in cross validation set
              when far (False Accept Rate) is set to a specific value.
        far: Array that contains False Acceptance Rate values per each fold in cross validation set.
    """

    # Calculate ROC metrics
    thresholds_roc = np.arange(0, 3.0, 0.001)
    true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, best_distances = \
        calculate_roc_values(
            thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
        )

    roc_auc = auc(false_positive_rate, true_positive_rate)

    #Calculate validation rate
    thresholds_val = np.arange(0, 3, 0.001)
    tar, far = calculate_val(
        thresholds_val=thresholds_val, distances=distances, labels=labels, far_target=far_target, num_folds=num_folds
    )

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc, best_distances, tar, far


def calculate_roc_values(thresholds, distances, labels, num_folds=2):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    true_positive_rates = np.zeros((num_folds, num_thresholds))
    false_positive_rates = np.zeros((num_folds, num_thresholds))
    false_negative_rates = np.zeros((num_folds, num_thresholds))
    precision = np.zeros(num_folds)
    recall = np.zeros(num_folds)
    accuracy = np.zeros(num_folds)
    best_distances = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best distance threshold for the k-fold cross validation using the train set
        accuracies_trainset = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            # print(threshold)
            _, _, _, _,_, accuracies_trainset[threshold_index] = calculate_metrics(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        best_threshold_index = np.argmax(accuracies_trainset)

        # Test on test set using the best distance threshold
        for threshold_index, threshold in enumerate(thresholds):
            true_positive_rates[fold_index, threshold_index], false_positive_rates[fold_index, threshold_index], false_negative_rates[fold_index, threshold_index], _, _,\
                _ = calculate_metrics(
                    threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
                )

        _, _, _, precision[fold_index], recall[fold_index], accuracy[fold_index] = calculate_metrics(
            threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
        )

        true_positive_rate = np.mean(true_positive_rates, 0)
        false_positive_rate = np.mean(false_positive_rates, 0)
        false_negative_rate = np.mean(false_negative_rates, 0)
        best_distances[fold_index] = thresholds[best_threshold_index]

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, best_distances


def calculate_metrics(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)
        
    false_negative_rate = 0 if (false_negatives + true_positives == 0) else \
        float(false_negatives) / float(false_negatives + true_positives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy


def calculate_val(thresholds_val, distances, labels, far_target=1e-3, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds_val)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    tar = np.zeros(num_folds)
    far = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the euclidean distance threshold that gives false acceptance rate (far) = far_target
        far_train = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds_val):
            _, far_train[threshold_index] = calculate_val_far(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds_val, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        tar[fold_index], far[fold_index] = calculate_val_far(
            threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
        )

    return tar, far


def calculate_val_far(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    num_same = np.sum(actual_issame)
    num_diff = np.sum(np.logical_not(actual_issame))

    if num_diff == 0:
        num_diff = 1
    if num_same == 0:
        return 0, 0

    tar = float(true_accept) / float(num_same)
    far = float(false_accept) / float(num_diff)

    return tar, far

# def calc_fairness_metrics(list_of_distances, list_of_labels):
#     fmr_fnr_at_tg = []
#     global_h_full = 0
#     fmr_p = 0.001
#     G = len(list_of_labels) - 1
#     for  i, (distances,labels) in enumerate(zip(list_of_distances, list_of_labels)):
#         if i == 0:        
# #             _, _, _, h_full, global_h,eer = validate_ijb(
# #                         labels=labels_full,
# #                         distances=distances_full,
# #                         ts=0,
# #                         fmr_p=0.001
# #                     )
# #         global_h_full = global_h
#             global_h_full = 1217
#         else:
#             _, fmr, fnr, _, _, _ = calculate_FNR_FPR_thresh(global_h_full*0.001, distances, labels)
#             fmr_fnr_at_tg.append([fmr, fnr])
#     fmr_at_tg = [lst[0] for lst in fmr_fnr_at_tg if lst]
#     fnr_at_tg = [lst[1] for lst in fmr_fnr_at_tg if lst]
#     max_FMR = max(fmr_at_tg)
#     min_FMR = min(fmr_at_tg)
#     max_FNR = max(fnr_at_tg)
#     min_FNR = min(fnr_at_tg)
#     SER = max_FMR/min_FMR
#     MAPE_SUM = 0
#     for i in fnr_at_tg:
#         MAPE_SUM += (i - fmr_p)/fmr_p 
#     MAPE = (100/G) * MAPE_SUM
#     FDR_FMR_diff = max_FMR - min_FMR
#     FDR_FNR_diff = max_FNR - min_FNR
#     alpha_FDR = 0.5
#     FDR = 1 - (alpha_FDR * FDR_FMR_diff) - ((1 - alpha_FDR) * FDR_FNR_diff)
    
    
#     sum_th_eer = 0
#     eer_list = []
#     for  i, (distances,labels) in enumerate(zip(list_of_distances, list_of_labels)):
#         if i != 0:
#             _, _, _, th_eer, _,eer = validate_ijb(
#                         labels=labels,
#                         distances=distances,
#                         ts=0,
#                         fmr_p=0.001
#                     )
#             sum_th_eer += th_eer
#             eer_list.append(eer)
#     avg_th_eer = sum_th_eer/G
#     std_in_eer = np.std(eer_list)
#     fmr_fnr_at_te = []
#     for  i, (distances,labels) in enumerate(zip(list_of_distances, list_of_labels)):
#         if i == 0:
#             _, fpr_full, fnr_full, _, _, _ = calculate_FNR_FPR_thresh(avg_th_eer*0.001, distances_full, labels_full)
#         if i != 0:
#             _, fmr, fnr, _, _, _ = calculate_FNR_FPR_thresh(avg_th_eer*0.001, distances, labels)
#             fmr_fnr_at_te.append([fmr, fnr])
#     d_FMR_g = []
#     d_FNMR_g = []
#     fmr_at_te = [lst[0] for lst in fmr_fnr_at_te if lst]
#     fnmr_at_te = [lst[1] for lst in fmr_fnr_at_te if lst]
#     for i in fmr_fnr_at_te:
#         d_FMR = abs(1-(i/fpr_full))
#         d_FMR_g.append(d_FMR)
#     for i in fmr_fnr_at_te:
#         d_FNMR = abs(1-(i/fpr_full))
#         d_FNMR_g.append(d_FNMR)
#     SED_g = list(map(add, d_FNMR_g, d_FMR_g))
#     std_in_SED = np.std(SED_g)        
#     return fmr_fnr_at_tg, SER, MAPE, FDR,std_in_eer, std_in_SED
