from functions import *

def calc_fairness_metrics(list_of_distances, list_of_labels):
    fmr_fnr_at_tg = []
    global_h_full = 0
    fmr_p = 0.001
    G = len(list_of_labels) - 1
    for  i, (distances,labels) in enumerate(zip(list_of_distances, list_of_labels)):
        if i == 0:        
#             _, _, _, h_full, global_h,eer = validate_ijb(
#                         labels=labels_full,
#                         distances=distances_full,
#                         ts=0,
#                         fmr_p=0.001
#                     )
#         global_h_full = global_h
            global_h_full = 1217
        else:
            _, fmr, fnr, _, _, _ = calculate_FNR_FPR_thresh(global_h_full*0.001, distances, labels)
            fmr_fnr_at_tg.append([fmr, fnr])
    fmr_at_tg = [lst[0] for lst in fmr_fnr_at_tg if lst]
    fnr_at_tg = [lst[1] for lst in fmr_fnr_at_tg if lst]
    max_FMR = max(fmr_at_tg)
    min_FMR = min(fmr_at_tg)
    max_FNR = max(fnr_at_tg)
    min_FNR = min(fnr_at_tg)
    SER = max_FMR/min_FMR
    MAPE_SUM = 0
    for i in fnr_at_tg:
        MAPE_SUM += (i - fmr_p)/fmr_p 
    MAPE = (100/G) * MAPE_SUM
    FDR_FMR_diff = max_FMR - min_FMR
    FDR_FNR_diff = max_FNR - min_FNR
    alpha_FDR = 0.5
    FDR = 1 - (alpha_FDR * FDR_FMR_diff) - ((1 - alpha_FDR) * FDR_FNR_diff)
    
    
    sum_th_eer = 0
    eer_list = []
    for  i, (distances,labels) in enumerate(zip(list_of_distances, list_of_labels)):
        if i != 0:
            _, _, _, th_eer, _,eer = validate_ijb(
                        labels=labels,
                        distances=distances,
                        ts=0,
                        fmr_p=0.001
                    )
            sum_th_eer += th_eer
            eer_list.append(eer)
    avg_th_eer = sum_th_eer/G
    std_in_eer = np.std(eer_list)
    fmr_fnr_at_te = []
    for  i, (distances,labels) in enumerate(zip(list_of_distances, list_of_labels)):
        if i == 0:
            _, fpr_full, fnr_full, _, _, _ = calculate_FNR_FPR_thresh(avg_th_eer*0.001, distances, labels)
        if i != 0:
            _, fmr, fnr, _, _, _ = calculate_FNR_FPR_thresh(avg_th_eer*0.001, distances, labels)
            fmr_fnr_at_te.append([fmr, fnr])
    d_FMR_g = []
    d_FNMR_g = []
    fmr_at_te = [lst[0] for lst in fmr_fnr_at_te if lst]
    fnmr_at_te = [lst[1] for lst in fmr_fnr_at_te if lst]
    for i in fmr_at_te:
        d_FMR = abs(1-(i/fpr_full))
        d_FMR_g.append(d_FMR)
    for i in fnmr_at_te:
        d_FNMR = abs(1-(i/fpr_full))
        d_FNMR_g.append(d_FNMR)
    SED_g = [sum(i) for i in zip(d_FNMR_g, d_FMR_g)]  
    std_in_SED = np.std(SED_g)        
    return SER, MAPE, FDR,std_in_eer, std_in_SED