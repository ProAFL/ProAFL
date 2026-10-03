from scipy import stats
from cliffs_delta import cliffs_delta

def wtl(our_data_list,baseline_data_list,sign=1):
    # data list 是正向指标
    p_value = stats.wilcoxon(our_data_list, baseline_data_list).pvalue
    sorted_our_data_list = sorted(our_data_list)
    sorted_baseline_data_list = sorted(baseline_data_list)
    delta,info = cliffs_delta(sorted_our_data_list, sorted_baseline_data_list)
    if sign == 1:
        # 正向指标 
        if p_value < 0.05 and delta > 0.147:
            # ours和baseline有差异，且我们的值偏大
            return 'W'
        elif p_value < 0.05 and delta < -0.147:
            # ours和baseline有差异，且我们的值偏小
            return 'L'
        else:
            return 'T'
    else:
        # 负向指标
        if p_value < 0.05 and delta > 0.147:
            # ours和baseline有差异，且我们的值偏大
            return 'L'
        elif p_value < 0.05 and delta < -0.147:
            # ours和baseline有差异，且我们的值偏小
            return 'W'
        else:
            return 'T'
        
if __name__ == "__main__":
    our_data_list = [0.773,0.685,0.683,0.686,0.686,0.686,0.685,0.682,0.685,0.686]
    datactive_list = [0.674,0.668,0.572,0.64,0.639,0.638,0.645,0.646,0.645,0.641]
    print(wtl(our_data_list,datactive_list,1))