import joblib



def check_rank_1():
    rank_1_path = "/data/mml/data_debugging_data/Results/ours/VOC2012/YOLOv7/exp_01/rank/rank.joblib"
    rank_2_path = "/data/mml/data_debugging_data/Results/ours/VOC2012/YOLOv7/exp_03/rank/rank.joblib"
    rank_1 = joblib.load(rank_1_path)
    rank_2 = joblib.load(rank_2_path)

    consis_flag = 1
    for e1,e2 in zip(rank_1,rank_2):
        if e1 != e2:
            consis_flag = 0
            break
    if consis_flag == 0:
        print("排序不一致")
    else:
        print("排序是完全一致的")
    return consis_flag


def main():
    check_rank_1()



if __name__ == "__main__":
    main()