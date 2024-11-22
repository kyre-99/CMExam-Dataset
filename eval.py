import pandas as pd
import re
ans_data = pd.read_csv("模拟3.csv")
predictions = []
ground_truths = []
breakpoint()
for i in range(400):
    ans_data.iloc[i]["Answer"]
    predictions.append(ans_data.iloc[i]["Answer"])
    ground_truths.append(
        re.sub("[\[\]\"\']", "", ans_data.iloc[i]["llm"]))


def calculate_metrics(predictions, ground_truths):
    # 初始化TP, FP, FN
    tp, fp, fn = 0, 0, 0
    if len(predictions) != len(ground_truths):
        raise ValueError(
            "The lengths of the prediction set and ground truth set are not equal."
        )
    for pred, gt in zip(predictions, ground_truths):
        # 将预测和ground truth转为集合，方便操作
        pred_set = set(pred)
        gt_set = set(gt)

        # 计算TP: 在ground truth中被正确预测的数量
        true_positives = len(pred_set & gt_set)
        tp += true_positives

        # 计算FP: 预测结果中多出来不属于ground truth的部分
        false_positives = len(pred_set - gt_set)
        fp += false_positives

        # 计算FN: ground truth中模型未预测到的部分
        false_negatives = len(gt_set - pred_set)
        fn += false_negatives

    # 计算Micro Precision, Recall和F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1


print(calculate_metrics(predictions, ground_truths))
