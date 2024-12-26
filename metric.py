from rouge_chinese import Rouge
import jieba

def evaluate_with_rouge(reference_list, generated_list):
    # 创建ROUGE对象
    rouge = Rouge()

    # 存储ROUGE评分的列表
    rouge1_scores = []
    rougeL_scores = []

    # 遍历每一对字符串，计算ROUGE评分
    for reference, generated in zip(reference_list, generated_list):
        reference = ' '.join(jieba.cut(reference))
        generated = ' '.join(jieba.cut(generated))
        scores = rouge.get_scores(reference, generated)
        # 将每个ROUGE评分的F1值添加到列表中
        rouge1_scores.append(scores[0]['rouge-1']["f"])
        rougeL_scores.append(scores[0]['rouge-l']["f"])

    # 计算ROUGE-1和ROUGE-L的平均F1值
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

    # 返回平均分数
    return avg_rouge1, avg_rougeL

