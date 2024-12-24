import json

from fontTools.ttLib.tables.ttProgram import instructions

from metric import evaluate_with_rouge
import re
train_data = []
with open("data/zh_seed_tasks.jsonl", "r", encoding='utf-8') as file:
    for line in file:
        features = json.loads(line)
        response = features["response"]
        train_data.append(response)
train_data = train_data[:5]
x = ["中国的首都是北京"]
test_data = ["北京是中国的首都"]

score = evaluate_with_rouge(x, test_data)
print(type(train_data[0]))
print(train_data)
print(score)


x = [1,2,3,4,5,6,7,8,9,10]
print(x[:2])