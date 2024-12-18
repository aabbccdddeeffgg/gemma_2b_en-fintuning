import os
import json
import keras
import keras_nlp
import re
from metric import evaluate_with_rouge

os.environ["KAGGLE_USERNAME"] = "wyq1234597"
os.environ["KAGGLE_KEY"] = "b3439dbc7b41a30c50a40b6b44613c06"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

# 加载训练集
train_data = []
with open("train.json") as file:
    for line in file:
        features = json.loads(line)
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        train_data.append(template.format(**features))
train_data = train_data[:100]

# 加载测试集，分别存为指令以及响应
test_data_instruction = []
test_data_response = []
with open("test.json") as file:
    for line in file:
        features = json.loads(line)
        instruction = features["instruction"]
        response = features["response"]
        test_data_instruction.append(instruction)
        test_data_response.append(response)

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()

print("Before fine-tuning:\n\n")

# 示例
prompt = template.format(
    instruction="中国历史上第一次大一统，请问这是哪一个朝代？",
    response="",
)
generate_data = gemma_lm.generate(prompt, max_length=256)
match = re.search(r"Response:\s*(.*?)(?=(\nInstruction:|\n$))", generate_data, re.DOTALL)
print(match.group(1).strip()) # 打印第一个response后的内容，strip去掉前后空格

# 加载模型预测，得出初始模型在测试集上的rouge分数
model_response = []
for data in test_data_instruction:
    prompt = template.format(
        instruction=data,
        response="",
    )
    generate_data = gemma_lm.generate(prompt, max_length=256)
    match = re.search(r"Response:\s*(.*?)(?=(\nInstruction:|\n$))", generate_data, re.DOTALL)
    model_response.append(match.group(1).strip())

rouge_scores = evaluate_with_rouge(test_data_response, model_response)
print("Rouge-1: %.4f", rouge_scores[0])
print("Rouge-L: %.4f", rouge_scores[1])

# 微调，启用Lora
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()

# 设置超参数
gemma_lm.preprocessor.sequence_length = 256
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

# 微调
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(train_data, epochs=1, batch_size=1)
gemma_lm.save('finetuned_model.keras')


print("After fine-tuning:\n")

# 示例对比
prompt = template.format(
    instruction="中国历史上第一次大一统，请问这是哪一个朝代？",
    response="",
)
generate_data = gemma_lm.generate(prompt, max_length=256)
match = re.search(r"Response:\s*(.*?)(?=(\nInstruction:|\n$))", generate_data, re.DOTALL)
print(match.group(1).strip())

# 微调后模型的输出和测试集的参考输出对比后的分数
# 加载微调后的模型
gemma_lm = keras.models.load_model('finetuned_model.keras', custom_objects={'GemmaCausalLM': keras_nlp.models.GemmaCausalLM})

# 加载模型预测，得出微调后模型在测试集上的rouge分数
model_response_finetuned = []
for data in test_data_instruction:
    prompt = template.format(
        instruction=data,
        response="",
    )
    generate_data = gemma_lm.generate(prompt, max_length=256)
    match = re.search(r"Response:\s*(.*?)(?=(\nInstruction:|\n$))", generate_data, re.DOTALL)
    model_response_finetuned.append(match.group(1).strip())

# 计算ROUGE分数
rouge_scores_finetuned = evaluate_with_rouge(test_data_response, model_response_finetuned)

# 打印微调后ROUGE评分
print("After fine-tuning:")
print("Rouge-1: %.4f" % rouge_scores_finetuned[0])
print("Rouge-L: %.4f" % rouge_scores_finetuned[1])



