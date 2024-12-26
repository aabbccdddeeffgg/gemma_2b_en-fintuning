import os
import json
import keras
import keras_nlp
import re
import pandas as pd
import shutil

os.environ["KAGGLE_USERNAME"] = "wyq1234597"
os.environ["KAGGLE_KEY"] = "b3439dbc7b41a30c50a40b6b44613c06"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

template = "问题:\n{instruction}\n\n回答:\n{response}"

# 加载训练集(json)格式(问答)
train_data = []
with open("/kaggle/input/d/wyq1234597/data-of-wyq1234/zh_seed_tasks.jsonl") as tarin_file:
    for line in tarin_file:
        line = line.strip()  # 去掉首尾空白符
        if not line:  # 跳过空行
            continue
        try:
            features = json.loads(line)
            # template_qa = "问题:\n{instruction}\n\n回答:\n{response}"
            instruction = features["instruction"]
            response = features["response"]
            train_data.append(template.format(instruction=instruction, response=response))
        except json.JSONDecodeError as e:
            print(f"跳过格式错误的行: {line} 错误: {e}")

# 加载测试集(json)，分别存为指令以及响应
test_data_instruction = []
test_data_response = []
with open("/kaggle/input/d/wyq1234597/data-of-wyq1234/test.jsonl") as test_file:
    for line in test_file:
        features = json.loads(line)
        instruction = features["instruction"]
        response = features["response"]
        test_data_instruction.append(instruction)
        test_data_response.append(response)

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()

print("Before fine-tuning:\n\n")

# 示例
with open("/kaggle/input/d/wyq1234597/data-of-wyq1234/generate_text.jsonl", "r") as generate_file:
    for line in generate_file:
        line = line.strip()  # 去掉首尾空白符
        if not line:  # 跳过空行
            continue
        try:
            features = json.loads(line)
            instruction = features["instruction"]
            response = features["response"]
            prompt = template.format(
                instruction=instruction,
                response="",
            )
            generate_data = gemma_lm.generate(prompt, max_length=512)
            match = re.search(r"回答：\s*(.*?)(?=\n问题：|$)", generate_data, re.DOTALL)
            s = str(match.group(1).strip()) if match else str(generate_data)
            print('reference answer:')
            print(response)
            print('generate data:')  # 打印第一个response后的内容，strip去掉前后空格
            print(s if match else generate_data)
        except json.JSONDecodeError as e:
            print(f"跳过格式错误的行: {line} 错误: {e}")

# 加载模型预测，得出初始模型在测试集上的rouge分数
model_response = []
for data in test_data_instruction:
    prompt = template.format(
        instruction=data,
        response="",
    )
    generate_data = gemma_lm.generate(prompt, max_length=512)
    match = re.search(r"回答：\s*(.*?)(?=\n问题：|$)", generate_data, re.DOTALL)
    s = str(match.group(1).strip()) if match else str(generate_data)
    model_response.append(s)

rouge_scores = evaluate_with_rouge(test_data_response, model_response)
print("Rouge-1: %0.4f" % rouge_scores[0])
print("Rouge-L: %0.4f" % rouge_scores[1])

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
gemma_lm.fit(train_data, epochs=5, batch_size=1)  # batch_size太大GPU内存会不够
gemma_lm.save('finetuned_model.keras')
shutil.move('finetuned_model.keras', '/kaggle/working/finetuned_model.keras')
gemma_lm.backbone.save_lora_weights("model.lora.h5")
shutil.move('model.lora.h5', '/kaggle/working/model.lora.h5')
print("Lora weights saved.")

print("After fine-tuning:\n")

# 示例对比
with open("/kaggle/input/d/wyq1234597/data-of-wyq1234/generate_text.jsonl", "r") as generate_file:
    for line in generate_file:
        line = line.strip()  # 去掉首尾空白符
        if not line:  # 跳过空行
            continue
        try:
            features = json.loads(line)
            instruction = features["instruction"]
            response = features["response"]
            prompt = template.format(
                instruction=instruction,
                response="",
            )
            generate_data = gemma_lm.generate(prompt, max_length=512)
            match = re.search(r"回答：\s*(.*?)(?=\n问题：|$)", generate_data, re.DOTALL)
            s = str(match.group(1).strip()) if match else str(generate_data)
            print('reference answer:')
            print(response)
            print('generate data:')  # 打印第一个response后的内容，strip去掉前后空格
            print(s if match else generate_data)
        except json.JSONDecodeError as e:
            print(f"跳过格式错误的行: {line} 错误: {e}")

# 微调后模型的输出和测试集的参考输出对比后的分数

# 加载模型预测，得出微调后模型在测试集上的rouge分数
model_response_finetune = []
for data in test_data_instruction:
    prompt = template.format(
        instruction=data,
        response="",
    )
    generate_data = gemma_lm.generate(prompt, max_length=512)
    match = re.search(r"回答：\s*(.*?)(?=\n问题：|$)", generate_data, re.DOTALL)
    s = str(match.group(1).strip()) if match else str(generate_data)
    model_response_finetune.append(s)
    # model_response_finetune.append(generate_data)

# 计算ROUGE分数
rouge_scores_finetune = evaluate_with_rouge(test_data_response, model_response_finetune)

# 打印微调后ROUGE评分
print("After fine-tuning:")
print("Rouge-1: %.4f" % rouge_scores_finetune[0])
print("Rouge-L: %.4f" % rouge_scores_finetune[1])
