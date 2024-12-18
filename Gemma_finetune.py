
"""
需要的函数库
pip install "tensorflow[and-cuda]"
pip install -U keras-nlp
pip install -U keras
pip install tensorrt
"""



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

# 训练集
train_data = []
with open("train.json") as file:
    for line in file:
        features = json.loads(line)
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        train_data.append(template.format(**features))
train_data = train_data[:100]

test_data = []
with open("test.json") as file:
    for line in file:
        features = json.loads(line)
        response = features["response"]
        test_data.append(response)

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()

print("Before fine-tuning:\n\n")

prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))

prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))

gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()

gemma_lm.preprocessor.sequence_length = 512
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(train_data, epochs=1, batch_size=1)

gemma_lm.save('finetuned_model.keras')

print("After fine-tuning:\n")

prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
generate_data = gemma_lm.generate(prompt, max_length=256)
match = re.search(r"Response:\s*(.*?)(?=(\nInstruction:|\n$))", generate_data, re.DOTALL)
print(match.group(1).strip())  # 打印第一个response后的内容，strip去掉前后空格

prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))

