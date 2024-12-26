import keras_nlp
import keras
import re
# 加载基础模型
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

# 启用 Lora 参数支持
gemma_lm.backbone.enable_lora(rank=4)

# 加载微调后的 Lora 权重
lora_weights_path = "/kaggle/input/fintuned-lora/keras/default/1/model.lora (1).h5"
gemma_lm.backbone.load_lora_weights(lora_weights_path)

# 确保加载的权重正确
gemma_lm.summary()

# 使用加载的模型进行推断
template = "问题:\n{instruction}\n\n回答:\n{response}"
prompt = template.format(
    instruction="创作一首五言律诗",
    response="",
)
generate_data = gemma_lm.generate(prompt, max_length=256)
match = re.search(r"Response:\s*(.*?)(?=(\nInstruction:|$))", generate_data, re.DOTALL)
print(match.group(1).strip() if match else str(generate_data))