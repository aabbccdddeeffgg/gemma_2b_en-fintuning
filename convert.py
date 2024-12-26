import pandas as pd
from opencc import OpenCC

# 创建繁体转简体的转换器
t2s_converter = OpenCC('t2s')

# 读取CSV文件
df = pd.read_csv("D:\\课程资料及作业\\kaggle\\data\\Tang Poetry Dataset.csv", encoding="utf-8")

# 对指定列进行繁体字到简体字的转换
for index, row in df.iterrows():
    # 确保在转换之前，每个值都是字符串
    row['title'] = t2s_converter.convert(str(row['title']))
    row['author'] = t2s_converter.convert(str(row['author']))
    row['poetry'] = t2s_converter.convert(str(row['poetry']))

    # 将转换后的值更新回DataFrame
    df.at[index, 'title'] = row['title']
    df.at[index, 'author'] = row['author']
    df.at[index, 'poetry'] = row['poetry']

# 打印转换后的DataFrame
print(df)

# 保存转换后的CSV文件
df.to_csv("D:\\课程资料及作业\\kaggle\\data\\converted_file.csv", index=False, encoding="utf-8")
