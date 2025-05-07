import os
import json


def DataProcessing(input_path , output_path ,description_text):
    """
    :param input_path: 数据库中要输入的图片的文件夹路径，包含多张图片，对应相同的描述
    :param output_path: json 的输出路径
    :param description_text: 统一的图片描述
    :return : None
    """
    data = []

    for imgPath in os.listdir(input_path):
        fullPath = os.path.join(input_path, imgPath)
        entry = {
            "img_path" : fullPath,
            "description" : description_text
        }
        data.append(entry)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

description_text = "目前掌子面桩号 K0+956 。探测时掌子面整体潮湿，岩性为灰岩，围岩完程度为完整性差,较破碎，掌子面底板处有出水，掌子面发育有多组节理。左侧掌子面处有岩溶情况，综合地质情况推断，该区域地下水受季节性影响大。"
input_path = "D:/大模型PDF数据源/20220512水源工程图片"
output_path = "D:/deepseek大模型开发/Knowledge.json"

if __name__ == "__main__":
    DataProcessing(input_path = input_path, output_path = output_path, description_text = description_text)