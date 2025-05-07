"""视觉版RAG从处理"""
import requests
import clip
import torch
from PIL import Image
import json
import uuid
import numpy as np
import chromadb
from posthog.exception_utils import epoch

clip_model , preprocess = clip.load("ViT-B/32")

config = {
    "knowledge_path" : "D:/deepseek大模型开发/Knowledge.json",
    "Ask_image_path" : "D:/大模型PDF数据源/20220512水源工程图片/1.png"
}

def get_image_embedding(image_path):
    """提取图片向量并返回原始数据类型
    """
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        image_features = clip_model.encode_image(image) # 提取图片向量
    return image_features[0].cpu().numpy().tolist() # 列表形式

def file_chunk_list(knowledge_path):
    """读取知识库的内容
    :param knowledge_path: 知识库地址
    :return: list
    """
    with open(knowledge_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunk_list = []
    for chunk in data :
        image_path = chunk["img_path"]
        description = chunk["description"]
        imageEmbedding = get_image_embedding(image_path) # 向量化的图片
        epoch = {
            "id" : str(uuid.uuid4()), # 每个数据生成对应的标识符
            "img_path" : image_path,
            "description" : description,
            "embedding" : imageEmbedding
        }
        chunk_list.append(epoch)

    return chunk_list

def initial() :
    """
    初始化图像向量数据库
    :return: None
    """
    client = chromadb.PersistentClient(path = "db/Image_chroma") # 数据库
    collection = client.get_or_create_collection(name = "Image_collection_v1") # 创建集合(表)

    chunks = file_chunk_list(config["knowledge_path"])
    for chunk in chunks :
        ids = [chunk["id"]]
        descriptions = [chunk["description"]]
        embeddings = [chunk["embedding"]]

        collection.add(ids = ids, documents = descriptions, embeddings = embeddings) # 搭建好了数据库

def ollama_generate_by_api(prompt):
    """
    调用 Deepseek-r1 模型
    :param prompt: 输入的内容
    :return: text(回答)
    """
    response = requests.post(
        url = "http://127.0.0.1:11434/api/generate",
        json = {
            "model" : "deepseek-r1:7b",
            "prompt" : prompt,
            "stream" : False,
            "temperature" : 0.1
        }
    )

    response = response.json()["response"]
    return response

def run():
    """
    进行向量检索并生成回答
    :return: None
    """
    qs = "请对图片进行描述分析"
    client = chromadb.PersistentClient(path = "db/Image_chroma")
    collection = client.get_or_create_collection(name = "Image_collection_v1")

    qs_image_embedding = get_image_embedding(config["Ask_image_path"]) # 对要进行询问的图片进行向量化
    res = collection.query(query_embeddings = [qs_image_embedding] , n_results = 2)
    print("匹配到的参考描述:", res["documents"][0])  # 加上这个打印
    res = res["documents"][0]
    inputImage = '\n'.join(res)

    prompt = f"""你是一个地表图像分析专家，任务是根据以下参考图片描述并分析信息来回答用户问题，如果参考信息不足，请回复“不知道” ， 不要随意编造
                参考信息:{inputImage} , 问题 : {qs}"""

    answer = ollama_generate_by_api(prompt)
    print(answer)

if __name__ == "__main__":
    initial()
    run()






