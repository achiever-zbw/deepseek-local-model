import requests
import json
import uuid
import chromadb
from chromadb.errors import NotFoundError 


config = {
    "filePath" : "D:/deepseek大模型开发/知识库.json"
}

def file_chunk_list(filePath):
    with open(filePath, encoding='utf-8') as f:
        data = json.load(f)  # 读取 json 格式数据

    # 将每个条目转换为一个字符串段
    chunk_list = []
    for item in data:
        category = item.get("分类", "")
        sub_algo = item.get("子算法", "")
        idea = item.get("思路", "")
        chunk = f"{category}\n子算法名字：{sub_algo}\n解题思路：{idea}"
        chunk_list.append(chunk)

    return chunk_list


def ollama_embedding_by_api(text):
    res = requests.post(
        url = "http://127.0.0.1:11434/api/embeddings",
        json = {
            "model" : "nomic-embed-text",
            "prompt" : text
        }
    )

    embedding = res.json()['embedding']
    return embedding

def ollama_generate_by_api(prompt):
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

def initial():
    client = chromadb.PersistentClient(path = "db/chroma_demo")  # 数据库
    try:
        client.delete_collection(name="collection_v1")
    except NotFoundError:
        print("集合 collection_v1 不存在，无需删除。")
    collection = client.get_or_create_collection(name = "collection_v1")  # 集合

    # 构造数据
    documents = file_chunk_list(filePath = config["filePath"])
    ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    embeddings = [ollama_embedding_by_api(text) for text in documents]
    # 插入数据
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

def run():
    # 关键字搜索
    qs = "动态规划的解题思路"
    qs_embedding = ollama_embedding_by_api(qs)  # 对关键字进行向量化
    client = chromadb.PersistentClient(path = "db/chroma_demo")
    collection = client.get_collection(name = "collection_v1")
    res = collection.query(query_embeddings = [qs_embedding] ,n_results= 3)

    result = res["documents"][0]
    context = "\n".join(result)


    prompt = f"""你是一个算法工程师，任务是根据参考信息回答用户问题，如果参考信息不足以回答用户问题，请回复不知道，不要去杜撰不实信息，
                参考信息:{context} ，来回答问题{qs}
            """
        # 调用大模型生成回答
    answer = ollama_generate_by_api(prompt)
    print("回答：", answer)


if __name__ == "__main__":
    initial()
    run()