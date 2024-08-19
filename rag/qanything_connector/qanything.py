import os
import requests
import json
import sys
import time

KB_id = 'KB0677632d6bdf44a9b379fe386f14bdd7'
# 上传文件
# url = "http://localhost:8777/api/local_doc_qa/upload_files"
# folder_path = "/home/hjl/Chat_BDA/config/knowledge"  # 文件所在文件夹，注意是文件夹！！
# data = {
#     "user_id": "zzp",
#     "kb_id": KB_id,
# 	"mode": "soft"
# }
# files = []
# for root, dirs, file_names in os.walk(folder_path):
#     for file_name in file_names:    
#         print(file_name)
#         file_path = os.path.join(root, file_name)
#         files.append(("files", open(file_path, "rb")))
# response = requests.post(url, files=files, data=data)
# print(response.text)

# 查看知识库
# url = "http://localhost:8777/api/local_doc_qa/list_knowledge_base"
# headers = {
#     "Content-Type": "application/json"
# }
# data = {
#     "user_id": "zzp"
# }
# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(response.status_code)
# print(response.text)


# 查看文件
# url = "http://localhost:8777/api/local_doc_qa/list_files"
# headers = {
#     "Content-Type": "application/json"
# }
# data = {
# 	"user_id": "zzp",  
# 	"kb_id": KB_id
# }
# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(response.status_code)
# print(response.text)


# 问答
# def send_request(question):
#     url = 'http://localhost:8777/api/local_doc_qa/local_doc_chat'
#     headers = {
#         'content-type': 'application/json'
#     }
#     data = {
#         "user_id": "zzp",
#         "kb_ids": [KB_id],
#         "question": question,
#     }
#     try:
#         response = requests.post(url=url, headers=headers, json=data, timeout=60)
#         res = response.json()
#         return res
#     except Exception as e:
#         print(f"请求发送失败: {e}")

# query = '哪种孔隙率分布有利于提高电池高倍率下的性能？造成不同性能的原因是什么？'
# r = send_request(query)
# print(r)