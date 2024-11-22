# 读取数据 csv
# 构建向量数据库
# Prompt ： 反思 + 多次 + 检索增强

import faiss
import numpy as np
from typing import Dict
import pandas as pd
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_ollama import OllamaEmbeddings
from typing import List
import re


class RAG:
    def __init__(self, text_book, model):
        self.text_book = text_book
        self.embed_model = model
        self.index = faiss.IndexFlatIP(768)
        self.__load_textbook()

    def __load_textbook(self) -> None:
        for i in tqdm.tqdm(range(len(self.text_book))):
            self.index.add(self._embed_text(self.text_book.iloc[i]['over']))
            if i == 500:
                break
        print("知识库转载完毕！")

    def search(
        self,
        query: str,
        k=2
    ) -> List[str]:
        query_vector = self._embed_text(query)
        D, I = self.index.search(query_vector, k=2)
        return [self.text_book.iloc[i]['over'] for i in I[0]]

    def _embed_text(self, text):
        return np.array(self.embed_model.embed_documents(
            text)).astype("float32")


class Student:
    def __init__(self, system_prompt, model_path, rag: RAG):
        self.model_path = model_path
        self.messages = [{"role": "system", "content": system_prompt}]
        self.rag = rag
        self.is_rag = 0

    def load_model(self, device):
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.device = device

    def answer(self, instruction: str):
        self.messages.append({"role": "user", "content": instruction})
        try:
            input = self.tokenizer.apply_chat_template(
                self.messages, return_tensors="pt")
        except Exception:
            conversation_history = "\n".join([
                f"{msg['role']}:\n{msg['content']}"
                for msg in self.messages
            ])
            input = self.tokenizer.apply_chat_template(
                conversation_history, return_tensors="pt", chat_template=conversation_history)
        ret = self.llm.generate(
            input.to(self.device), max_new_tokens=100, temperature=0.7, top_k=50, do_sample=True, top_p=0.9, repetition_penalty=1.2
        )
        output = self.tokenizer.decode(
            ret[0][len(input[0]):], skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": output})
        return output

    def test(self, question: Dict):
        instruction = f'不定向选择题题干为：\n\n{question["question"]}?选项为\n\n{question["options"]}。对于你认为可以判断准确的题目，只需要回复正确的答案选项，严格以JSON格式返回，格式如下：单选题返回{{"type": "single", "answer": ["选项"]}}，多选题返回`{{"type": "multiple", "answers": ["选项1", "选项2"]}}`。如果你对题目不熟悉，可以选择阅读参考书，回复`{{"type": "RAG", "status": "需要参考"}}`。'
        output = self.answer(instruction)
        if "rag" in output or "RAG" in output:
            self.is_rag += 1
            knowledge = self.read_book(str(question))
            rag_instruction = f"""你可以参考的资料为{knowledge}\n\n请一步步思考，尽可能保证答案的准确性。只需要回复正确的答案选项，且请返回json格式，格式如下：单选题返回{{"type": "single", "answer": ["选项"]}}，多选题返回`{{"type": "multiple", "answers": ["选项1", "选项2"]}}`"""
            output = self.answer(rag_instruction)
            self.messages = self.messages[:1]
            return output
        else:
            self.messages = self.messages[:1]
            return output

    def read_book(self, question: str):
        return "\n".join(self.rag.search(query=question))


def get_raw_text():
    # 'Question', 'Options', 'Answer', 'Explanation'
    train_data = pd.read_csv("train.csv")
    # Question', 'Options', 'Answer', 'Explanation'
    test_data = pd.read_csv("val.csv")
    train_data["over"] = train_data['Question']+":\n"+train_data['Options'] + \
        ":\n"+"答案为："+train_data['Answer']+"\n解析："+train_data['Explanation']
    train_data.drop(
        columns=['Question', 'Options', 'Answer', 'Explanation'], inplace=True)
    return train_data, test_data


if __name__ == "__main__":
    device = "cuda:0"
    system_prompt = "你作为一个医学专家，正在参加中国国家医师资格考试,考试形式为不定向选择题，请充分利用你的专业知识，进行作答。"
    model_path = "/disk3/wzh/LLM_Model/models--prithivMLmods--Llama-Doctor-3.2-3B-Instruct/snapshots/b9faea0637d78f643edfcdaf9d3b407333963171"
    train_data, test_data = get_raw_text()
    rag = RAG(text_book=train_data, model=OllamaEmbeddings(
        model="nomic-embed-text"))
    student = Student(system_prompt, model_path, rag)
    student.load_model(device)
    test_data['llm'] = ""
    for i in tqdm.tqdm(range(len(test_data))):
        question = {}
        question["question"] = test_data.iloc[i]["Question"]
        question["options"] = test_data.iloc[i]["Options"]
        output = student.test(question)
        if re.search(r'(\[.*?\])', output) is not None:
            test_data.loc[i, 'llm'] = re.search(
                r'(\[.*?\])', output).group(1)
        elif re.search(r'"answer":\s*"([A-Z]+)"', output) is not None:
            test_data.loc[i, 'llm'] = re.search(
                r'"answer":\s*"([A-Z]+)"', output).group(1)
        elif re.search(r'([A-Z])', output) is not None:
            test_data.loc[i, 'llm'] = re.search(r'([A-Z])', output).group(1)
        else:
            test_data.loc[i, 'llm'] = '["C"]'
    test_data.to_csv("模拟2.csv", index=False, encoding='utf-8')
