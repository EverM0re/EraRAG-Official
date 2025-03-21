import pandas as pd
from torch.utils.data import Dataset
import os

class RAGQueryDataset(Dataset):
    def __init__(self,data_dir):
        super().__init__()
      
        self.corpus_path = os.path.join(data_dir, "Corpus.json")
        self.qa_path = os.path.join(data_dir, "Question.json")
        self.dataset = pd.read_json(self.qa_path, lines=True, orient="records")

    def get_corpus(self):
        corpus = pd.read_json(self.corpus_path, lines=True)
        corpus_list = []
        for i in range(len(corpus)):
            corpus_list.append(
                {
                    "title": corpus.iloc[i]["title"],
                    "content": corpus.iloc[i]["context"],
                    "doc_id": i,
                }
            )
        return corpus_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset.iloc[idx]["question"]
        answer = self.dataset.iloc[idx]["answer"]
        # other_attrs = self.dataset.iloc[idx].drop(["answer", "question"])
        other_attrs = self.dataset.iloc[idx].drop(["answer", "question"])
        return {"id": idx, "question": question, "answer": answer, **other_attrs}


if __name__ == "__main__":
    corpus_path = "tmp.json"
    qa_path = "tmp.json"
    query_dataset = RAGQueryDataset(qa_path=qa_path, corpus_path=corpus_path)
    corpus = query_dataset.get_corpus()
    print(corpus[0])
    print(query_dataset[0])
