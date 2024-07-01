import os
from functools import lru_cache

import torch
import torch.nn.functional as F

from dudes import utils
from llm.query_score_models.t5 import QueryScoreT5


class LLMQuerySelector:
    def __init__(self, model_path, input_target_length=512):
        self.model = QueryScoreT5.load_from_checkpoint(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.input_target_length = input_target_length

    def _input_string(self, question, query, dudes, numresults):
        return """Question: {question}
        Number of results: {numres1}
        SPARQL: 
        {sparql1} 
        DUDES: 
        {dudes1}""".format(
            question=question,
            numres1=numresults,
            sparql1=utils.replace_namespaces_dirty(utils.remove_prefix(query)),
            dudes1=str(dudes),
        )

    def compare_queries(self, question, query1, query2, dudes1, dudes2, numresults1, numresults2):
        self.model.eval()
        with torch.no_grad():
            de = {
                "input1": self.model.tokenizer.encode(self._input_string(question=question, query=query1, dudes=dudes1, numresults=numresults1), max_length=512, truncation=True),
                "input2": self.model.tokenizer.encode(self._input_string(question=question, query=query2, dudes=dudes2, numresults=numresults2), max_length=512, truncation=True),
            }
            input1: torch.Tensor = torch.LongTensor(de["input1"])
            input1 = F.pad(input1, (0, self.input_target_length - len(de["input1"])), value=self.model.tokenizer.pad_token_id)
            input2: torch.Tensor = torch.LongTensor(de["input2"])
            input2 = F.pad(input2, (0, self.input_target_length - len(de["input2"])), value=self.model.tokenizer.pad_token_id)

            model_input = torch.cat((input1, input2)).unsqueeze(0)
            #model_input2 = torch.cat((input2, input1)).unsqueeze(0)
            model_input = model_input.to(self.device)
            res = self.model(model_input)
            res = res.cpu()
            #res2 = self.model(model_input2)
            #print(res)
            #print(res2)
            #return F.sigmoid(res).tolist()
            return res.tolist()[0]


class MultiLLMQuerySelector:
    def __init__(self, query_score_models):
        self.query_score_models = query_score_models

    @classmethod
    def from_paths(cls, paths):
        assert all([os.path.isfile(qsp) for qsp in paths])
        return cls([LLMQuerySelector(model_path=p) for p in paths])


    @lru_cache(maxsize=512)
    def compare_queries(self, question, query1, query2, dudes1, dudes2, numresults1, numresults2):
        res = []
        for qs in self.query_score_models:
            qsres = qs.compare_queries(question, query1, query2, dudes1, dudes2, numresults1, numresults2)
            res.append(qsres)
        return res