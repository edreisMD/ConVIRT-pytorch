"""
Reference for BERT Sentence Embeddings method

@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",

"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from transformers import BertTokenizer, BertModel, BertConfig, BertForPreTraining

# Create the BertClassfier class
class BERTSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, freeze_layers, do_lower_case, truncation):
        super(BERTSimCLR, self).__init__()
        self.base_model = str(base_model)
        self.freeze_layers = freeze_layers
        self.bert_model = self._get_basemodel()
        # self.tokenizer = BertTokenizer.from_pretrained(base_model)

        # projection MLP
        self.l1 = nn.Linear(768, 768) #768 is the size of the BERT embbedings
        self.l2 = nn.Linear(768, out_dim) #768 is the size of the BERT embbedings

    def _get_basemodel(self):
        try:
            model = BertModel.from_pretrained(self.base_model)# return_dict=True)

            if self.freeze_layers is not None:
                for layer_idx in self.freeze_layers:
                    for param in list(model.encoder.layer[layer_idx].parameters()):
                        param.requires_grad = False

            print("Image feature extractor:", self.base_model)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        
    def forward(self, encoded_inputs):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fonte: https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        """
        outputs = self.bert_model(**encoded_inputs)

        # Max-Pooling para extrair as embeddings como no paper e no https://huggingface.co/sentence-transformers/bert-base-nli-max-tokens
        attention_mask = encoded_inputs['attention_mask']
        token_embeddings = outputs[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_over_time = torch.max(token_embeddings, 1)[0]

        # Mean over all layers - Option to be tested
        # sentence_embeddings = torch.mean(hidden_states[-1], dim=1).squeeze()

        x = self.l1(max_over_time)
        x = F.relu(x)
        x = self.l2(x)

        return x
