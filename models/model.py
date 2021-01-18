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
from transformers import AutoModel

# Create the BertClassfier class
class ModelCLR(nn.Module):
    def __init__(self, res_base_model, bert_base_model, out_dim, freeze_layers, do_lower_case):
        super(ModelCLR, self).__init__()
        #init BERT
        self.bert_model = self._get_bert_basemodel(bert_base_model,freeze_layers)
        # projection MLP for BERT model
        self.bert_l1 = nn.Linear(768, 768) #768 is the size of the BERT embbedings
        self.bert_l2 = nn.Linear(768, out_dim) #768 is the size of the BERT embbedings

        # init Resnet
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}
        resnet = self._get_res_basemodel(res_base_model)
        num_ftrs = resnet.fc.in_features
        self.res_features = nn.Sequential(*list(resnet.children())[:-1])
        # projection MLP for ResNet Model
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, out_dim)

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("Image feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def image_encoder(self, xis):
        h = self.res_features(xis)
        h = h.squeeze()

        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)

        return h, x

    def text_encoder(self, encoded_inputs):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fontes:
        - https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        - Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        https://www.sbert.net
        """
        outputs = self.bert_model(**encoded_inputs)
        
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask']).half()
            x = self.bert_l1(sentence_embeddings)
            x = F.relu(x)
            out_emb = self.bert_l2(x)

        return out_emb

    def forward(self, xis, encoded_inputs):

        h, zis = self.image_encoder(xis)

        zls = self.text_encoder(encoded_inputs)

        return zis, zls