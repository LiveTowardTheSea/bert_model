from transformers import BertTokenizer
import torch.nn as nn
import torch
import transformers
from transformers import BertModel
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese",add_pooling_layer=False)
    
    def forward(self,sentence_tensor,sentence_mask,use_gpu):
        encoded_result = self.bert(input_ids=sentence_tensor,attention_mask=sentence_mask,
                                    output_attentions=False,output_hidden_states=False)
        return encoded_result[0]


