from model.Linear_Decoder import Softmax_decoder
import torch.nn as nn
from Encoder import *
from CRF_Decoder import *
from Linear_Decoder import *
class bert_model(nn.Module):
    def __init__(self,config,tag_num):
        super(bert_model, self).__init__()
        self.config = config
        self.encoder = Encoder()
        if self.config.decoder == 'crf':
            self.decoder = CRF_decoder(config.d_model, tag_num)
        elif self.config.decoder == 'softmax':
            self.decoder = Softmax_decoder(config.d_model,tag_num)

    def forward(self, src, trg, src_mask, trg_mask,use_gpu):
        # src: (batch_size,seq_len)
        # trg: (batch_size,seq_len)
        # src_mask (batch_size,seq_len)
        # 如果 return_atten 为 true 的话，返回每一层的attention，否则是一个空列表
        encoder_output = self.encoder(src, src_mask, use_gpu)
        encoder_output = encoder_output[:,1:-1,:]
        loss,path_ = self.decoder.loss(encoder_output, trg, trg_mask, use_gpu)
        return loss,path_