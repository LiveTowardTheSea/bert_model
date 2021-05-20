class bert_config:
    def __init__(self):
        self.model_name = 'bert'
        self.d_model = 768
        self.lr = 0.0005
        self.bert_lr = 2e-5
        #self.regularization = 0.008
        self.lr_decay = 0.05
        self.batch_size = 8
        self.epoch_num = 5
        self.clip_value = 1.5
        self.decoder = 'crf'