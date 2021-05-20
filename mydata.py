# 读取文件
import pandas as pd
import codecs
import traceback
from vocab import *
import numpy as np
from data_iterator import *
from transformers import BertTokenizer

def get_data(input_path):
    """
    数据首先是存储在txt中，我们要把他放到目录下的csv文件中，利用pandas
    :param input_path: 文件路径
    :return: 返回一个DataFrame类型的数据，一列是字符列表，另一列是idx 列表
    """
    df_data = pd.DataFrame(columns=['sentence', 'label'])
    print('getting data: ',input_path)
    try:
        with codecs.open(input_path, 'r', 'utf-8') as f:
            string_list = []
            tag_list = []
            for line in f.readlines():
                if line not in ['\n', '\r\n']:
                    word_label = line.strip().split()
                    if len(word_label) >= 2:
                        string_list.append(word_label[0])
                        tag_list.append(word_label[1])
                else:
                    df_data.loc[df_data.index.size] = [string_list, tag_list]
                    string_list = []
                    tag_list = []
        return df_data
    except Exception as e:
        print(input_path, "file reading error")
        print(len(df_data))
        traceback.print_exc()
        exit(0)

class Data:
    def __init__(self, opt):
        self.opt = opt   # 命令行的一些操作参数
        self.char_vocab = BertTokenizer.from_pretrained("bert-base-chinese")
        self.tag_vocab = Vocab(vocab_type='tag')
        self.train_iter = None
        self.dev_iter = None
        self.test_iter = None

    # train_data 是一个,我们统一使用BIO格式, 这里 MSRA 使用了 BIO, 我们不再去判断别的，
    # 随着数据集的增加，这个肯定是要补充的啦
    def build_tag_vocab(self, train_data):
        """
        生成 tag 的 vocab,tag_vocab没有 unk_token,但是我给搞了一个pad,
        :param train_data:训练数据集，类型为 DataFrame,sentence 为 char list
        :return:
        """
        print('building tag vocab')
        for tag_list in train_data['label']:
            for tag in tag_list:
                self.tag_vocab.add(tag)
        self.tag_vocab.tag_add_pad()

    # 用于首次运行模型，将vocab, tag_vocab, pretrained_embedding保存
    # 通过如下代码我们假装自己准备好了 除了数据集之外的东西
    def build_vocab_pipeline(self):
        if self.opt.load_data is None:
            train_data = get_data(self.opt.train)
            self.build_tag_vocab(train_data)
            # 通过之前的步骤，在首次，我们已经加载好了词语向量。
            self.tag_vocab.save(self.opt.save_data + os.sep + 'tag_vocab')
        else:
            # 这种情况就是直接加载数据
            # 首先 加载两个词向量
            self.tag_vocab.load(self.opt.load_data + os.sep + 'tag_vocab')

    def build_data(self, batch_size):
        # 下面我们已经初始化了 词汇表 词向量
        self.build_vocab_pipeline()
        # 接下来，我们开始准备数据 然后初始化到当前的 data里面
        # 但是batch_size是存在模型的config里面的，记得传过来哦
        if self.opt.status.lower() == 'train':
            self.train_iter = data_iterator(self.opt.train, self.char_vocab, self.tag_vocab, batch_size)
            self.dev_iter = data_iterator(self.opt.dev, self.char_vocab, self.tag_vocab, batch_size)
        elif self.opt.status.lower() == 'test':
            self.test_iter = data_iterator(self.opt.test, self.char_vocab, self.tag_vocab, batch_size)
        elif self.opt.status.lower() == 'decode':
            pass
        else:
            print('input error:train or test or decode')



