# coding: UTF-8
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from pytorch_pretrained import BertModel
from transformers import BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'Text_MLT_Causual_RNN_ATT_SSM'
        self.class_list = [0, 1, 2]
        self.train_path = dataset + '/data/train.txt'                                 # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                     # 验证集
        self.test_path = dataset + '/data/test.txt'                                   # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]               # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'         # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.4                                              # GRU随机失活
        self.dropout2 = 0.4                                             # 自注意力随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练，在这个项目中没用了
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 100                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 100                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率
        self.hidden_size = 128                                           # GRU隐藏层
        self.num_layers = 1                                             # GRU层数

        self.embed = 768                                                # bert输出的词向量维度
        self.bert_path = './chinese-roberta-wwm-ext'                    # 使用chinese-roberta-wwm-ext作为动态嵌入
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # 建立分词器
        self.sbert_path = './sbert-chinese-qmc-domain-v1'               # 使用sbert-chinese-qmc-domain-v1作为动态嵌入


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 冻结roberta模型除最后一层外的所有参数  
        for param in self.bert.parameters():  
            param.requires_grad = False  
        last_layer_modules = list(self.bert.encoder.layer[-1].parameters())  
        for param in last_layer_modules:  
            param.requires_grad = True

        self.sbert = BertModel.from_pretrained(config.sbert_path)
        # 冻结sentence-BERT模型的所有参数  
        for param in self.sbert.parameters():  
            param.requires_grad = False
            
        self.gru = nn.GRU(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        
        self.fc_sen = nn.Linear(config.hidden_size * 2, 64)
        self.fc_sen_out = nn.Linear(64, 2)
        self.fc = nn.Linear(config.hidden_size * 5, 128)
        self.fc_att1 = nn.Linear(config.hidden_size * 4, config.hidden_size * 1)
        self.fc3 = nn.Linear(128, config.num_classes) # 3 class number
        self.fc_cls = nn.Linear(768, config.hidden_size)
        self.fc_cls1 = nn.Linear(512, config.hidden_size)
        
        self.sen_embedding = nn.Linear(2, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size * 2)

        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)   # 自适应平均池化层
        self.max_pool = nn.AdaptiveMaxPool1d(1)   # 自适应最大池化层

        # 定义线性变换层，生成Query, Key, Value  
        self.query = nn.Linear(config.embed, config.hidden_size * 2)  
        self.key = nn.Linear(config.embed, config.hidden_size * 2)  
        self.value = nn.Linear(config.embed, config.hidden_size * 2)
        
        self.dropout = nn.Dropout(config.dropout2)
        self.hidden_size = config.hidden_size
        self.tokenizer = config.tokenizer

    def self_attention(self, input_data):
        '''单头注意力机制的实现'''
        # 生成Query, Key, Value  
        Q = self.dropout(self.query(input_data))  
        K = self.dropout(self.key(input_data))
        V = self.dropout(self.value(input_data))

        # 计算注意力权重  
        attention_weights = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_size ** 0.5)  
        attention_weights = F.softmax(attention_weights, dim=2)  
          
        # 应用注意力权重  
        self_attention_output = torch.bmm(attention_weights, V)
        self_attention_output = self.attention_layer_norm(self_attention_output)
        
        return self_attention_output

    def sentence_embed(self, sent1, sent2, sent_mask1, sent_mask2):
        '''句子匹配向量提取的准备步骤'''
        sep_token_id = self.tokenizer.sep_token_id
        sent_batch_size = sent1.shape[0]
        sep_tensor = torch.full((sent_batch_size, 1), sep_token_id, dtype=torch.long, device=sent1.device)
        sep_mask = torch.ones((sent_batch_size, 1), dtype=torch.long, device=sent_mask1.device)  # 假设 sep_tensor 对应的掩码值为 1
        
        input_sent_data = torch.cat((sent1, sep_tensor, sent2[:, 1:]), dim=1)
        sent_mask = torch.cat((sent_mask1, sep_mask, sent_mask2[:, 1:]), dim=1)

        return input_sent_data, sent_mask

    def forward(self, x, x_1, a, b):
        input_data = torch.cat((x, x_1), dim=0)
        mask = torch.cat((a, b), dim=0)

        # 词嵌入：roberta-encoder的最后一个隐藏状态
        encoder_out, _ = self.bert(input_data, attention_mask=mask, output_all_encoded_layers=False)
        # 句子语义匹配向量：两个句子拼接后的[cls]token向量(使用sentence-bent做encoder)
        input_data1, mask1 = self.sentence_embed(x, x_1, a, b)
        encoder_out1, _ = self.sbert(input_data1, attention_mask=mask1, output_all_encoded_layers=False)
        cls_embeddings = encoder_out1[:, 0, :]
        semantic_matching_vector = F.gelu(self.fc_cls(cls_embeddings))

        # 1.主题内容编码——使用 单头self-Attention
        H = self.self_attention(encoder_out)
        raw_out, raw_out1 = H.chunk(2, dim=0)
        
        # 2.情感特征编码——使用 GRU + Attention
        H_sen, _ = self.gru(encoder_out)  # [batch_size, seq_len, hidden_size * num_direction]
        H_alpha = F.softmax(torch.matmul(H_sen, self.w), dim=1).unsqueeze(-1)
        H_sen = H_sen * H_alpha
        sen_mid = F.gelu(self.fc_sen(H_sen))
        sen_sum_out = self.fc_sen_out(sen_mid)
        sen_sum_out = self.max_pool(sen_sum_out.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1) 
        
        sen_out, sen_out1 = H_sen.chunk(2, dim=0)

        sen_gs_out = F.gumbel_softmax(sen_sum_out, hard=True)
        sen_out_gs, sen_out1_gs = self.layer_norm(self.sen_embedding(sen_gs_out)).chunk(2, dim=0)  # label embedding

        raw_out = torch.cat((raw_out, sen_out), dim=2)
        raw_out1 = torch.cat((raw_out1, sen_out1), dim=2)
        
        # 3.注意力机制：主题编码和情感编码进行交互注意力计算
        raw_out = self.fc_att1(raw_out)
        raw_out1 = self.fc_att1(raw_out1)
        
        raw_out_v = F.gelu(raw_out)
        raw_out1_v = F.gelu(raw_out1)
        
        attention_scores = torch.matmul(raw_out_v, raw_out1_v.transpose(1, 2))
        attention_weights = F.softmax(attention_scores, dim=-1)
        self.attention_weights1 = attention_weights
        attention_out = torch.matmul(attention_weights, raw_out1)
        attention_scores = torch.matmul(raw_out1_v, raw_out_v.transpose(1, 2))
        attention_weights = F.softmax(attention_scores, dim=-1)
        self.attention_weights2 = attention_weights
        attention_out1 = torch.matmul(attention_weights, raw_out)
        
        # 4.预测模块：拼接Ⅰ.上一步注意力机制输出的向量 Ⅱ.情感标签向量 Ⅲ.句子语义匹配向量
        attention_out = torch.cat((attention_out[:,-1,:], sen_out_gs), dim=1)
        attention_out1 = torch.cat((attention_out1[:,-1,:], sen_out1_gs), dim=1)
        
        raw_input = F.gelu(torch.cat((attention_out, attention_out1, semantic_matching_vector), dim=1))
        
        out = self.fc(raw_input)
        out = self.fc3(F.gelu(out))
        return out, sen_sum_out