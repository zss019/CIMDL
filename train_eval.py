# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path)
    ls = nn.CrossEntropyLoss()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains,trains_1, labels, labels_1, labels_2, mask, mask2) in enumerate(train_iter):
            # print(i)
            outputs, sen_output = model(trains, trains_1, mask, mask2)
            # print('predict')
            model.zero_grad()
            # print('update parameters')
            label_sen = torch.cat((labels,labels_1))
            loss_main = ls(outputs, labels_2)
            loss_aux = ls(sen_output, label_sen)
            loss = loss_main + 0.7 * loss_aux
            loss.backward()
            optimizer.step()
            scheduler.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels_2.data.cpu()
                true_sen = label_sen.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                predic_sen = torch.max(sen_output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                train_sen_acc = metrics.accuracy_score(true_sen, predic_sen)
                print('eval dev')

                dev_acc, dev_sen_acc, dev_loss, dev_f1, dev_recall, devp_recision = evaluate(config, model, dev_iter) 
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, Train Acc Sen: {3:>6.2%}, Main Loss: {4:>6.2},  Aux Loss: {5:>6.2}, Dev Acc: {6:>6.2}, Dev Acc Sen: {7:>6.2}, Dev F1: {7:>6.2} '
                print(msg.format(total_batch, loss.item(), train_acc, train_sen_acc, loss_main.item(), loss_aux.item(), dev_acc, dev_sen_acc, dev_f1))
                # print(train_acc)
                # print(train_sen_acc)
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                writer.add_scalar("f1/dev", dev_f1, total_batch)
                writer.add_scalar("recall/dev", dev_recall, total_batch)
                writer.add_scalar("precision/dev", devp_recision, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)
 
  
def test(config, model, test_iter):    
    # test    
    model.load_state_dict(torch.load(config.save_path))    
    model.eval()    
    start_time = time.time()    
    test_acc, test_loss, test_report, test_confusion, all_predictions, all_sentiment_predictions = evaluate(config, model, test_iter, test=True)    
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'    
    print(msg.format(test_loss, test_acc))    
    print("Precision, Recall and F1-Score...")    
    print(test_report)    
    print("Confusion Matrix...")    
    print(test_confusion)    
    time_dif = get_time_dif(start_time)    
    print("Time usage:", time_dif)    
        
    result_test_file = open('Results/' + config.model_name + '.txt', 'w')    
    result_test_file.write(msg.format(test_loss, test_acc))    
    result_test_file.write('\n')    
    result_test_file.write("Precision, Recall and F1-Score...")    
    result_test_file.write(test_report)    
    result_test_file.write('\n')    
    result_test_file.write(str(test_confusion))    
    result_test_file.write('\n')    
    result_test_file.write("Time usage:" + str(time_dif))    
    result_test_file.write('\n')    
        
    # 写入每个样本的最终分类结果    
    result_test_file.write("Sample Predictions:\n")    
    for i, pred in enumerate(all_predictions):    
        result_test_file.write(f"Prediction: {pred}\n")  
      
    # 写入每个样本的情感预测结果  
    result_test_file.write("Sample Sentiment Predictions:\n")  
    for i, sent_pred in enumerate(all_sentiment_predictions):  
        result_test_file.write(f"Sentiment Prediction: {sent_pred}\n")  
        
    result_test_file.close()  
  
def evaluate(config, model, data_iter, test=False):    
    model.eval()    
        
    loss_total = 0    
    predict_all_contro = np.array([], dtype=int)    
    predict_sen_all = np.array([], dtype=int)    
    labels_all_contro = np.array([], dtype=int)    
    labels_sen_all = np.array([], dtype=int)    
    all_predictions = []  # 新增一个列表来存储所有预测结果    
    all_sentiment_predictions = []  # 新增一个列表来存储所有情感预测结果  
        
    with torch.no_grad():    
        for trains, trains_1, labels, labels_1, labels_2, mask, mask2 in data_iter:    
            outputs, sen_output = model(trains, trains_1, mask, mask2)    
                
            label_sen = torch.cat((labels, labels_1))    
            loss = F.cross_entropy(outputs, labels_2) + F.cross_entropy(sen_output, label_sen)    
            loss_total += loss    
                
            labels_contro = labels_2.data.cpu().numpy()    
            label_sen = label_sen.data.cpu().numpy()    
                
            predic_contro = torch.max(outputs.data, 1)[1].cpu().numpy()    
            predic_sen = torch.max(sen_output.data, 1)[1].cpu().numpy()    
                
            labels_all_contro = np.append(labels_all_contro, labels_contro)    
            labels_sen_all = np.append(labels_sen_all, label_sen)    
                
            predict_sen_all = np.append(predict_sen_all, predic_sen)    
            predict_all_contro = np.append(predict_all_contro, predic_contro)    
                
            all_predictions.extend(predic_contro)    
            all_sentiment_predictions.extend(predic_sen)  # 记录情感预测结果  
    
    acc = metrics.accuracy_score(labels_all_contro, predict_all_contro)    
    f1 = metrics.f1_score(labels_all_contro, predict_all_contro, average='weighted')    
    recall = metrics.recall_score(labels_all_contro, predict_all_contro, average='weighted')    
    precision = metrics.precision_score(labels_all_contro, predict_all_contro, average='weighted')    
    acc_sen = metrics.accuracy_score(labels_sen_all, predict_sen_all)    
    if test:    
        report = metrics.classification_report(labels_all_contro, predict_all_contro, target_names=config.class_list, digits=4)    
        confusion = metrics.confusion_matrix(labels_all_contro, predict_all_contro)    
        return acc, loss_total / len(data_iter), report, confusion, all_predictions, all_sentiment_predictions   
        
    return acc, acc_sen, loss_total / len(data_iter), f1, recall, precision