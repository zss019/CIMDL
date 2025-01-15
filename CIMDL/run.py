# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Opinion Conflict Interaction Recognition')
parser.add_argument('--model', type=str, required=True, help='Choose a model')
parser.add_argument('--dataset', default='zhihu_dataset', type=str, help='random or pre_trained')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.dataset  # 数据集
    model_name = args.model  # 模型名称

    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    # torch.cuda.empty_cache()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
