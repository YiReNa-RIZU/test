import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np


def write_head(data, ouput_path):
    with open(ouput_path, 'a+') as f:
        head = "#" + "ifndef" + " " + "__WEIGHT_H" + "\n"
        head = head + "#" + "define" + " " + "__WEIGHT_H" + "\n"
        head = head + "\n"
        head = head + "#" + "include" + " " + "\"net_weight.h\""
        head = head + "\n" + "\n"
        f.write(head)


def write_mat_dim(param):

    if len(param.size()) == 1:
        data_str = f"[{param.size(0)}]"
    elif len(param.size()) == 2:
        data_str = f"[{param.size(0)}][{param.size(1)}]"
    elif len(param.size()) == 3:
        data_str = f"[{param.size(0)}][{param.size(1)}][{param.size(2)}]"
    elif len(param.size()) == 4:
        data_str = f"[{param.size(0)}][{param.size(1)}][{param.size(2)}][{param.size(3)}]"
    else:
        raise ValueError('param len error!')

    return data_str


def write_name_data(model, module_name):
    data_str = ''
    new_str = ''
    num = 1
    name_weight = ''
    name_bias = ''
    name_var = ''
    name_mean = ''
    if module_name.__name__ == 'BatchNorm1d':
        name_mean = f'float batchNorm1d_mean'
        name_var = f'float batchNorm1d_var'
        name_weight = f'float batchNorm1d_weight'
        name_bias = f'float batchNorm1d_bias'
    elif module_name.__name__ == 'Conv1d':
        name_weight = f'float weight_conv1d'
    elif module_name.__name__ == 'Linear':
        name_weight = f'float Weight_con'
    else:
        raise ValueError('module name error!')

    for m in model.modules():
        if isinstance(m, module_name):

            if m.weight != None:
                mat_dim = write_mat_dim(m.weight)
                new_str += f'{name_weight}{num}{mat_dim} = '
                data_str = str(m.weight.tolist())
                for i in data_str:
                    if i == '[':
                        new_str += "{"
                    elif i == ']':
                        new_str += '}'
                    else:
                        new_str += i
                new_str = new_str + ';' + '\n' + '\n'

            if m.bias != None:
                mat_dim = write_mat_dim(m.bias)
                new_str += f'{name_bias}{num}{mat_dim} = '
                data_str = str(m.bias.tolist())
                for i in data_str:
                    if i == '[':
                        new_str += "{"
                    elif i == ']':
                        new_str += '}'
                    else:
                        new_str += i
                new_str = new_str + ';' + '\n' + '\n'

            try:
                running_mean = m.running_mean
            except AttributeError:
                running_mean = None

            try:
                running_var = m.running_var
            except AttributeError:
                running_var = None

            if running_mean != None:
                mat_dim = write_mat_dim(m.running_mean)
                new_str += f'{name_mean}{num}{mat_dim} = '
                data_str = str(m.running_mean.tolist())
                for i in data_str:
                    if i == '[':
                        new_str += "{"
                    elif i == ']':
                        new_str += '}'
                    else:
                        new_str += i
                new_str = new_str + ';' + '\n' + '\n'

            if running_var != None:
                mat_dim = write_mat_dim(m.running_var)
                new_str += f'{name_var}{num}{mat_dim} = '
                data_str = str(m.running_var.tolist())
                for i in data_str:
                    if i == '[':
                        new_str += "{"
                    elif i == ']':
                        new_str += '}'
                    else:
                        new_str += i
                new_str = new_str + ';' + '\n' + '\n'

            num += 1

    return new_str


def write_data(data, output_path):
    output_str = ''
    output_str += write_name_data(data, nn.Conv1d)
    output_str += '\n'
    output_str += write_name_data(data, nn.BatchNorm1d)
    output_str += '\n'
    output_str += write_name_data(data, nn.Linear)
    output_str += '\n'

    with open(output_path, "a+") as f:
        f.write(output_str)
        f.write("#endif")


if __name__ == '__main__':
    input_path = './save_model/best_model.pth'
    output_path = 'net_weight.h'
    data = torch.load(input_path)
    data.eval()
    write_head(data, output_path)
    write_data(data, output_path)