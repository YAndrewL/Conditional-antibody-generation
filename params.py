# -*- coding: utf-8 -*-
# @Time         : 2022/5/13 13:14
# @Author       : Yufan Liu
# @Description  : Load training parameters


import yaml

class GetParams(object):
    def __init__(self, yaml_file):
        self.param_dict = yaml.load(open(yaml_file), Loader=yaml.FullLoader)

    def model_params(self):
        return self.param_dict['model_params']
    
    def trainer_params(self):
        return self.param_dict['trainer_params']

    def data_params(self):
        return self.param_dict['data_params']
