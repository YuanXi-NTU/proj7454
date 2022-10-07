import os
import time
import random
import torch
import numpy as np
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def backup(log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tmp = time.localtime(time.time())
    cur_time = f"{tmp.tm_mon}-{tmp.tm_mday}-{tmp.tm_hour}-{tmp.tm_min}-{tmp.tm_sec}"
    path = os.path.join(log_path, cur_time)
    os.mkdir(path)
    os.system(f'cp train.py cfg.yaml model.py {path}/')
    return path

ori_classes = ['over', 'in front of', 'beside', 'on', 'in', 'attached to'
            , 'hanging from', 'on the back of', 'falling off', 'going down',
                        'painted on','walking on', 'running on', 'crossing', 'standing on',
                        'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from',
                        'wearing', 'holding', 'carrying', 'looking at', 'guiding', 'kissing',
                        'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking',
                        'playing with', 'chasing', 'climbing', 'cleaning', 'playing',
                        'touching', 'pushing', 'pulling', 'opening','cooking', 'talking to',
                        'throwing', 'slicing','driving', 'riding', 'parked on', 'driving on',
                        'about to hit', 'kicking', 'swinging','entering', 'exiting', 'enclosing', 'leaning on']

classes = [ 'hanging from', 'on the back of', 'falling off', 'going down',
                        'painted on','walking on', 'running on', 'crossing', 'standing on',
                        'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from',
                        'wearing', 'holding', 'carrying', 'looking at', 'guiding', 'kissing',
                        'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking',
                        'playing with', 'chasing', 'climbing', 'cleaning', 'playing',
                        'touching', 'pushing', 'pulling', 'opening','cooking', 'talking to',
                        'throwing', 'slicing','driving', 'riding', 'parked on', 'driving on',
                        'about to hit', 'kicking', 'swinging','entering', 'exiting', 'enclosing', 'leaning on']

simp_classes = ['hanging', 'on the back of', 'falling', 'going down',
                        'painted','walking', 'running', 'crossing', 'standing',
                        'lying', 'sitting', 'flying', 'jumping', 'jumping',
                        'wearing', 'holding', 'carrying', 'looking', 'guiding', 'kissing',
                        'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking',
                        'playing with', 'chasing', 'climbing', 'cleaning', 'playing',
                        'touching', 'pushing', 'pulling', 'opening','cooking', 'talking',
                        'throwing', 'slicing','driving', 'riding', 'parked', 'driving',
                        'hit', 'kicking', 'swinging','entering', 'exiting', 'enclosing', 'leaning']

