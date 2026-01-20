from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from .FDAMb import ReplayMemory
import matplotlib.pyplot as plt

import numpy as np
from .Fourier_Tans import FDA_get_amp_pha_tensor, FDA_target_to_source,arc_add_amp
import matplotlib.pyplot as plt

torch.set_printoptions(precision=5)
buffer_size = 40
learning_rate = []
knnsize = 0


def plot_learning_rate (learning_rate) :
    if not learning_rate:
        print("No learning rate data to plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rate, label='Learning Rate')
    plt.title('Learning Rate Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.legend()
    plt.show()



class Tent(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.memory = ReplayMemory(buffer_size)
        self.mse = nn.MSELoss()
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.gt = None

    def forward(self, x):
        if self.episodic:
            self.reset()
            print('Image-specific')
        # if check_if_reset(self.optimizer):
        #     self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, self.memory, self.mse, self.gt)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.memory = ReplayMemory(buffer_size)
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)



@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    x_reshaped = x.view(-1, x.shape[2], x.shape[3])
    softmax_x = x_reshaped.softmax(0)
    log_softmax_x = x_reshaped.log_softmax(0)
    entropy = -(softmax_x * log_softmax_x).sum()/(x.shape[2]*x.shape[3])
    return entropy



@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, memory, mse, gt):

    amp,pha = FDA_get_amp_pha_tensor(x) #获取x的振幅和相位
    style = amp # 1,3,256,256

    memory_size = memory.get_size()
    pseudo_past_logits_input = None

    diff_loss = 0
    if memory_size > knnsize:
        print("memory_size: ", memory_size)
        with torch.no_grad():
            retrieved_batches = memory.get_neighbours(style.cpu().numpy(), k=knnsize).squeeze(0)  # 找最接近的几个风格
            pseudo_past_style = retrieved_batches.cuda() # 找的近似的(1,3,256,256)
            pseudo_past_logits_input = arc_add_amp( style, pseudo_past_style, pha,L=0.001) #更改过去风格后的本次图片
            # 计算pseudo_past_style和style之间的KL散度
            diff_loss = F.kl_div(pseudo_past_style.log(), style, reduction='none')
            diff_loss = (diff_loss-torch.mean(diff_loss)) #1,3,256,256
            # 将KL散度作为损失添加到总损失中
            diff_loss = torch.sum(diff_loss, dim=1) # 获取的loss
            len_loss= len(diff_loss[0]) #65536
            diff_loss = diff_loss.cpu().numpy().tolist()
            sum_loss= sum(sum(sublist) for sublist in diff_loss[0]) #float
            diff_loss = abs(sum_loss/len_loss)

            for param_group in optimizer.param_groups:
                param_group['lr'] = diff_loss * param_group['lr']
                print("learningrate:", param_group['lr'])

            # # l2
            # diff_map = (pseudo_past_style - style) ** 2
            # diff_map = torch.sum(diff_map, dim=1)
            # diff_map = diff_map - diff_map.mean()
            # diff_scalar = diff_map.mean().abs().item()
            #
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] *= diff_scalar

    if pseudo_past_logits_input!= None:
        outputs = model(pseudo_past_logits_input)
    else:
        outputs = model(x)

    loss = softmax_entropy(outputs).mean(0)
    loss += abs(diff_loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()



    with torch.no_grad(): #这里是把风格加入到memory中
        amp,pha = FDA_get_amp_pha_tensor(x)
        memory.push(amp.cpu().numpy(), amp.cpu().numpy())


    #这里output要换成tensor
    return outputs





def check_if_reset(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print(lr)
        if lr == 0:
            return True
        else:
            return False


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model