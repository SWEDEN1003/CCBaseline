'''

Copyright (C) 2019 Shuangwei Liu

All rights reserved.
'''
import torch
import torch.nn as nn

class Batchnorm2d_wei(nn.Module):

     def __init__(self, num_features, eps=1e-5, momentum=0.1):
         super(Batchnorm2d_wei, self).__init__()
         self.in_channel = num_features
         self.eps = eps
         self.momentum = momentum
         self.gamma = nn.Parameter(torch.ones(num_features))
         self.beta = nn.Parameter(torch.zeros(num_features))
         self.running_mean = torch.zeros(num_features).cuda()
         self.running_var = torch.ones(num_features).cuda()

     def forward(self, x, is_train='True'):
         N, C, H, W = x.shape
         x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
         out_flat = self.batchnorm_forward(x_flat, is_train, N)
         out = out_flat.reshape(N, H, W, C).permute(0, 3, 1, 2)
         return out

     def batchnorm_forward(self, x_flat, is_train, N):
         NHW, C = x_flat.shape

         if is_train == 'True':
             x_state = x_flat.reshape(N, -1, C).mean(1).squeeze()

             sample_mean = torch.mean(x_state, 0)
             sample_var = torch.var(x_state, 0)
             out = (x_flat-sample_mean) / torch.sqrt(sample_var+self.eps)

             self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*sample_mean
             self.running_var = self.momentum*self.running_var + (1-self.momentum)*sample_var

             out = self.gamma*out + self.beta

         elif is_train == 'False':
             scale = self.gamma/torch.sqrt(self.running_var+self.eps)
             out = x_flat * scale + (self.beta-self.running_mean*scale)

         return out






