import numpy as np
import torch
import torch.nn as nn


class LayerBase(nn.Module):
    
    def __init__(self, in_channel, out_channel, isNorm=True, isRelu=False, isDrop=False, dr=0.2):
        super(LayerBase, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.isNorm = isNorm
        self.isRelu = isRelu
        self.isDrop = isDrop
        self.liner = nn.Linear(self.in_channel, self.out_channel)
        if self.isNorm:
            self.bn = nn.LayerNorm(self.out_channel)
        if self.isRelu:
            self.relu = nn.ReLU()
        if self.isDrop:
            self.drop = nn.Dropout2d(dr, False)

    def forward(self, x):
        x = self.liner(x)
        if self.isNorm:
            x = self.bn(x)
        if self.isRelu:
            x = self.relu(x)
        if self.isDrop:
            x = self.drop(x)
        return x


class PatchMlpBase(nn.Module):
    def __init__(self, channels):
        super(PatchMlpBase, self).__init__()
        self.channels = channels
        self.isDrop = True
        self.dr = 0.1
        # self.mlp = nn.Sequential(
        #     LayerBase(self.channels[0], self.channels[1], isRelu=True),
        #     LayerBase(self.channels[1], self.channels[2], isRelu=True),
        #     LayerBase(self.channels[2], self.channels[3], isRelu=False, isDrop=self.isDrop, dr=self.dr)
        # )
        self.mlp = nn.Sequential(
            LayerBase(self.channels[0], self.channels[1], isRelu=False, isDrop=self.isDrop, dr=self.dr),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class MyPatchPC(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_patch = self.model_cfg.NUM_PATCH
        self.dim_list = self.model_cfg.MLP_DIM_LIST
        self.in_channel = self.model_cfg.IN_CHANNEL
        self.out_channel = self.model_cfg.OUT_CHANNEL

        self.mlp_patchs = nn.ModuleList()
        for i in range(self.num_patch):
            cur_mlp = [
                PatchMlpBase(self.dim_list)
            ]
            self.mlp_patchs.append(nn.Sequential(*cur_mlp))

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features
    
    def forward(self, data_dict):
        points = data_dict['points']
        batch_size = data_dict['batch_size']

        batch_idx, xyz, features = self.break_up_pc(points)

        patchs = torch.chunk(xyz, self.num_patch, dim=1)
        for id, patch in enumerate(patchs):
            patch = self.mlp_patchs[id](patch)



        return