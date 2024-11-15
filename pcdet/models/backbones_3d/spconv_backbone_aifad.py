from functools import partial
import torch
import torch.nn as nn
import numpy as np

from ...utils.spconv_utils import replace_feature, spconv


def index2points_1(indices, pts_range=[0, -368, -120, 776, 368, 40], voxel_size=[0.5, 0.5, 4], stride=8):
    """
    convert 3D voxel indices to a set of grid points.
    """

    voxel_size = np.array(voxel_size) * stride
    min_x = pts_range[0] + voxel_size[0] / 2
    min_y = pts_range[1] + voxel_size[1] / 2
    min_z = pts_range[2] + voxel_size[2] / 2

    new_indices = indices.clone().float()
    indices_float = indices.clone().float()
    new_indices[:, 1] = indices_float[:, 3] * voxel_size[0] + min_x
    new_indices[:, 2] = indices_float[:, 2] * voxel_size[1] + min_y
    new_indices[:, 3] = indices_float[:, 1] * voxel_size[2] + min_z

    return new_indices


def index2points(indices, stride, pts_range=[0, -368, -120, 776, 368, 40], voxel_size=[0.5, 0.5, 4]):
    """
    convert 3D voxel indices to a set of grid points.
    """

    voxel_size = np.array(voxel_size) * stride

    new_indices = indices.clone().float()
    indices_float = indices.clone().float()
    new_indices[:, 1] = indices_float[:, 3] * voxel_size[0] + pts_range[0]
    new_indices[:, 2] = indices_float[:, 2] * voxel_size[1] + pts_range[1]
    new_indices[:, 3] = (indices_float[:, 1] - 1) * voxel_size[2] + pts_range[2]

    return new_indices


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m


def post_act_block2d(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                     conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.indice_key = indice_key
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SparseBasicBlock2D(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock2D, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.indice_key = indice_key
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SPConvFuse(nn.Module):
    def __init__(self, input_channel, output_channel, norm_fn, indice_key):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.norm_fn = norm_fn
        self.indice_key = indice_key

        # block = post_act_block
        self.conv = spconv.SubMConv3d(
            input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True, indice_key=indice_key
        )
        self.bn1 = norm_fn(output_channel)
        self.relu = nn.ReLU()

    def forward(self, sp_tensor_1, sp_tensor_2):
        sp_tensor_2.indices[:, 1:] *= 2
        sp_tensor_1 = sp_tensor_1.replace_feature(torch.cat([sp_tensor_1.features, sp_tensor_2.features]))
        sp_tensor_1.indices = torch.cat([sp_tensor_1.indices, sp_tensor_2.indices])

        out = self.conv(sp_tensor_1)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        return out


class SPConvFuse2D(nn.Module):
    def __init__(self, input_channel, output_channel, norm_fn, indice_key):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.norm_fn = norm_fn
        self.indice_key = indice_key

        # block = post_act_block
        self.conv = spconv.SubMConv2d(
            input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True, indice_key=indice_key
        )
        self.bn1 = norm_fn(output_channel)
        self.relu = nn.ReLU()

    def forward(self, sp_tensor_1, sp_tensor_2):
        sp_tensor_2.indices[:, 1:] *= 2
        sp_tensor_1 = sp_tensor_1.replace_feature(torch.cat([sp_tensor_1.features, sp_tensor_2.features]))
        sp_tensor_1.indices = torch.cat([sp_tensor_1.indices, sp_tensor_2.indices])

        out = self.conv(sp_tensor_1)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        return out
    

class SPConvFuseAdd(nn.Module):
    def __init__(self, input_channel, output_channel, norm_fn, indice_key):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.norm_fn = norm_fn
        self.indice_key = indice_key

        # block = post_act_block
        self.conv = spconv.SubMConv3d(
            input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True, indice_key=indice_key
        )
        self.bn1 = norm_fn(output_channel)
        self.relu = nn.ReLU()

    def forward(self, sp_tensor_1, sp_tensor_2):
        sp_tensor_1 = sp_tensor_1.replace_feature(torch.add(sp_tensor_1.features, sp_tensor_2.features))

        out = self.conv(sp_tensor_1)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        return out


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
        if isRelu:
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


# xnd BasicBlock add
class SparseConvBasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, indice_key=None, sparse_shape=None):
        super(SparseConvBasicBlock, self).__init__()
        assert norm_fn is not None
        self.indice_key = indice_key
        self.sparse_shape = sparse_shape
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu1 = nn.ReLU()

        self.indices_mlp_1 = LayerBase(in_channel=3, out_channel=planes // 2, isNorm=True, isRelu=True, isDrop=False)
        self.indices_mlp_2 = LayerBase(in_channel=planes // 2, out_channel=planes, isNorm=True, isRelu=True, isDrop=False)

        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.relu2 = nn.ReLU()

    @staticmethod
    def get_stride(sp_shape_1, sp_shape_2):
        stride = sp_shape_2
        stride[0] = (sp_shape_1[0] - 1) / (sp_shape_2[0] - 1)
        stride[1] = sp_shape_1[1] / sp_shape_2[1]
        stride[2] = sp_shape_1[2] / sp_shape_2[2]
        return stride

    def forward(self, x):
        identity = x
        x_spatial_shape = x.spatial_shape

        stride = self.get_stride(self.sparse_shape, np.array(x_spatial_shape, dtype=self.sparse_shape.dtype))
        new_x_indices = index2points(x.indices, stride=stride)
        new_x_indices_mlp = self.indices_mlp_1(new_x_indices[:, 1:])
        new_x_indices_mlp = self.indices_mlp_2(new_x_indices_mlp)

        x = replace_feature(x, x.features + new_x_indices_mlp)

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu1(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu2(out.features))

        return out


# xnd1 BasicBlock cat
class SparseConvBasicBlock_1(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, indice_key=None, sparse_shape=None):
        super(SparseConvBasicBlock_1, self).__init__()
        assert norm_fn is not None
        self.inplanes = inplanes
        self.planes = planes
        self.indice_key = indice_key
        self.sparse_shape = sparse_shape
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu1 = nn.ReLU()

        self.indices_mlp_1 = LayerBase(in_channel=inplanes, out_channel=inplanes // 2, isNorm=True, isRelu=True, isDrop=False)
        self.indices_mlp_2 = LayerBase(in_channel=3, out_channel=inplanes // 2, isNorm=True, isRelu=True, isDrop=False)

        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.relu2 = nn.ReLU()

    @staticmethod
    def get_stride(sp_shape_1, sp_shape_2):
        stride = sp_shape_2
        stride[0] = (sp_shape_1[0] - 1) / (sp_shape_2[0] - 1)
        stride[1] = sp_shape_1[1] / sp_shape_2[1]
        stride[2] = sp_shape_1[2] / sp_shape_2[2]
        return stride

    def forward(self, x):
        identity = x
        x_spatial_shape = x.spatial_shape

        stride = self.get_stride(self.sparse_shape, np.array(x_spatial_shape, dtype=self.sparse_shape.dtype))
        new_x_indices = index2points(x.indices, stride=stride)

        new_x_features = self.indices_mlp_1(x.features)
        new_x_indices_mlp = self.indices_mlp_2(new_x_indices[:, 1:])

        x = replace_feature(x, torch.cat([new_x_features, new_x_indices_mlp], dim=-1))

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu1(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu2(out.features))

        return out


# epoch500_0809
class PillarBackBone8xAIFAD(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # self.sparse_shape = grid_size[[1, 0]]
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        self.channels = channels
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        block = post_act_block
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm0'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )

        # conv1
        self.conv1 = spconv.SparseSequential(
            block(channels[0], channels[0], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            # SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )
        self.sp_conv_fuse_12 = SPConvFuse(channels[0], channels[0], norm_fn=norm_fn, indice_key='fuse12')

        # conv2
        self.conv2 = spconv.SparseSequential(
            block(channels[0], channels[1], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            # SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )
        self.conv2_2 = spconv.SparseSequential(
            block(channels[1], channels[1], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv22', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res22'),
            # SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res22'),
        )
        self.sp_conv_fuse_22 = SPConvFuse(channels[1], channels[1], norm_fn=norm_fn, indice_key='fuse22')

        # conv3
        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            # SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )
        self.conv3_2 = spconv.SparseSequential(
            block(channels[2], channels[2], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv32', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res32'),
            # SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res32'),
        )
        self.sp_conv_fuse_32 = SPConvFuse(channels[2], channels[2], norm_fn=norm_fn, indice_key='fuse32')

        # conv4
        self.conv4 = spconv.SparseSequential(
            block(channels[2], channels[3], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            # SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )
        self.conv4_2 = spconv.SparseSequential(
            block(channels[3], channels[3], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv42', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res42'),
            # SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res42'),
        )
        self.sp_conv_fuse_42 = SPConvFuse(channels[3], channels[3], norm_fn=norm_fn, indice_key='fuse42')
        
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3],
        }

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out
    
    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv_fuse_12 = self.sp_conv_fuse_12(x, x_conv1)

        x_conv2 = self.conv2(x_conv_fuse_12)
        x_conv2_2 = self.conv2_2(x_conv2)
        x_conv_fuse_22 = self.sp_conv_fuse_22(x_conv2, x_conv2_2)

        x_conv3 = self.conv3(x_conv_fuse_22)
        x_conv3_2 = self.conv3_2(x_conv3)
        x_conv_fuse_32 = self.sp_conv_fuse_32(x_conv3, x_conv3_2)

        x_conv4 = self.conv4(x_conv_fuse_32)
        x_conv4_2 = self.conv4_2(x_conv4)
        x_conv_fuse_42 = self.sp_conv_fuse_42(x_conv4, x_conv4_2)

        out = self.bev_out(x_conv_fuse_42)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv_fuse_12,
                'x_conv2': x_conv_fuse_22,
                'x_conv3': x_conv_fuse_32,
                'x_conv4': x_conv_fuse_42,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


# epoch500_0809_2d
class PillarBackBone8xAIFAD2D(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # self.sparse_shape = grid_size[[1, 0]]
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3, 3])
        channels = model_cfg.get('CHANNELS', [16, 32, 64, 128, 128])
        self.channels = channels
        out_channel = model_cfg.get('OUT_CHANNEL', 128)

        block = post_act_block2d
        
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv2d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm0'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )

        # conv1
        self.conv1 = spconv.SparseSequential(
            block(channels[0], channels[0], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock2D(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            # SparseBasicBlock2D(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )
        self.sp_conv_fuse_12 = SPConvFuse2D(channels[0], channels[0], norm_fn=norm_fn, indice_key='fuse12')

        # conv2
        self.conv2 = spconv.SparseSequential(
            block(channels[0], channels[1], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock2D(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            # SparseBasicBlock2D(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )
        self.conv2_2 = spconv.SparseSequential(
            block(channels[1], channels[1], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv22', conv_type='spconv'),
            SparseBasicBlock2D(channels[1], channels[1], norm_fn=norm_fn, indice_key='res22'),
            # SparseBasicBlock2D(channels[1], channels[1], norm_fn=norm_fn, indice_key='res22'),
        )
        self.sp_conv_fuse_22 = SPConvFuse2D(channels[1], channels[1], norm_fn=norm_fn, indice_key='fuse22')

        # conv3
        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock2D(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            # SparseBasicBlock2D(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )
        self.conv3_2 = spconv.SparseSequential(
            block(channels[2], channels[2], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv32', conv_type='spconv'),
            SparseBasicBlock2D(channels[2], channels[2], norm_fn=norm_fn, indice_key='res32'),
            # SparseBasicBlock2D(channels[2], channels[2], norm_fn=norm_fn, indice_key='res32'),
        )
        self.sp_conv_fuse_32 = SPConvFuse2D(channels[2], channels[2], norm_fn=norm_fn, indice_key='fuse32')

        # conv4
        self.conv4 = spconv.SparseSequential(
            block(channels[2], channels[3], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock2D(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            # SparseBasicBlock2D(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )
        self.conv4_2 = spconv.SparseSequential(
            block(channels[3], channels[3], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv42', conv_type='spconv'),
            SparseBasicBlock2D(channels[3], channels[3], norm_fn=norm_fn, indice_key='res42'),
            # SparseBasicBlock2D(channels[3], channels[3], norm_fn=norm_fn, indice_key='res42'),
        )
        self.sp_conv_fuse_42 = SPConvFuse2D(channels[3], channels[3], norm_fn=norm_fn, indice_key='fuse42')
        
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3],
        }

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x_conv.spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out
    
    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords[:, [0, 2, 3]].int(),
            spatial_shape=self.sparse_shape[1:],
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv_fuse_12 = self.sp_conv_fuse_12(x, x_conv1)

        x_conv2 = self.conv2(x_conv_fuse_12)
        x_conv2_2 = self.conv2_2(x_conv2)
        x_conv_fuse_22 = self.sp_conv_fuse_22(x_conv2, x_conv2_2)

        x_conv3 = self.conv3(x_conv_fuse_22)
        x_conv3_2 = self.conv3_2(x_conv3)
        x_conv_fuse_32 = self.sp_conv_fuse_32(x_conv3, x_conv3_2)

        x_conv4 = self.conv4(x_conv_fuse_32)
        x_conv4_2 = self.conv4_2(x_conv4)
        x_conv_fuse_42 = self.sp_conv_fuse_42(x_conv4, x_conv4_2)

        out = self.bev_out(x_conv_fuse_42)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv_fuse_12,
                'x_conv2': x_conv_fuse_22,
                'x_conv3': x_conv_fuse_32,
                'x_conv4': x_conv_fuse_42,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict
