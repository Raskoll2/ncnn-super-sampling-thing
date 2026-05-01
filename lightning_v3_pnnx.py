import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.conv_in = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=12, kernel_size=(3,3), out_channels=8, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.res_blocks_0_body_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(3,3), out_channels=8, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.res_blocks_0_body_1 = nn.LeakyReLU(negative_slope=0.2)
        self.res_blocks_0_body_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(3,3), out_channels=8, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.res_blocks_0_ca_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.res_blocks_0_ca_fc_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(1,1), out_channels=2, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.res_blocks_0_ca_fc_1 = nn.LeakyReLU(negative_slope=0.2)
        self.res_blocks_0_ca_fc_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2, kernel_size=(1,1), out_channels=8, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.res_blocks_0_ca_fc_3 = nn.Sigmoid()
        self.res_blocks_0_sa_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2, kernel_size=(3,3), out_channels=1, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.res_blocks_0_sa_sigmoid = nn.Sigmoid()
        self.res_blocks_1_body_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(3,3), out_channels=8, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.res_blocks_1_body_1 = nn.LeakyReLU(negative_slope=0.2)
        self.res_blocks_1_body_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(3,3), out_channels=8, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.res_blocks_1_ca_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.res_blocks_1_ca_fc_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(1,1), out_channels=2, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.res_blocks_1_ca_fc_1 = nn.LeakyReLU(negative_slope=0.2)
        self.res_blocks_1_ca_fc_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2, kernel_size=(1,1), out_channels=8, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.res_blocks_1_ca_fc_3 = nn.Sigmoid()
        self.res_blocks_1_sa_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=2, kernel_size=(3,3), out_channels=1, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.res_blocks_1_sa_sigmoid = nn.Sigmoid()
        self.conv_mid = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(3,3), out_channels=8, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv_up = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=8, kernel_size=(3,3), out_channels=48, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=4)

        archive = zipfile.ZipFile('lightning_v3.pnnx.bin', 'r')
        self.conv_in.bias = self.load_pnnx_bin_as_parameter(archive, 'conv_in.bias', (8), 'float32')
        self.conv_in.weight = self.load_pnnx_bin_as_parameter(archive, 'conv_in.weight', (8,12,3,3), 'float32')
        self.res_blocks_0_body_0.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.body.0.bias', (8), 'float32')
        self.res_blocks_0_body_0.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.body.0.weight', (8,8,3,3), 'float32')
        self.res_blocks_0_body_2.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.body.2.bias', (8), 'float32')
        self.res_blocks_0_body_2.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.body.2.weight', (8,8,3,3), 'float32')
        self.res_blocks_0_ca_fc_0.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.ca.fc.0.bias', (2), 'float32')
        self.res_blocks_0_ca_fc_0.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.ca.fc.0.weight', (2,8,1,1), 'float32')
        self.res_blocks_0_ca_fc_2.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.ca.fc.2.bias', (8), 'float32')
        self.res_blocks_0_ca_fc_2.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.ca.fc.2.weight', (8,2,1,1), 'float32')
        self.res_blocks_0_sa_conv.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.sa.conv.bias', (1), 'float32')
        self.res_blocks_0_sa_conv.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.0.sa.conv.weight', (1,2,3,3), 'float32')
        self.res_blocks_1_body_0.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.body.0.bias', (8), 'float32')
        self.res_blocks_1_body_0.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.body.0.weight', (8,8,3,3), 'float32')
        self.res_blocks_1_body_2.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.body.2.bias', (8), 'float32')
        self.res_blocks_1_body_2.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.body.2.weight', (8,8,3,3), 'float32')
        self.res_blocks_1_ca_fc_0.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.ca.fc.0.bias', (2), 'float32')
        self.res_blocks_1_ca_fc_0.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.ca.fc.0.weight', (2,8,1,1), 'float32')
        self.res_blocks_1_ca_fc_2.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.ca.fc.2.bias', (8), 'float32')
        self.res_blocks_1_ca_fc_2.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.ca.fc.2.weight', (8,2,1,1), 'float32')
        self.res_blocks_1_sa_conv.bias = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.sa.conv.bias', (1), 'float32')
        self.res_blocks_1_sa_conv.weight = self.load_pnnx_bin_as_parameter(archive, 'res_blocks.1.sa.conv.weight', (1,2,3,3), 'float32')
        self.conv_mid.bias = self.load_pnnx_bin_as_parameter(archive, 'conv_mid.bias', (8), 'float32')
        self.conv_mid.weight = self.load_pnnx_bin_as_parameter(archive, 'conv_mid.weight', (8,8,3,3), 'float32')
        self.conv_up.bias = self.load_pnnx_bin_as_parameter(archive, 'conv_up.bias', (48), 'float32')
        self.conv_up.weight = self.load_pnnx_bin_as_parameter(archive, 'conv_up.weight', (48,8,3,3), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = F.upsample(v_0, align_corners=False, mode='bilinear', scale_factor=(2.0,2.0))
        v_2 = self.unshuffle(v_0)
        v_3 = self.conv_in(v_2)
        v_4 = F.leaky_relu(v_3, negative_slope=0.2)
        v_5 = self.res_blocks_0_body_0(v_4)
        v_6 = self.res_blocks_0_body_1(v_5)
        v_7 = self.res_blocks_0_body_2(v_6)
        v_8 = self.res_blocks_0_ca_pool(v_7)
        v_9 = self.res_blocks_0_ca_fc_0(v_8)
        v_10 = self.res_blocks_0_ca_fc_1(v_9)
        v_11 = self.res_blocks_0_ca_fc_2(v_10)
        v_12 = self.res_blocks_0_ca_fc_3(v_11)
        v_13 = (v_7 * v_12)
        v_14 = torch.mean(v_13, dim=(1,), keepdim=True)
        v_15, _ = torch.max(v_13, dim=1, keepdim=True)
        v_16 = torch.cat((v_14, v_15), dim=1)
        v_17 = self.res_blocks_0_sa_conv(v_16)
        v_18 = self.res_blocks_0_sa_sigmoid(v_17)
        v_19 = ((v_13 * v_18) + v_4)
        v_20 = self.res_blocks_1_body_0(v_19)
        v_21 = self.res_blocks_1_body_1(v_20)
        v_22 = self.res_blocks_1_body_2(v_21)
        v_23 = self.res_blocks_1_ca_pool(v_22)
        v_24 = self.res_blocks_1_ca_fc_0(v_23)
        v_25 = self.res_blocks_1_ca_fc_1(v_24)
        v_26 = self.res_blocks_1_ca_fc_2(v_25)
        v_27 = self.res_blocks_1_ca_fc_3(v_26)
        v_28 = (v_22 * v_27)
        v_29 = torch.mean(v_28, dim=(1,), keepdim=True)
        v_30, _ = torch.max(v_28, dim=1, keepdim=True)
        v_31 = torch.cat((v_29, v_30), dim=1)
        v_32 = self.res_blocks_1_sa_conv(v_31)
        v_33 = self.res_blocks_1_sa_sigmoid(v_32)
        v_34 = ((v_28 * v_33) + v_19)
        v_35 = self.conv_mid(v_34)
        v_36 = (v_35 + v_4)
        v_37 = self.conv_up(v_36)
        v_38 = self.pixel_shuffle(v_37)
        v_39 = (v_38 + v_1)
        return v_39

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 256, 256, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("lightning_v3_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 256, 256, dtype=torch.float)

    torch.onnx.export(net, v_0, "lightning_v3_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 256, 256, dtype=torch.float)

    import pnnx
    pnnx.export(net, "lightning_v3_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 256, 256, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
