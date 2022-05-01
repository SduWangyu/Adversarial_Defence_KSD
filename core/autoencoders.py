from torch import nn
import torch


class AutoEncoderMNIST(nn.Module):
    def __init__(self):
        super(AutoEncoderMNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output


class DenoiseAutoEncoderMNIST(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoderMNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.autocode = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.autocode(x)
        return output


class ConvAutoEncoderMNIST(nn.Module):
    def __init__(self, encoded_space_dim=10, fc2_input_dim=128):
        super(ConvAutoEncoderMNIST, self).__init__()
        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )

        # decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8), nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

    def load_exist(self, path, is_eval=True):
        self.load_state_dict(torch.load(path))
        if is_eval:
            self.eval()


test_tensor = torch.rand((1, 3, 32, 32))


class ConvAutoEncoderCIFAR10(nn.Module):
    def __init__(self):
        super(ConvAutoEncoderCIFAR10, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]`
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """

    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in

    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        # self.BN
        # self.relu
        self.c3 = nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False)
        self.BN_ex = nn.BatchNorm2d(out_channels * BottleNeck.expansion)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

    def forward(self, x):
        tmp = self.c1(x)
        tmp = self.BN(tmp)
        tmp = self.relu(tmp)
        tmp = self.c2(tmp)
        tmp = self.BN(tmp)
        tmp = self.relu(tmp)
        tmp = self.c3(tmp)
        tmp = self.BN_ex(tmp)
        tmp2 = self.shortcut(x)
        tmp = nn.ReLU(inplace=True)(tmp + tmp2)
        return tmp


class BottleNeckDecode(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_out_channels = in_channels // self.expansion
        self.c1 = nn.ConvTranspose2d(in_channels, mid_out_channels, kernel_size=1, bias=False)
        self.BN_ex = nn.BatchNorm2d(mid_out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.c2 = nn.ConvTranspose2d(mid_out_channels, mid_out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        # self.BN_ex
        # self.relu
        self.c3 = nn.ConvTranspose2d(mid_out_channels, out_channels, kernel_size=1, bias=False)
        self.BN = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        tmp = self.c1(x)
        tmp = self.BN_ex(tmp)
        tmp = self.relu(tmp)
        tmp = self.c2(tmp)
        tmp = self.BN_ex(tmp)
        tmp = self.relu(tmp)
        tmp = self.c3(tmp)
        tmp = self.BN(tmp)
        tmp2 = self.shortcut(x)
        tmp = nn.ReLU(inplace=True)(tmp + tmp2)
        return tmp


class ResAutoEncoderCIFAR100(nn.Module):
    num_blocks = [3, 4, 6, 3]  # Resnet50

    def __init__(self):
        super(ResAutoEncoderCIFAR100, self).__init__()
        self.in_channels = 64
        self.out_channels = 64
        self.init_conv = nn.Conv2d(3, 64, 3, 1, 1)  # 16 32 32
        self.BN = nn.BatchNorm2d(64)

        self.encode_layer1 = self._make_layer(BottleNeck, 64, self.num_blocks[0], stride=1)
        self.encode_layer2 = self._make_layer(BottleNeck, 128, self.num_blocks[1], stride=1)  # 32 8 8 - 48 8 8
        self.encode_layer3 = self._make_layer(BottleNeck, 256, self.num_blocks[2], stride=1)
        self.encode_layer4 = self._make_layer(BottleNeck, 512, self.num_blocks[3], stride=1)  # 32 8 8 - 48 8 8
        self.encode_relu = nn.ReLU()

        self.decode_layer1 = self._make_decode_layer(BottleNeckDecode, 256, self.num_blocks[3], stride=1)
        self.decode_layer2 = self._make_decode_layer(BottleNeckDecode, 128, self.num_blocks[2], stride=1)
        self.decode_layer3 = self._make_decode_layer(BottleNeckDecode, 64, self.num_blocks[1], stride=1)
        self.decode_layer4 = self._make_decode_layer(BottleNeckDecode, 16, self.num_blocks[0], stride=1)
        self.decode_out_conv = nn.ConvTranspose2d(64, 3, 3, 1, 1)  # 3 32 32
        self.decode_tanh = nn.Tanh()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        other_strides = [1] * (num_blocks - 1)
        layers = []

        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _stride in other_strides:
            layers.append(block(self.in_channels, out_channels, _stride))

        return nn.Sequential(*layers)

    def _make_decode_layer(self, block, out_channels, num_blocks, stride):
        other_strides = [1] * (num_blocks - 1)
        layers = []

        for _stride in other_strides:
            layers.append(block(self.in_channels, self.in_channels, _stride))

        layers.append(block(self.in_channels, out_channels * block.expansion, stride))
        self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, inputs):
        init_conv = self.init_conv(inputs)
        init_conv = self.encode_relu(self.BN(init_conv))
        rb1 = self.encode_layer1(init_conv)
        rb2 = self.encode_layer2(rb1)
        rb3 = self.encode_layer3(rb2)
        rb4 = self.encode_layer4(rb3)

        rb5 = self.decode_layer1(rb4)
        rb6 = self.decode_layer2(rb5)
        rb7 = self.decode_layer3(rb6)
        output = self.decode_layer4(rb7)
        output = self.decode_out_conv(output)
        output = self.decode_tanh(output)
        return output


class ResAutoEncoderCIFAR10(nn.Module):

    def __init__(self):
        super(ResAutoEncoderCIFAR10, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1)  # 16 32 32
        self.BN = nn.BatchNorm2d(16)
        self.encode_rb1 = ResBlock(16, 16, 3, 2, 1, 'encode')  # 16 16 16
        self.encode_rb2 = ResBlock(16, 32, 3, 1, 1, 'encode')  # 32 16 16
        self.encode_rb3 = ResBlock(32, 32, 3, 2, 1, 'encode')  # 32 8 8
        self.encode_rb4 = ResBlock(32, 48, 3, 1, 1, 'encode')  # 48 8 8
        self.encode_rb5 = ResBlock(48, 48, 3, 2, 1, 'encode')  # 48 4 4
        self.encode_rb6 = ResBlock(48, 64, 3, 2, 1, 'encode')  # 64 2 2
        self.encode_relu = nn.ReLU()

        self.decode_rb1 = ResBlock(64, 48, 2, 2, 0, 'decode')  # 48 4 4
        self.decode_rb2 = ResBlock(48, 48, 2, 2, 0, 'decode')  # 48 8 8
        self.decode_rb3 = ResBlock(48, 32, 3, 1, 1, 'decode')  # 32 8 8
        self.decode_rb4 = ResBlock(32, 32, 2, 2, 0, 'decode')  # 32 16 16
        self.decode_rb5 = ResBlock(32, 16, 3, 1, 1, 'decode')  # 16 16 16
        self.decode_rb6 = ResBlock(16, 16, 2, 2, 0, 'decode')  # 16 32 32
        self.decode_out_conv = nn.ConvTranspose2d(16, 3, 3, 1, 1)  # 3 32 32
        self.decode_tanh = nn.Tanh()

    def forward(self, inputs):
        init_conv = self.encode_relu(self.BN(self.init_conv(inputs)))
        rb1 = self.encode_rb1(init_conv)
        rb2 = self.encode_rb2(rb1)
        rb3 = self.encode_rb3(rb2)
        rb4 = self.encode_rb4(rb3)
        rb5 = self.encode_rb5(rb4)
        rb6 = self.encode_rb6(rb5)

        rb1 = self.decode_rb1(rb6)
        rb2 = self.decode_rb2(rb1)
        rb3 = self.decode_rb3(rb2)
        rb4 = self.decode_rb4(rb3)
        rb5 = self.decode_rb5(rb4)
        rb6 = self.decode_rb6(rb5)
        out_conv = self.decode_out_conv(rb6)
        output = self.decode_tanh(out_conv)
        return output