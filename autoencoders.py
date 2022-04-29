from torch import nn
import torch


class AutoEncoder_MNIST(nn.Module):
    def __init__(self):
        super(AutoEncoder_MNIST, self).__init__()
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


class Denoise_AutoEncoder_MNIST(nn.Module):
    def __init__(self):
        super(Denoise_AutoEncoder_MNIST, self).__init__()
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


class ConvAutoEncoder(nn.Module):
    def __init__(self, encoded_space_dim=10, fc2_input_dim=128):
        super(ConvAutoEncoder, self).__init__()
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


