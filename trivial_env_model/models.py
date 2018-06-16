import torch.nn.functional as F
import torch.nn as nn


class GenerativeModel(nn.Module):
    def __init__(self):
        """
        Model architecture based on "Action-Conditional Video Prediction using Deep Neural Networks" paper
        url: https://arxiv.org/pdf/1507.08750.pdfs
        """
        nn.Module.__init__(self)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, 8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 6, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 6, stride=2, padding=2)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2, padding=1)

        # FC layer pre action-conditional transformation
        self.fc1 = nn.Linear(128 * 5 * 5, 2048)
        self.fc2 = nn.Linear(2048, 2048)

        # Todo add action-conditional transformations

        # FC layer pre action-conditional transformation
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 128 * 5 * 5)

        # Deconv layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=6, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=6, stride=2, padding=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=8, stride=2, padding=3)

    def forward(self, state, action=None):
        conv1_relu = F.relu(self.conv1(state))
        conv2_relu = F.relu(self.conv2(conv1_relu))
        conv3_relu = F.relu(self.conv3(conv2_relu))
        conv4_relu = F.relu(self.conv4(conv3_relu))

        flatten = conv4_relu.view(-1, 128 * 5 * 5)
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)

        # Todo implement action-conditional transformation

        fc3 = self.fc3(fc2)
        fc4 = self.fc4(fc3)
        deflatten = fc4.view(-1, 128, 5, 5)

        deconv1_relu = F.relu(self.deconv1(deflatten))
        deconv2_relu = F.relu(self.deconv2(deconv1_relu))
        deconv3_relu = F.relu(self.deconv3(deconv2_relu))
        deconv4_relu = F.relu(self.deconv4(deconv3_relu))
        return deconv4_relu


class GenerativeModelMini(nn.Module):
    def __init__(self, dense_size=200, input_channels=1, action_space=4):
        nn.Module.__init__(self)

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, 5, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=3, padding=1)

        # dense layers
        self.state_dense = nn.Linear(32 * 7 * 7, dense_size)
        self.action_dense = nn.Linear(action_space, dense_size)
        self.action_state_dense = nn.Linear(dense_size, 32 * 7 * 7)

        # Deconv layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=3, padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, stride=4, padding=3, output_padding=1)

    def forward(self, state, action=None):

        conv1_relu = F.relu(self.conv1(state))
        conv2_relu = F.relu(self.conv2(conv1_relu))

        state_enc_flat = conv2_relu.view(-1, 32 * 7 * 7)

        # State dense
        state_dense = self.state_dense(state_enc_flat)

        # Action-conditional transformation
        action_dense = self.action_dense(action)
        state_action_tranformation = state_dense*action_dense

        # Output dense
        action_state_dense = self.action_state_dense(state_action_tranformation)

        deflat = action_state_dense.view(-1, 32, 7, 7)

        deconv1_relu = F.relu(self.deconv1(deflat))
        deconv2 = self.deconv2(deconv1_relu)
        return F.tanh(deconv2)

    @property
    def name(self):
        return "AE-action-conditional"


class autoencoder(nn.Module):
    def __init__(self, dense_size=500):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.fc1 = nn.Linear(8 * 6 * 6, dense_size)
        self.fc2 = nn.Linear(dense_size, 8 * 6 * 6)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=0),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x, action=None):
        # Encoding state
        state_enc = self.encoder(x)

        flatten = state_enc.view(-1, 8 * 6 * 6)
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)

        deflat = fc2.view(-1, 8, 6, 6)

        output = self.decoder(deflat)
        return output

    @property
    def name(self):
        return "AE"

