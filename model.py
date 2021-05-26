import torch
import torch.nn as nn



class AttentionMIL(nn.Module):

    def __init__(self, feature_size=1024, classes=3):

        super(AttentionMIL, self).__init__()

        self.classes = classes
        self.feature_size = input_shape

        self.extractor = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, self.classes),
        )

    def forward(self, input):

        # feature-extraction
        f = self.extractor(input)

        # attention-mechanism
        a = self.attention(f)
        a = torch.transpose(a, 1, 0)
        a = nn.Softmax(dim=1)(a)

        # aggregation
        m = torch.mm(a, f)

        # classification
        y = self.classifier(m)

        return y, a
