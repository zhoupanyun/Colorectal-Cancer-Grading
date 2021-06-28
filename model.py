import torch
import torch.nn as nn
import tensorflow as tf



def Extractor(weights_path, input_shape):
    
    base_model = tf.keras.applications.DenseNet121(include_top=False,
                                                   weights=weights_path,
                                                   input_shape=input_shape,
                                                   pooling='avg')
    
    inputs = base_model.input
    
    x = base_model.layers[-3].output
    
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model



class AttentionMIL(nn.Module):

    def __init__(self, feature_size=1024, classes=3):

        super(AttentionMIL, self).__init__()

        self.classes = classes
        self.feature_size = feature_size

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
