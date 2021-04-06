import torch.nn as nn
from .modules import VGG_FeatureExtractor, BidirectionalLSTM


class OCR(nn.Module):
    def __init__(self, config):
        super(OCR, self).__init__()
        input_channel = config["input_channel"]
        output_channel = config["output_channel"]
        hidden_size = config["hidden_size"]
        num_class = len(config["character"]) + 1
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(
            input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output,
                              hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
