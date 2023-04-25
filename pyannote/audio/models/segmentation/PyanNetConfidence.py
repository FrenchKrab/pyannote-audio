# MIT License
#
# Copyright (c) 2023 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# AUTHORS
# Alexis PLAQUET

from typing import Union
import torch
from pyannote.audio.core.task import Task
from pyannote.audio.models.segmentation import PyanNet


class ConfidenceSpecificActivation(torch.nn.Module):
    def __init__(self, activation: torch.nn.Module, confidence_classes: int = 1):
        super().__init__()
        self.activation = activation
        self.confidence_classes = confidence_classes

    def forward(self, x):
        y_regular = self.activation(x[:, :, : -self.confidence_classes])
        y_confidence = torch.nn.functional.sigmoid(x[:, :, -self.confidence_classes :])
        y = torch.cat([y_regular, y_confidence], dim=-1)
        return y

    def string(self):
        return f"Confidence-specific activation, default activation={self.activation}; {self.confidence_classes} confidences"


class PyanNetConfidence(PyanNet):
    """PyanNet variant with additionnal output class(es) reserved for confidence prediction.
    The additional confidence classes use a sigmoid activation, apart from that the network behaves according to
    its specifications just like a regular PyanNet model.
    """

    def __init__(
        self,
        confidence_classes: int = 1,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Union[Task, None] = None,
    ):
        super().__init__(sincnet, lstm, linear, sample_rate, num_channels, task)

        self.confidence = {
            "class_count": confidence_classes,
        }
        self.save_hyperparameters("confidence")

    def build(self):
        super().build()

        # let the super().build() compute the number of in/out features
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features
        self.classifier = torch.nn.Linear(
            in_features, out_features + self.confidence_classes
        )
        self.activation = ConfidenceSpecificActivation(
            self.default_activation(), self.confidence_classes
        )
