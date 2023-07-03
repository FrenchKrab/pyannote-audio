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

from typing import Union, Optional
import torch
from pyannote.audio.core.task import Task
from pyannote.audio.models.segmentation import PyanNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict


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


class PyanNetConfidence(Model):
    """PyanNet variant with additionnal output class(es) reserved for confidence prediction.
    The additional confidence classes use a sigmoid activation, apart from that the network behaves according to
    its specifications just like a regular PyanNet model.
    """

    CONFIDENCE_DEFAULTS = {
        "num_classes": 1,
    }

    def __init__(
        self,
        confidence: dict = None,
        sincnet: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Union[Task, None] = None,
    ):
        # ------------- COPY PASTE FROM PYANNET

        # TODO/ make this cleaner :/
        super().__init__(
            sample_rate=sample_rate, num_channels=num_channels, task=task
        )
        
        sincnet = merge_dict(PyanNet.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate
        lstm = merge_dict(PyanNet.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(PyanNet.LINEAR_DEFAULTS, linear)
        confidence = merge_dict(
            PyanNetConfidence.CONFIDENCE_DEFAULTS, confidence
        )  # not copypasted
        self.save_hyperparameters(
            "confidence", "sincnet", "lstm", "linear"
        )  # not copypasted

        self.sincnet = SincNet(**self.hparams.sincnet)

        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(60, **multi_layer_lstm)

        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        60
                        if i == 0
                        else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
                        **one_layer_lstm,
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.sincnet(waveforms)

        if self.hparams.lstm["monolithic"]:
            outputs, _ = self.lstm(
                rearrange(outputs, "batch feature frame -> batch frame feature")
            )
        else:
            outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
    

    # ------------- CONFIDENCE RELATED

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        if self.specifications.powerset:
            out_features = self.specifications.num_powerset_classes
        else:
            out_features = len(self.specifications.classes)

        self.classifier = nn.Linear(in_features, out_features)
        self.activation = self.default_activation()





        confidence_classes_count = self.hparams.confidence["num_classes"]

        # let the super().build() compute the number of in/out features
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features

        self.classifier = torch.nn.Linear(
            in_features, out_features + confidence_classes_count
        )
        self.activation = ConfidenceSpecificActivation(
            self.default_activation(), confidence_classes_count
        )
