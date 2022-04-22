from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Text, Tuple, Union

import pytorch_lightning as pl
import torch
from pyannote.database import Protocol
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
from typing_extensions import Literal

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task, ValDataset
from pyannote.audio.tasks import Segmentation


class UnsupervisedSegmentation(Segmentation, Task):
    OVERLAP_DEFAULTS = {"probability": 0.0, "snr_min": 0.0, "snr_max": 10.0}

    def __init__(
        self,
        model: Model,  # unsupervised param: model to use to generate truth
        protocol: Protocol,
        fake_in_train=True,  # generate fake truth in training mode
        fake_in_val=True,  # generate fake truth in val mode
        augmentation_model: BaseWaveformTransform = None,
        pl_fw_passes: int = 1,  # how many forward passes to average to get the pseudolabels
        # supervised params
        duration: float = 2.0,
        max_num_speakers: int = None,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        overlap: dict = OVERLAP_DEFAULTS,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        loss: Literal["bce", "mse"] = "bce",
        vad_loss: Literal["bce", "mse"] = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
    ):
        super().__init__(
            # Mixin params
            protocol,
            duration=duration,
            warm_up=warm_up,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            # Segmentation params
            max_num_speakers=max_num_speakers,
            overlap=overlap,
            balance=balance,
            weight=weight,
            loss=loss,
            vad_loss=vad_loss,
            metric=metric,
        )

        self.teacher = model
        self.fake_in_train = fake_in_train
        self.fake_in_val = fake_in_val
        self.augmentation_model = augmentation_model
        self.pl_fw_passes = pl_fw_passes

        if pl_fw_passes > 1 and augmentation_model is None:
            raise ValueError(
                "There is no reason to do multiple forward passes to generate pseudolabels if there is no augmentation applied to the input."
            )

        self.teacher.eval()

    def get_teacher_output(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ):
        out_fw_passes = []
        with torch.no_grad():  # grad causes problems when crossing process boundaries
            for i in range(fw_passes):
                teacher_input = x

                if aug is not None:
                    teacher_input = aug(
                        teacher_input, sample_rate=self.model.hparams.sample_rate
                    )

                    # detach is necessary to avoid memory leaks
                    pl = self.teacher(waveforms=x).detach()
                out_fw_passes.append(pl)

            out = torch.mean(torch.stack(out_fw_passes), dim=0)
            out = torch.round(out).type(torch.int8)

        return out

    def use_pseudolabels(self, stage: Literal["train", "val"]):
        return (stage == "train" and self.fake_in_train) or (
            stage == "val" and self.fake_in_val
        )

    def collate_fn(self, batch):
        collated_batch = default_collate(batch)

        # Generate annotations y with teacher if they are not provided
        if self.use_pseudolabels("train"):
            x = collated_batch["X"]

            pseudo_y = self.get_teacher_output(
                x=x, aug=self.augmentation_model, fw_passes=self.pl_fw_passes
            )

            collated_batch["y"] = pseudo_y
            collated_batch["X"] = x

        if self.augmentation is not None:
            collated_batch["X"] = self.augmentation(
                collated_batch["X"], sample_rate=self.model.hparams.sample_rate
            )
        return collated_batch

    def collate_fn_val(self, batch):
        collated_batch = default_collate(batch)

        # Generate annotations y with teacher if they are not provided
        if self.use_pseudolabels("val"):
            teacher_input = collated_batch["X"]
            collated_batch["y"] = self.get_teacher_output(x=teacher_input)
            collated_batch = torch.round(collated_batch).type(torch.int8)

        return collated_batch

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.has_validation:
            return DataLoader(
                ValDataset(self),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
                collate_fn=self.collate_fn_val,
            )
        else:
            return None


class TeacherUpdate(Callback):
    def __init__(
        self,
        when: Literal["epoch", "batch"] = "epoch",
        update_interval: int = 1,
        weight_update_rate: float = 0.0,
        average_of: int = 1,
    ):
        self.when = when
        self.update_interval = update_interval
        self.weight_update_rate = weight_update_rate
        self.average_of = average_of

        self.last_weights: List[OrderedDict[str, torch.Tensor]] = []
        self.teacher_weights_cache: OrderedDict[str, torch.Tensor] = None

    def enqueue_teacher(self, teacher: OrderedDict[str, torch.Tensor]):
        if len(self.last_weights) >= self.average_of:
            self.last_weights.pop(0)
        self.last_weights.append(teacher)

    def get_updated_weights(
        self,
        teacher_w: OrderedDict[str, torch.Tensor],
        student_w: OrderedDict[str, torch.Tensor],
    ):
        with torch.no_grad():
            return {
                k: teacher_w[k].to("cpu") * self.weight_update_rate
                + student_w[k].to("cpu") * (1 - self.weight_update_rate)
                for k in student_w
            }

    def compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        if len(self.last_weights) == 1:
            return self.last_weights[0]
        else:
            with torch.no_grad():
                new_w = {
                    k: torch.mean(torch.stack([w[k] for w in self.last_weights]), dim=0)
                    for k in self.last_weights[0]
                }
                return new_w

    def try_update_teacher(
        self, progress: int, trainer: pl.Trainer, model: pl.LightningModule
    ):
        if (
            self.update_interval > 0
            and self.weight_update_rate < 1.0
            and progress % self.update_interval == 0
        ):
            try:
                # Get new teacher "candidate" (from decayed weights) and enqueue it in the teacher history
                teacher_candidate_w = self.get_updated_weights(
                    self.teacher_weights_cache, model.state_dict()
                )
                self.enqueue_teacher(teacher_candidate_w)

                # Compute the real new teacher weights, cache it, and assign it
                new_teacher_w = self.compute_teacher_weights()
                self.teacher_weights_cache = new_teacher_w
                model.task.teacher.load_state_dict(new_teacher_w)

            except AttributeError as err:
                print(f"TeacherUpdate callback can't be applied on this model : {err}")

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if len(self.last_weights) == 0:
            self.last_weights.append(pl_module.task.teacher.state_dict())
            self.teacher_weights_cache = pl_module.task.teacher.state_dict()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if self.when == "batch":
            self.try_update_teacher(batch_idx, trainer, pl_module)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.when == "epoch":
            self.try_update_teacher(trainer.current_epoch, trainer, pl_module)
