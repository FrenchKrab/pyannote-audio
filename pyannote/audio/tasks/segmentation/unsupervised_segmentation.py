from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Text, Tuple, Union

import pytorch_lightning as pl
import torch
from pyannote.database import Protocol
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric, MetricCollection
from typing_extensions import Literal

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task, ValDataset
from pyannote.audio.tasks import Segmentation
from pyannote.audio.torchmetrics.audio.diarization_error_rate import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    SpeakerConfusionRate,
)
from pyannote.audio.torchmetrics.functional.audio.diarization_error_rate import (
    diarization_error_rate,
)
from pyannote.audio.utils.permutation import permutate


class PseudoLabelPostprocess:
    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class UnsupervisedSegmentation(Segmentation, Task):
    def __init__(
        self,
        teacher: Model,  # unsupervised param: model to use to generate truth
        val_model: Model,  # model used to generate truths during validation
        protocol: Protocol,
        fake_in_train=True,  # generate fake truth in training mode
        fake_in_val=True,  # generate fake truth in val mode
        augmentation_model: BaseWaveformTransform = None,
        pl_postprocess: Union[
            PseudoLabelPostprocess, List[PseudoLabelPostprocess]
        ] = None,
        pl_fw_passes: int = 1,  # how many forward passes to average to get the pseudolabels
        val_fw_passes: int = 1,  # how many forward passes to average to get the validation uncertainty
        val_augmentation: BaseWaveformTransform = None,
        # supervised params
        duration: float = 2.0,
        max_num_speakers: int = None,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
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
            balance=balance,
            weight=weight,
            loss=loss,
            vad_loss=vad_loss,
            metric=metric,
        )

        self.teacher = teacher
        self.val_model = val_model
        self.fake_in_train = fake_in_train
        self.fake_in_val = fake_in_val
        self.augmentation_model = augmentation_model
        self.pl_fw_passes = pl_fw_passes
        self.val_fw_passes = val_fw_passes
        self.val_augmentation = val_augmentation

        if fake_in_val:
            raise ValueError(
                "Fake validation (fake_in_val) is not supported, please disable it."
            )

        if pl_fw_passes > 1 and augmentation_model is None:
            raise ValueError(
                "There is no reason to do multiple forward passes to generate pseudolabels if there is no augmentation applied to the input."
            )

        if val_augmentation is None and val_fw_passes > 1:
            raise ValueError(
                f"You need to specify a validation augmentation if you want to do {val_fw_passes} val forward passes."
            )

        if isinstance(pl_postprocess, PseudoLabelPostprocess):
            self.pl_postprocess = [pl_postprocess]
        else:
            self.pl_postprocess = pl_postprocess

        self.teacher.eval()

    def setup_loss_func(self):
        # very strange and bad code placement, to replace/move (but it works)
        self.model.teacher_metrics = MetricCollection(
            [
                DiarizationErrorRate(),
                SpeakerConfusionRate(),
                MissedDetectionRate(),
                FalseAlarmRate(),
            ],
            prefix=f"{self.logging_prefix}-Teacher",
        )
        self.model.teacher_metrics.to(self.model.device)

        self.model.val_model_metrics = MetricCollection(
            [
                DiarizationErrorRate(),
                SpeakerConfusionRate(),
                MissedDetectionRate(),
                FalseAlarmRate(),
            ],
            prefix=f"{self.logging_prefix}-ValModel",
        )
        self.model.val_model_metrics.to(self.model.device)

        return super().setup_loss_func()

    def get_teacher_outputs(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ):
        out_fw_passes = []
        with torch.no_grad():  # grad causes problems when crossing process boundaries
            # for each forward pass
            for i in range(fw_passes):
                teacher_input = x

                # Augment input if necessary
                if aug is not None:
                    augmented = aug(
                        samples=teacher_input,
                        sample_rate=self.model.hparams.sample_rate,
                    )
                    teacher_input = augmented.samples
                # Compute pseudolabels and detach to avoid "memory leaks"
                pl = self.teacher(waveforms=teacher_input).detach()
                out_fw_passes.append(pl)
            # compute mean of forward passes
            out = torch.mean(torch.stack(out_fw_passes), dim=0)
            out = torch.round(out).type(torch.int8)

        return out, torch.stack(out_fw_passes)

    def get_teacher_output(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ):
        out, _ = self.get_teacher_outputs(x, aug, fw_passes)
        return out

    def use_pseudolabels(self, stage: Literal["train", "val"]):
        return (stage == "train" and self.fake_in_train) or (
            stage == "val" and self.fake_in_val
        )

    def collate_fn(self, batch, stage="train"):

        collated_X = self.collate_X(batch)
        collated_y = self.collate_y(batch)
        collated_batch = {"X": collated_X, "y": collated_y}

        if stage != "train":
            raise RuntimeError(f"Unexpected stage in collate_fn (stage={stage})")

        # Generate annotations y with teacher if necessary
        if self.use_pseudolabels("train"):
            x = collated_X
            # compute pseudo labels
            pseudo_y, computed_ys = self.get_teacher_outputs(
                x=x, aug=self.augmentation_model, fw_passes=self.pl_fw_passes
            )

            # apply post processing to pseudo labels and x
            y = collated_y
            if self.pl_postprocess is not None:
                for pp in self.pl_postprocess:
                    pseudo_y, x = pp.process(pseudo_y, y, x, computed_ys)

            collated_batch["y"] = pseudo_y
            collated_batch["X"] = x

        # Augment x/pseudo y if an augmentation is specified
        if self.augmentation is not None:
            augmented = self.augmentation(
                samples=collated_batch["X"],
                sample_rate=self.model.hparams.sample_rate,
                targets=collated_batch["y"].unsqueeze(1),
            )
            collated_batch["X"] = augmented.samples
            collated_batch["y"] = augmented.targets.squeeze(1)

        return collated_batch

    def collate_fn_val(self, batch):
        collated_X = self.collate_X(batch)
        collated_y = self.collate_y(batch)

        collated_batch = {"X": collated_X, "y": collated_y}

        # Generate annotations y with teacher if they are not provided
        # TODO: reimplement or discard fake validation
        if self.use_pseudolabels("val"):
            pass

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

    def get_summed_gradients(
        self, model_hyp: LightningModule, model_ref: LightningModule, x: torch.Tensor
    ):
        torch.set_grad_enabled(True)
        model_hyp.train()
        model_hyp.zero_grad()
        model_ref.zero_grad()
        prediction = model_hyp(x)
        ref_prediction = model_ref(x.to(model_ref.device)).to(model_hyp.device)
        ref_prediction = (ref_prediction > 0.5).float()
        permutated_prediction, _ = permutate(ref_prediction, prediction)
        loss = self.segmentation_loss(permutated_prediction, ref_prediction)
        loss.backward()

        summed_grad_norm = 0
        for p in model_hyp.parameters():
            summed_grad_norm += torch.norm(p.grad)

        model_hyp.zero_grad()
        model_ref.zero_grad()
        model_hyp.eval()
        torch.set_grad_enabled(False)
        return summed_grad_norm, prediction, loss, ref_prediction

    def training_step(self, batch, batch_idx: int):
        loss = super().training_step(batch, batch_idx)

        sgn = 0
        for p in self.model.parameters():
            sgn += torch.norm(p.grad)
        self.model.log(
            f"{self.logging_prefix}Gradients",
            sgn,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int):
        super().validation_step(batch, batch_idx)

        X = batch["X"]

        (
            val_model_sgn,
            prediction,
            loss_val_model,
            val_model_prediction,
        ) = self.get_summed_gradients(self.model, self.val_model, X)
        self.model.log(
            f"{self.logging_prefix}GradientsValModel",
            val_model_sgn,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.model.log(
            f"{self.logging_prefix}LossValModel",
            loss_val_model,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        outputs = []
        if self.val_augmentation is not None:
            for i in range(self.val_fw_passes):
                augmented = self.val_augmentation(
                    samples=X, sample_rate=self.model.hparams.sample_rate
                )
                pred = self.model(augmented.samples)
                outputs.append(pred)
        else:
            outputs.append(prediction)
        outputs = torch.stack(outputs)

        # Compute ang log uncertainty
        std = torch.std(outputs, dim=0)
        self.model.log(
            f"{self.logging_prefix}Uncertainty",
            std.flatten(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Compute and log confidence
        outputs_mean = torch.mean(outputs, dim=0)
        CONFIDENCE_CENTER = 0.5
        confidence = 1 - torch.abs(CONFIDENCE_CENTER - outputs_mean) / 0.5
        self.model.log(
            f"{self.logging_prefix}Confidence",
            confidence.flatten(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        CONFIDENCE_CENTER = 0.25
        confidence = 1 - torch.abs(CONFIDENCE_CENTER - outputs_mean) / torch.where(
            outputs_mean > CONFIDENCE_CENTER, 1 - CONFIDENCE_CENTER, CONFIDENCE_CENTER
        )

        self.model.log(
            f"{self.logging_prefix}Confidence25",
            confidence.flatten(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        _, num_frames, _ = prediction.shape
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        preds = prediction[:, warm_up_left : num_frames - warm_up_right : 10]

        teacher_prediction = self.teacher(X.to(self.teacher.device)).to(
            val_model_prediction.device
        )
        teacher_prediction = (teacher_prediction > 0.5).float()
        target_teacher = teacher_prediction[
            :, warm_up_left : num_frames - warm_up_right : 10
        ]
        target_val_model = val_model_prediction[
            :, warm_up_left : num_frames - warm_up_right : 10
        ]

        self.model.teacher_metrics(
            torch.transpose(preds, 1, 2),
            torch.transpose(target_teacher, 1, 2),
        )
        self.model.val_model_metrics(
            torch.transpose(preds, 1, 2),
            torch.transpose(target_val_model, 1, 2),
        )

        self.model.log_dict(
            self.model.teacher_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.model.log_dict(
            self.model.val_model_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


def _compute_ders(
    pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor
) -> Tuple[torch.Tensor]:
    batch_size = pseudo_y.shape[0]
    ders = torch.zeros(batch_size)

    tm_pseudo_y = pseudo_y.swapaxes(1, 2)
    tm_true_y = y.swapaxes(1, 2)
    for i in range(batch_size):
        ders[i] = diarization_error_rate(
            tm_pseudo_y[i][None, :, :], tm_true_y[i][None, :, :]
        )

    return ders


class DiscardConfidence(PseudoLabelPostprocess):
    def __init__(self, threshold=0.75) -> None:
        super().__init__()
        self.threshold = threshold

    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ys is (fw_passes, batch_size, num_frames, num_speakers)-shaped
        fw_passes, batch_size, num_frames, num_speakers = ys.shape

        # ys_confidence is of shape (batch_size)
        ys_confidence = torch.mean(
            torch.abs(
                torch.mean(ys, dim=0).reshape(batch_size, num_frames * num_speakers)
                - 0.5
            )
            / 0.5,
            dim=1,
        )

        filter = ys_confidence > self.threshold
        return pseudo_y[filter], x[filter]


class DiscardPercentDer(PseudoLabelPostprocess):
    def __init__(self, ratio_to_discard: float = 0.1) -> None:
        self.ratio_to_discard = ratio_to_discard

    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = pseudo_y.shape[0]
        ders = _compute_ders(pseudo_y, y, x)
        sorted_ders, sorted_indices = torch.sort(ders)

        to_discard_count = min(
            batch_size, max(1, round(batch_size * self.ratio_to_discard))
        )
        pseudo_y = pseudo_y[sorted_indices][:-to_discard_count, :, :]
        x = x[sorted_indices][:-to_discard_count, :, :]

        return pseudo_y, x


class DiscardThresholdDer(PseudoLabelPostprocess):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ders = _compute_ders(pseudo_y, y, x)

        filter = torch.where(ders < self.threshold)
        pseudo_y = pseudo_y[filter]
        x = x[filter]

        return pseudo_y, x


class TeacherUpdate(Callback):
    def __init__(self, when: Literal["epoch", "batch"]) -> None:
        """Abstract class for teacher update callback.
        _teacher_update_state, _compute_teacher_weights and _setup_initial needs to be overriden.

        Parameters
        ----------
        when : Literal['epoch', 'batch'], optional
            When the update is applied.
        """

        super().__init__()
        self.initialized = False

        self.when = when
        if self.when != "epoch" and self.when != "batch":
            raise ValueError(
                "TeacherUpdate 'when' argument can only be 'epoch' or 'batch'"
            )

        # The "real" teacher weights to be used by the task.
        # TODO: determine if useless ?
        self.cache: OrderedDict[str, torch.Tensor] = None

    def _teacher_update_state(
        self, progress: int, current_model: pl.LightningModule
    ) -> bool:
        """Called every time the teacher might need to be updated.
        Parameters
        ----------
        progress : int
            Describes how far we are in training (epoch number or batch number)
        current_model : pl.LightningModule
            Current model being trained.
        Returns
        -------
        bool
            Whether or not the internal state has been updated.
            If true, the 'teacher' attribute of the model will be updated with
            newly computed weights from _compute_teacher_weights.
        """
        raise NotImplementedError()

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        """Computes the new teacher weights from the internal state.
        Returns
        -------
        OrderedDict[str, torch.Tensor]
            New teacher weights
        """
        raise NotImplementedError()

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        """Called once before training. Useful to setup the initial internal state.
        Parameters
        ----------
        initial_model_w : OrderedDict[str, torch.Tensor]
            Current (initial) model weights.
        """
        raise NotImplementedError()

    def update_teacher_and_cache(
        self, progress: int, trainer: pl.Trainer, model: pl.LightningModule
    ):
        # will indicate if teacher cache needs to be updated
        cache_dirty = self._teacher_update_state(progress, model)

        # If a weight has changed, compute updated teacher weights, cache it, and assign it
        if cache_dirty:
            try:
                new_teacher_w = self._compute_teacher_weights()
                self.cache = new_teacher_w
                model.task.teacher.load_state_dict(new_teacher_w)
            except AttributeError as err:
                print(f"TeacherUpdate callback can't be applied on this model : {err}")

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
            self.update_teacher_and_cache(batch_idx, trainer, pl_module)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.when == "epoch":
            self.update_teacher_and_cache(trainer.current_epoch, trainer, pl_module)

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Only initialize once
        if self.initialized:
            return
        self.initialized = True

        initial_teacher_w = pl_module.task.teacher.state_dict()
        self._setup_initial(initial_teacher_w)
        self.cache = initial_teacher_w

    @staticmethod
    def get_averaged_weights(
        weights: List[OrderedDict[str, torch.Tensor]]
    ) -> OrderedDict[str, torch.Tensor]:
        return {
            k: torch.mean(torch.stack([w[k] for w in weights]), dim=0)
            for k in weights[0]
        }


class TeacherQueueUpdate(TeacherUpdate):
    def __init__(
        self,
        when: Literal["epoch", "batch"] = "epoch",
        average_of: int = 1,
        update_interval: int = 1,
    ):
        super().__init__(when=when)

        self.average_of = average_of
        self.update_interval = update_interval
        self.weights_queue: List[OrderedDict[str, torch.Tensor]] = None
        pass

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        self.weights_queue = []

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        return TeacherUpdate.get_averaged_weights(self.weights_queue)

    def _teacher_update_state(self, progress: int, model: pl.LightningModule) -> bool:
        if progress % self.update_interval != 0:
            return False

        # Get new teacher "candidate" (from decayed weights) and enqueue it in the teacher history
        teacher_candidate_w = model.state_dict()
        self.enqueue_to_team(teacher_candidate_w)
        return True

    def enqueue_to_team(self, teacher: OrderedDict[str, torch.Tensor]):
        self.weights_queue.append(teacher)
        if len(self.weights_queue) > self.average_of:
            self.weights_queue.pop(0)


class TeacherTeamUpdate(TeacherUpdate):
    def __init__(
        self,
        when: Literal["epoch", "batch"],
        update_interval: List[int],
        weight_update_rate: List[float],
    ):
        """TeacherTeamUpdate callback.
        Keeps in memory a "team" of teachers weights that gets updated at different interval and at a different rate (EMA).
        The final teacher is the average of all these weights.

        Parameters
        ----------
        when : Literal['epoch', 'batch'], optional
            When the update is applied, by default 'epoch'
        update_interval : List[int]
            How frequently the teacher(s) are updated.
        weight_update_rate : List[float]
            How much of the old weights to keep: 0.0 means instant update, 1.0 means never update.
        """
        super().__init__(when=when)

        if len(update_interval) != len(weight_update_rate):
            raise ValueError(
                "update_interval should be the same size as weight_update_rate"
            )

        self.update_interval = update_interval
        self.weight_update_rate = weight_update_rate

        # The weights of the team of teachers that will be averaged
        self.team_weights: List[OrderedDict[str, torch.Tensor]] = []

    def get_decayed_weights(
        self,
        teacher_w: OrderedDict[str, torch.Tensor],
        student_w: OrderedDict[str, torch.Tensor],
        tau: float,
    ):
        with torch.no_grad():
            return {
                k: teacher_w[k] * tau + student_w[k].to(teacher_w[k].device) * (1 - tau)
                for k in student_w.keys()
            }

    def get_update_rate(self, index) -> float:
        return self.weight_update_rate[index]

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        self.team_weights = [initial_model_w] * len(self.update_interval)

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        return TeacherUpdate.get_averaged_weights(self.team_weights)

    def _teacher_update_state(self, progress: int, model: pl.LightningModule) -> bool:
        cache_dirty = False
        # Check for each member of the ensemble if it needs to be updated
        for i in range(len(self.update_interval)):
            interval = self.update_interval[i]
            if interval == 0 or progress % interval != 0:
                print(f"NOT updated teacher {i}")
                continue
            print(f"updating teacher {i}")

            new_teacher_i_w = self.get_decayed_weights(
                teacher_w=self.team_weights[i],
                student_w=model.state_dict(),
                tau=self.get_update_rate(i),
            )
            self.team_weights[i] = new_teacher_i_w
            cache_dirty = True
        return cache_dirty


def get_decayed_weights(
    teacher_w: OrderedDict[str, torch.Tensor],
    student_w: OrderedDict[str, torch.Tensor],
    tau: float,
):
    with torch.no_grad():
        return {
            k: teacher_w[k].to(student_w[k].device) * tau + student_w[k] * (1 - tau)
            for k in student_w.keys()
        }


class AlmostMetaPseudoLabel(Callback):
    def __init__(
        self, teacher: Model, loss, strategy: Literal["ema"] = "ema", ema_speed=0.15
    ) -> None:
        super().__init__()
        # params
        self.teacher = teacher
        self.loss_fn = loss
        self.strategy = strategy
        self.ema_speed = ema_speed

        if strategy != "ema":
            raise ValueError("strategy has to be one of : 'ema'")

        # work variables
        self.old_loss = None
        self.x = None
        self.targets = None

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        labelled_data = next(iter(self.teacher.train_dataloader()))
        self.x = labelled_data["X"].to(pl_module.device)
        self.targets = labelled_data["y"].to(pl_module.device)
        with torch.no_grad():
            s_output = pl_module(self.x).detach()

        s_output, _ = permutate(self.targets, s_output)
        self.old_loss = pl_module.task.segmentation_loss(
            s_output.detach(), self.targets
        )

        return super().on_train_batch_start(
            trainer, pl_module, batch, batch_idx, unused
        )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        with torch.no_grad():
            s_output = pl_module(self.x).detach()
        s_output, _ = permutate(self.targets, s_output)
        new_loss = pl_module.task.segmentation_loss(s_output.detach(), self.targets)

        # compute the "h" factor of the mpl algorithm
        h = new_loss - self.old_loss

        if self.strategy == "ema":
            # low h -> lower tau, h >=0 -> tau=1
            tau = torch.min(
                torch.exp(h) * self.ema_speed + (1 - self.ema_speed),
                torch.ones_like(h) * 1.0,
            )
            new_teacher_w = get_decayed_weights(
                teacher_w=self.teacher.state_dict(),
                student_w=pl_module.state_dict(),
                tau=tau,
            )
            self.teacher.load_state_dict(new_teacher_w)

        # print(f"h={h}  ->  tau={tau}")

        return super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, unused
        )
