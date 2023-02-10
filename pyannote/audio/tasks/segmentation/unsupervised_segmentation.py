from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Text, Tuple, Union

import pytorch_lightning as pl
import torch
from pyannote.database import Protocol
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
from typing_extensions import Literal

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.tasks import Segmentation

from pyannote.audio import Inference
from pyannote.audio.pipelines.utils import get_devices
from pyannote.audio.utils.powerset import Powerset


class PseudoLabelPostprocess:
    def setup(self, protocol: Protocol, model: Model, teacher: Model) -> None:
        raise NotImplementedError()

    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns (modified x, modified pseudo_y)
        raise NotImplementedError()


class UnsupervisedSegmentation(Segmentation, Task):
    def __init__(
        self,
        protocol: Protocol,
        teacher: Model,  # unsupervised param: model to use to generate truth
        use_pseudolabels: bool = True,  # generate pseudolabels in training mode
        augmentation_teacher: BaseWaveformTransform = None,
        loss_confidence_weighting: Optional[Literal['maxprob', 'probdelta']] = None,
        filter_confidence: Optional[Literal['maxprob', 'probdelta']] = None,
        filter_confidence_threshold: Optional[float] = None,
        filter_confidence_mode: Optional[Literal['absolute', 'quantile']] = None,
        pl_fw_passes: int = 1,  # how many forward passes to average to get the pseudolabels
        pl_postprocess: List[PseudoLabelPostprocess] = None,
        # supervised params
        duration: float = 2.0,
        max_speakers_per_chunk: int = None,
        max_speakers_per_frame: int = None,
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
        """Unsupervised segmentation task.

        Parameters
        ----------
        protocol : Protocol
            pyannote.database protocol
        teacher : Model, optional
            Teacher model to use, will use the Task model if left unspecified. Defaults to None.
        use_pseudolabels : bool, optional
            Whether or not to use pseudolabels for training. Defaults to True.
        augmentation_teacher: BaseWaveformTransform, optional
            What augmentation to apply on the Teacher input. Defaults to None.
        pl_fw_passes : int, optional
            How many forward passes to average to get the pseudolabels. Defaults to 1.
        duration : float, optional
            Chunks duration. Defaults to 2s.
        max_speakers_per_chunk : int, optional
            Maximum number of speakers per chunk (must be at least 2).
            Defaults to estimating it from the training set.
        max_speakers_per_frame : int, optional
            Maximum number of (overlapping) speakers per frame.
            Setting this value to 1 or more enables `powerset multi-class` training.
            Default behavior is to use `multi-label` training.
        warm_up : float or (float, float), optional
            Use that many seconds on the left- and rightmost parts of each chunk
            to warm up the model. While the model does process those left- and right-most
            parts, only the remaining central part of each chunk is used for computing the
            loss during training, and for aggregating scores during inference.
            Defaults to 0. (i.e. no warm-up).
        balance: str, optional
            When provided, training samples are sampled uniformly with respect to that key.
            For instance, setting `balance` to "uri" will make sure that each file will be
            equally represented in the training samples.
        weight: str, optional
            When provided, use this key to as frame-wise weight in loss function.
        batch_size : int, optional
            Number of training samples per batch. Defaults to 32.
        num_workers : int, optional
            Number of workers used for generating training samples.
            Defaults to multiprocessing.cpu_count() // 2.
        pin_memory : bool, optional
            If True, data loaders will copy tensors into CUDA pinned
            memory before returning them. See pytorch documentation
            for more details. Defaults to False.
        augmentation : BaseWaveformTransform, optional
            torch_audiomentations waveform transform, used by dataloader
            during training.
        loss : {"bce", "mse"}, optional
            Permutation-invariant segmentation loss. Defaults to "bce".
        vad_loss : {"bce", "mse"}, optional
            Add voice activity detection loss.
        metric : optional
            Validation metric(s). Can be anything supported by torchmetrics.MetricCollection.
            Defaults to AUROC (area under the ROC curve).
        """

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
            max_speakers_per_chunk=max_speakers_per_chunk,
            max_speakers_per_frame=max_speakers_per_frame,
            balance=balance,
            weight=weight,
            loss=loss,
            vad_loss=vad_loss,
            metric=metric,
        )

        if pl_fw_passes < 1:
            raise ValueError("pl_fw_passes must be strictly positive.")
        if pl_fw_passes > 1:
            raise ValueError(
                "pl_fw_passes for multiple forward passes isn't properly implemented yet "
            )
        if teacher is None:
            raise ValueError(
                "Using the model as its own teacher isn't supported yet. Please pass a teacher model."
            )

        self.teacher = teacher
        teacher_specs = self.teacher.specifications
        self.use_pseudolabels = use_pseudolabels
        self.augmentation_teacher = augmentation_teacher
        self.pl_fw_passes = pl_fw_passes
        self.pl_postprocess = pl_postprocess
        if self.pl_postprocess is not None:
            for pp in self.pl_postprocess:
                pp.setup(self.protocol, self, self.teacher)

        if loss_confidence_weighting not in [None, 'maxprob', 'probdelta']:
            raise ValueError(f"unknown confidence estimation : {loss_confidence_weighting}")
        if loss_confidence_weighting is not None and not teacher_specs.powerset:
            raise ValueError("loss confidence weighting is only compatible with powerset models")
        self.loss_confidence_weighting = loss_confidence_weighting

        if filter_confidence not in [None, 'maxprob', 'probdelta']:
            raise ValueError(f"unknown confidence estimation : {filter_confidence}")
        if filter_confidence is not None and not teacher_specs.powerset:
            raise ValueError("confidence filtering is only compatible with powerset models")
        self.filter_confidence = filter_confidence
        self.filter_confidence_threshold = filter_confidence_threshold
        self.filter_confidence_mode = filter_confidence_mode
        self.discarded_percent = None   # not very clean

        self.teacher.eval()
        if teacher_specs.powerset:
            self._powerset = Powerset(
                len(teacher_specs.classes), teacher_specs.powerset_max_classes
            )
            # TODO: move self._powerset to the correct device (self.model's ? not accessible from there)



    # TODO: use torch.inference_mode() decorator instead of 'with torch.no_grad()' ? It should work + speed up ?
    def get_teacher_outputs_passes(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the teacher output on the input given an augmentation.
        Handles averaging multiple forward passes (each with the augmentation reapplied).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        aug : BaseWaveformTransform
            Input augmentation
        fw_passes : int, optional
            Number of forward passes to apply to get the final output, by default 1

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The tuple :
            - y, the final output, of size (batch_size, num_frames, num_speakers)
            - y_passes, a tensor of all passes, of size (fw_passes, batch_size, num_frames, num_speakers)
        """
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
                model_output = self.teacher(waveforms=teacher_input).detach()
                out_fw_passes.append(model_output)
            # compute mean of forward passes if needed, and round to make pseudolabels
            # TODO: make it work properly by permutating the forward passes so that they "agree"
            stacked_passes = torch.stack(out_fw_passes)
            if fw_passes == 1:
                out = out_fw_passes[0]
            else:
                out = torch.mean(stacked_passes, dim=0)
                raise RuntimeError("Multiple teacher forward passes is not implemented.")
            
            # Convert "out" to multilabel if powerset
            if self.teacher.specifications.powerset:
                one_hot_out = torch.nn.functional.one_hot(
                    torch.argmax(out, dim=-1),
                    self.teacher.specifications.num_powerset_classes,
                ).float()
                out = self._powerset.to_multilabel(one_hot_out)

            out = torch.round(out).type(torch.int8)

        return out, stacked_passes

    def get_teacher_output(
        self, x: torch.Tensor, aug: BaseWaveformTransform, fw_passes: int = 1
    ) -> torch.Tensor:
        out, _ = self.get_teacher_outputs_passes(x, aug, fw_passes)
        return out

    def collate_fn(self, batch, stage="train"):
        collated_X = self.collate_X(batch)
        collated_y = self.collate_y(batch)
        collated_batch = {"X": collated_X, "y": collated_y}

        if stage == "val":
            return collated_batch

        # Generate pseudolabels with teacher if necessary
        if self.use_pseudolabels:
            x = collated_X
            # compute pseudo labels
            pseudo_y, model_out_passes = self.get_teacher_outputs_passes(
                x=x, aug=self.augmentation_teacher, fw_passes=self.pl_fw_passes
            )
            collated_batch["teacher_out"] = model_out_passes[0]
            if self.pl_postprocess is not None:
                processed_x, processed_pseudo_y = collated_batch["X"], pseudo_y
                for pp in self.pl_postprocess:
                    processed_x, processed_pseudo_y = pp.process(
                        processed_pseudo_y, collated_batch["y"], processed_x, model_out_passes
                    )
                collated_batch["X"] = processed_x
                collated_batch["y"] = processed_pseudo_y
            else:
                collated_batch["y"] = pseudo_y

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

    # TODO: tmp: keep or clean
    def training_step_loss_weighting(self, batch, weight):
        weight = super().training_step_loss_weighting(batch, weight)
        if self.loss_confidence_weighting is not None and "teacher_out" in batch:
            out_probas = batch["teacher_out"].exp()
            sorted_out_probas = out_probas.sort(axis=-1)[0]     # sorted from less confident to more confident
            if self.loss_confidence_weighting == "maxprob":
                return weight * sorted_out_probas[:,:,-2:-1]
            elif self.loss_confidence_weighting == "probdelta":
                return weight * (sorted_out_probas[:,:,-1] - sorted_out_probas[:,:,-2])[:,:,None]
        return weight

    # TODO: tmp: keep or clean
    def train__iter__(self):

        for chunk in super().train__iter__():
            if self.filter_confidence is not None:

                _, outs = self.get_teacher_outputs_passes(chunk["X"][None,:,:], None)
                out = outs[0,0] # first fw pass, first (only) batch element
                out_probas = out.exp()
                sorted_out_probas = out_probas.sort(axis=-1)[0]     # sorted from less confident to more confident
                if self.filter_confidence == "maxprob":
                    confidence = sorted_out_probas[:,-2:-1].mean(axis=0)
                elif self.filter_confidence == "probdelta":
                    confidence = (sorted_out_probas[:,-1] - sorted_out_probas[:,-2]).mean(axis=0)


                if self.filter_confidence_mode == "absolute" and confidence > self.filter_confidence_threshold:
                    yield chunk
                elif self.filter_confidence_mode == "quantile" and confidence > self._confidence_quantile_value:
                    yield chunk

            else:
                yield chunk

    def validation_step(self, batch, batch_idx: int):
        super().validation_step(batch, batch_idx)

        if self.filter_confidence_mode == "quantile":
            if batch_idx == 0:
                self._val_confidences = []
            
            _, outs = self.get_teacher_outputs_passes(batch["X"].to(self.teacher.device), None)
            out = outs[0] # first fw pass, first (only) batch element
            out_probas = out.exp()
            sorted_out_probas = out_probas.sort(axis=-1)[0]     # sorted from less confident to more confident
            if self.filter_confidence == "maxprob":
                confidence = sorted_out_probas[:,-2:-1].mean(axis=1)
            elif self.filter_confidence == "probdelta":
                confidence = (sorted_out_probas[:,:,-1] - sorted_out_probas[:,:,-2]).mean(axis=1)
            
            self._val_confidences.append(confidence)


    def validation_epoch_end(self, outputs):
        if self.filter_confidence_mode == "quantile":
            confidences = torch.cat(self._val_confidences)
            self._confidence_quantile_value = confidences.quantile(self.filter_confidence_threshold)
            print(f"confidence quantile estimated : {self._confidence_quantile_value};  out of {confidences.shape[0]} values")



class TeacherEmaUpdate(Callback):
    def __init__(
        self,
        when: Literal["epoch", "batch"] = "epoch",
        update_interval: int = 1,
        update_rate: float = 0.99,
    ):
        """Exponential moving average of weights.

        Parameters
        ----------
        when : Literal['epoch', 'batch'], optional
            When should the update happen, by default "epoch"
        update_interval : int, optional
            Update will happen every 'update_interval' epochs/batches, by default 1
        update_rate : float, optional
            How much to keep of the old weights each update. 0=instant copy, 1=never update weights. By default 0.99.
        """

        super().__init__()

        self.when = when
        if self.when != "epoch" and self.when != "batch":
            raise ValueError(
                "TeacherUpdate 'when' argument can only be 'epoch' or 'batch'"
            )

        if update_rate < 0.0 or update_rate > 1.0:
            raise ValueError(
                f"Illegal update rate value ({update_rate}), it should be in [0.0,1.0]"
            )

        self.update_interval = update_interval
        self.update_rate = update_rate
        self.teacher_weights = None

    @staticmethod
    def get_decayed_weights(
        teacher_w: OrderedDict[str, torch.Tensor],
        student_w: OrderedDict[str, torch.Tensor],
        tau: float,
    ):
        with torch.no_grad():
            return {
                k: teacher_w[k] * tau + student_w[k].to(teacher_w[k].device) * (1 - tau)
                for k in student_w.keys()
            }

    # --- Methods that get called when necessary

    def _setup_initial(self, initial_model_w: OrderedDict[str, torch.Tensor]):
        self.teacher_weights = initial_model_w

    def _compute_teacher_weights(self) -> OrderedDict[str, torch.Tensor]:
        return self.teacher_weights

    def _teacher_update_state(
        self, progress: int, current_model: pl.LightningModule
    ) -> bool:
        if progress % self.update_interval != 0:
            return False

        self.teacher_weights = TeacherEmaUpdate.get_decayed_weights(
            teacher_w=self.teacher_weights,
            student_w=current_model.state_dict(),
            tau=self.update_rate,
        )
        return True

    # --- Callback hooks

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

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        initial_teacher_w = pl_module.task.teacher.state_dict()
        self._setup_initial(initial_teacher_w)

    # --- Generic logic called from hooks

    def update_teacher_and_cache(
        self, progress: int, trainer: pl.Trainer, model: pl.LightningModule
    ):
        # will indicate if teacher cache needs to be updated
        cache_dirty = self._teacher_update_state(progress, model)

        # If a weight has changed, compute updated teacher weights, cache it, and assign it
        if cache_dirty:
            try:
                new_teacher_w = self._compute_teacher_weights()
                model.task.teacher.load_state_dict(new_teacher_w)
            except AttributeError as err:
                raise AttributeError(
                    f"TeacherUpdate callback can't be applied on this model : {err}"
                )


class ConfidenceFilter(PseudoLabelPostprocess):

    def __init__(self, estimator: Literal['maxprob','probdelta'], threshold: float) -> None:
        if estimator not in ['maxprob','probdelta']:
            raise ValueError(f"Estimator {estimator} unknown")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"Wrong threshold value: {threshold}")

        self.estimator = estimator
        self.threshold = threshold

    def setup(self, protocol: Protocol, model: Model, teacher: Model) -> None:
        if not teacher.specifications.powerset:
            raise RuntimeError("Confidence filter currently only supports POWERSET models")
        self.teacher = teacher
    
    def process(self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = ys[0]
        out_probas = out.exp()
        sorted_out_probas = out_probas.sort(axis=-1)[0]     # sorted from less confident to more confident

        if self.estimator == "maxprob":
            # max probability averaged over time
            maxprob = sorted_out_probas[:,:,-1].mean(axis=1)
            filter = maxprob > self.threshold
        elif self.estimator == "probdelta":
            # difference between the two classes of highest probability, averaged over time
            probdelta = (sorted_out_probas[:,:,-1] - sorted_out_probas[:,:,-2]).mean(axis=1)
            filter = probdelta > self.threshold
        
        return x[filter], pseudo_y[filter]


class BrouhahaPseudolabelsFilter(PseudoLabelPostprocess):
    def __init__(
        self,
        model: Model,
        data: Literal["snr", "c50"],
        mode: Literal["threshold", "quantile"],
        threshold: float,
        step_size_percent: float = 0.5,
        setup_protocol: Protocol = None,      # defaults to the protocol the model is trained on
        setup_subset: str = "train",
    ):
        self.model = model
        self.data = data
        if self.data not in ["snr", "c50"]:
            raise ValueError(f"Invalid data type ({data}), use snr or c50")
        self.data_index = 1 if "snr" else 2
        self.mode = mode
        if self.mode not in ["threshold", "quantile"]:
            raise ValueError(f"Invalid mode ({mode}), use threshold or quantile")
        self.threshold = threshold
        self.step_size_percent = step_size_percent

        self.setup_protocol = setup_protocol
        self.setup_subset = setup_subset

        self.is_setup = False

    @staticmethod
    def compute_vad_masked_means(output: torch.Tensor):
        # Means the SNR (and C50 and VAD) by chunk, only accounting frames where there is voice activity.
        # output should be shaped [NB_CHUNKED,NB_FRAMES,NB_OUTPUTS]
        speech_frames = output[:,:, 0] > 0.5   # shape [NB_CHUNKS,NB_FRAMES]
        speech_frames_per_chunk = torch.sum(speech_frames,dim=1).clamp(min=1.0)     # shape [NB_CHUNKS]
        mean_output_per_chunk = torch.sum(speech_frames[:,:,None] * output, dim=2) / speech_frames_per_chunk[:,None]    # shape [NB_CHUNKS,NB_OUTPUTS]
        return mean_output_per_chunk

    def setup(self, protocol: Protocol, model: Model, teacher: Model) -> None:
        if self.is_setup:
            return
        # Only "quantile" mode needs setup
        if not self.mode == "quantile":
            self.is_setup = True
            return

        # Load default protocol if none provided
        if self.setup_protocol is None:
            self.setup_protocol = protocol
        # Do the setup (compute x% quantile on the setup protocol/subset)
        self.compute_quantile()

    def compute_quantile(self):
        files = list(getattr(self.setup_protocol, self.setup_subset)())

        (device,) = get_devices(needs=1)
        inference = Inference(
            self.model,
            device=device,
            duration=self.model.specifications.duration,
            step=self.step_size_percent * self.model.specifications.duration,
            skip_aggregation=True
        )

        output_list = []

        for i, file in enumerate(files):
            output = inference(file)
            output_list.append(torch.from_numpy(output.data))

        outputs = torch.cat(output_list) # shape [NB_CHUNKS,NB_FRAMES,OUTPUTS]
        
        # TODO : use something other than mean ? (max ?) (if changed, dont forget to update accordingly in process)
        mean_output_per_chunk = BrouhahaPseudolabelsFilter.compute_vad_masked_means(outputs)

        self.quantile_value = float(
            torch.quantile(mean_output_per_chunk[:,self.data_index], torch.tensor([self.threshold])).item()
        )
        print(
            f"Computed quantile {self.threshold} value = {self.quantile_value} on {len(files)} files"
        )
        self.is_setup = True


    def process(
        self, pseudo_y: torch.Tensor, y: torch.Tensor, x: torch.Tensor, ys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        mean_output_per_chunk = BrouhahaPseudolabelsFilter.compute_vad_masked_means(outputs)

        if self.mode == "quantile":
            filter = mean_output_per_chunk[:, self.data_index] < self.quantile_value
        elif self.mode == "threshold":
            filter = mean_output_per_chunk[:, self.data_index] < self.threshold

        return x[filter], pseudo_y[filter]
