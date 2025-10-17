from typing import Optional, Union, List, Callable

import torch
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.models.utils import avalanche_forward
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F
from torch.utils.data import DataLoader


def logmeanexp_previous(x, classes1, classes2, dim=None):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim)
    old_pre = torch.logsumexp(x[:, classes1], dim=1)
    new_pre = torch.logsumexp(x[:, classes2], dim=1)
    pre = torch.stack([old_pre, new_pre], dim=-1)
    return pre


class LODE(SupervisedTemplate):
    """
    Loss Decoupling for Task-Agnostic Continual Learning paper,
    Yan-Shuo Liang and Wu-Jun Li, https://openreview.net/pdf?id=9Oi3YxIBSa
    The code is based on the official implementation:
    https://github.com/liangyanshuo/Loss-Decoupling-for-Task-Agnostic-Continual-Learning
    """

    def __init__(
            self,
            rho,
            model: torch.nn.Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(reduction='sum'),
            mem_size: int = 200,
            batch_size_mem: Optional[int] = None,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = 1,
            device: Union[str, torch.device] = "cpu",
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: Union[
                EvaluationPlugin, Callable[[], EvaluationPlugin]
            ] = default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
    ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int       : Fixed memory size
        :param tau: float          : The temperature used in the KL loss
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param transforms: Callable: Transformations to use for
                                     both the dataset and the buffer data, on
                                     top of already existing
                                     test transformations.
                                     If any supplementary transformations
                                     are applied to the
                                     input data, it will be
                                     overwritten by this argument
        :param train_mb_size: mini-batch size for training.
        :param train_passes: number of training passes.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` experiences and at the end of
            the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations (Default='experience').
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )

        self.rho = rho
        self.seen_classes = []
        self.past_model = None
        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size

        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None

    def inter_cls(self, logits, y, classes1, classes2):
        inter_logits = logmeanexp_previous(logits, classes1, classes2, dim=-1)
        inter_y = torch.ones_like(y)
        return F.cross_entropy(inter_logits, inter_y, reduction='none')

    def intra_cls(self, logits, y, classes):
        mask = torch.zeros_like(logits)
        mask[:, classes] = 1
        logits1 = logits - (1 - mask) * 1e9
        # ipdb.set_trace()
        return F.cross_entropy(logits1, y, reduction='none')

    def _after_training_exp(self, **kwargs):
        self.storage_policy.update(self, **kwargs)
        self.seen_classes.extend(self.experience.classes_in_this_experience)
        super()._after_training_exp(**kwargs)

    def training_epoch(self, **kwargs):
        classes = self.experience.classes_in_this_experience

        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()

            self._after_forward(**kwargs)

            if len(self.storage_policy.buffer) > 0:
                mb_buffer_x, mb_buffer_y, mb_buffer_tid = next(
                    iter(DataLoader(self.storage_policy.buffer,
                                    batch_size=len(self.mb_output),
                                    shuffle=True)))
                mb_buffer_x, mb_buffer_y, mb_buffer_tid = (
                    mb_buffer_x.to(self.device),
                    mb_buffer_y.to(self.device),
                    mb_buffer_tid.to(self.device),
                )

                past_logits = avalanche_forward(
                    self.model, mb_buffer_x, mb_buffer_tid
                )
                # past_logits = self.model(mb_buffer_x, mb_buffer_tid)
                new_inter_cls = self.inter_cls(self.mb_output,
                                               self.mb_y,
                                               self.seen_classes,
                                               self.experience.classes_in_this_experience)

                new_intra_cls = self.intra_cls(self.mb_output, self.mb_y, self.experience.classes_in_this_experience)
                loss = self.rho / self.clock.train_exp_counter * new_inter_cls.mean()

                loss = loss + new_intra_cls.mean()

                loss = loss + F.cross_entropy(past_logits, mb_buffer_y)
                self.loss = loss / 2

            else:
                self.loss = self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)