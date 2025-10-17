################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-01-2021                                                             #
# Author(s): Vincenzo Lomonaco, Lorenzo Pellegrini                             #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from torch.nn import Module
from typing import List, Optional
from torch import Tensor

from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric


class Fraction(Metric[float]):
    """
    Standalone Multiply-and-accumulate metric. Provides a lower bound of the
    computational cost of a model in a hardware-independent way by
    computing the number of multiplications. Currently supports only
    Linear or Conv2d modules. Other operations are ignored.
    """

    def __init__(self):
        """
        Creates an instance of the MAC metric.
        """
        self.fraction = 0.0

    def update(self, fraction: float):
        """
        Computes the MAC metric.

        :param model: current model.
        :param dummy_input: A tensor of the correct size to feed as input
            to model. It includes batch size
        :return: MAC metric.
        """

        self.fraction = fraction

    def result(self) -> Optional[float]:
        """
        Return the number of MAC operations as computed in the previous call
        to the `update` method.

        :return: The number of MAC operations or None if `update` has not been
            called yet.
        """
        return self.fraction

    def reset(self):
        self.fraction = 0.0



class FractionPluginMetric(GenericPluginMetric):
    def __init__(self, reset_at, emit_at, mode):
        self._fraction = Fraction()

        super(FractionPluginMetric, self).__init__(
            self._fraction, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def update(self, strategy):
        self._fraction.update(strategy.fraction)


class MinibatchFraction(FractionPluginMetric):
    """
    The minibatch MAC metric.
    This plugin metric only works at training time.

    This metric computes the MAC over 1 pattern
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochMAC` instead.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchMAC metric.
        """
        super(MinibatchFraction, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Fraction_MB"


class EpochFraction(FractionPluginMetric):
    """
    The Fraction at the end of each epoch computed on a
    single pattern.
    This plugin metric only works at training time.

    The Fraction will be logged after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochFraction metric.
        """
        super(EpochFraction, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "Fraction_Epoch"


class ExperienceFraction(FractionPluginMetric):
    """
    At the end of each experience, this metric reports the
    Fraction computed on a single pattern.
    This plugin metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceFraction metric
        """
        super(ExperienceFraction, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "Fraction_Exp"

class StreamFraction(FractionPluginMetric):
    """
    At the end of each experience, this metric reports the
    Fraction computed on a single pattern.
    This plugin metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceFraction metric
        """
        super(StreamFraction, self).__init__(
            reset_at="never", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "Fraction_Stream"



def fraction_metrics(
    *, minibatch=False, epoch=False, experience=False, stream=False,
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the Fraction after each iteration at training time.
    :param epoch: If True, will return a metric able to log
        the Fraction after each epoch at training time.
    :param experience: If True, will return a metric able to log
        the Fraction after each eval experience.

    :return: A list of plugin metrics.
    """

    metrics: List[PluginMetric] = []
    if minibatch:
        metrics.append(MinibatchFraction())

    if epoch:
        metrics.append(EpochFraction())

    if experience:
        metrics.append(ExperienceFraction())

    if stream:
        metrics.append(StreamFraction())
    return metrics


__all__ = ["Fraction", "MinibatchFraction", "EpochFraction", "ExperienceFraction", "StreamFraction","fraction_metrics"]
