import numpy as np
from sklearn.neighbors import KernelDensity


def score_learned_distribution(samples, truth):
    """ Score the learned parameter distribution.

    Example of how to score the inferred parameter distribution for
    a given orbital parameter as per Grant and Blundell 2021. The
    scoring used is the natural logarithm of the posterior density
    evaluated at the true parameter value.

    :param samples: np.array (1d), samples of a given orbital
    parameter from mcmc or a similar inference method.
    :param truth: float, true orbital parameter value.
    :return: float, score.
    """
    # Auto define bandwidth using Silverman's rule fo thumb.
    bandwidth = 0.9 * min(np.std(samples), np.subtract(
        *np.percentile(samples, [75, 25])) / 1.34) \
        * len(samples) ** (-1 / 5)

    # Kernel density estimate for pdf of param samples.
    kd = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kd.fit(samples.reshape(-1, 1))

    # Log density of truth.
    return kd.score_samples(np.array([truth]).reshape(-1, 1))[0]
