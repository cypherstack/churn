"""
Estimate critical values of the harmonic mean of d-tuples of random variables (especially Beta random variables)
"""
from multiprocessing import Pool, cpu_count
from itertools import repeat
import numpy as np
from scipy import stats
from tqdm import tqdm

DEFAULT_FILENAME: str = "output.csv"
DEFAULT_MODE: str = "w"
DEFAULT_INIT_LINE: str = "d,a,b,alpha,sample_size,seed,mean,ci_half_width,relative_error\n"
DEFAULT_EPSILON: float = 1e-4
DEFAULT_DS: list[int] = [16, 256]  # dimensions
DEFAULT_BETA_DIST_PARAMS: list[float] = [2**i for i in range(1, 4)]
DEFAULT_ALPHAS: list[float] = [i/1000 for i in range(1,51) if 1000 % i == 0]  # significances
DEFAULT_SAMPLE_SIZE: int = 2**14
DEFAULT_SEED: int = 42  # for reproducibility

def sample_beta(a: float, b: float, d: int) -> np.ndarray:
    # Sample d independent Beta(a,b) random variables
    return np.random.beta(a, b, d)

def sample_log_harmonic_mean_of_betas(n: int, d: int, a: float, b: float) -> np.ndarray:
    # Sample n*d beta random variables and, for each d-tuple, compute the logarithm of the harmonic mean
    result = []
    while len(result) < n:
        # let h be the logarithm of the harmonic mean of d beta random variables
        betas = sample_beta(a, b, d)
        h = np.log(d / np.mean(1/betas))
        result.append(h)
    return np.array(result)

def get_critical_t_values(sample_size, alpha=0.05) -> float:
    if sample_size <= 1:
        raise ValueError("Sample size must be greater than 1 to calculate t-critical value.")
    df = sample_size - 1  # degrees of freedom
    # Compute the two-tailed t-critical value for the given degrees of freedom
    t_critical = stats.t.ppf(1 - alpha / 2, df)
    return t_critical

def observations_to_n_mean_and_std(observations: np.ndarray) -> tuple[int, float, float]:
    # Compute the mean and standard deviation of some observations
    n: int = len(observations)
    mean: float = float(np.mean(observations))
    std: float = float(np.std(observations, ddof=1))
    return n, mean, std

def n_and_std_to_pi_half_width(n: int, std: float) -> float:
    # Compute a 2-sided 95% prediction interval half-width
    t_critical = get_critical_t_values(n)
    pi_half_width: float = t_critical * std * np.sqrt(1 + 1/n)
    return pi_half_width

def observations_to_mean_and_pi_half_width(observations: np.ndarray) -> tuple[float, float]:
    # Compute a 2-sided 95% confidence interval for some observations
    n, mean, std = observations_to_n_mean_and_std(observations)
    pi_half_width: float = n_and_std_to_pi_half_width(n, std)
    return float(mean), float(pi_half_width)

def sample_order_statistics(n: int, d: int, a: float, b: float, special_indices: list[int], seed: int) -> dict:
    # The special indices are the indices which are unbiased estimators corresponding to particular values of alpha we want
    np.random.seed(seed)
    h_samples = sample_log_harmonic_mean_of_betas(n, d, a, b)
    h_samples.sort()
    return {(n, d, a, b, (x+1)/n, seed): h_samples[x] for _, x in enumerate(special_indices)}

def collect_samples(n, d, a, b, special_indices, seeds, num_procs):
    """
    Run sample_order_statistics in parallel.
    """
    args = zip(repeat(n), repeat(d), repeat(a), repeat(b), repeat(special_indices), seeds)
    with Pool(processes=num_procs) as pool:
        results = list(tqdm(pool.starmap(sample_order_statistics, args), total=len(seeds)))
    return results

def update_estimates(results, estimates_per_alpha):
    """
    Update the estimates_per_alpha dictionary with new results.
    """
    for result in results:
        for k, v in result.items():
            (_, _, _, _, alpha, _) = k
            if alpha not in estimates_per_alpha:
                estimates_per_alpha[alpha] = []
            estimates_per_alpha[alpha].append(v)
    return estimates_per_alpha

def compute_estimates_summary(estimates_per_alpha, alphas, epsilon):
    """
    Compute estimates summary and check for convergence.
    """
    estimates_summary = {}
    converged = True
    for alpha in alphas:
        observations = np.array(estimates_per_alpha[alpha])
        mean = np.mean(observations)
        std_dev = np.std(observations, ddof=1)
        sample_size = len(observations)
        degrees_of_freedom = sample_size - 1
        t_critical = stats.t.ppf(1 - 0.025, degrees_of_freedom)
        pi_half_width = t_critical * std_dev / np.sqrt(sample_size)
        relative_error = pi_half_width / abs(mean)
        estimates_summary[alpha] = (mean, pi_half_width, relative_error, sample_size)
        if relative_error > epsilon:
            converged = False
    return estimates_summary, converged

def efficiently_sample_log_harmonic_mean_critical_values(
    epsilon: float, d: int, a: float, b: float, alphas: list[float], initial_sample_size: int, seed: int
) -> dict:
    """
    Estimate critical values for multiple alphas using a small initial sample to estimate required sample sizes
    before beginning the sample-size-doubling procedure.
    """
    num_procs = cpu_count()
    estimates_per_alpha = {}
    # Compute the least common multiple (LCM) of the 1/alpha values
    reciprocals = np.reciprocal(alphas).astype(int)
    n = np.lcm.reduce(reciprocals)
    special_indices = [int(n * alpha) - 1 for alpha in alphas]  # Zero-based indexing

    # Step 1: Use a small initial sample to estimate standard deviation
    initial_seeds = [seed + i for i in range(1, initial_sample_size + 1)]
    # we use n-1 because the k-th order statistic in a sample of size n is an unbiased estimator of the k/(n+1)-th quantile.
    initial_results = collect_samples(n-1, d, a, b, special_indices, initial_seeds, num_procs)
    estimates_per_alpha = update_estimates(initial_results, estimates_per_alpha)

    # Estimate standard deviation and required sample size for each alpha
    required_sample_sizes = {}
    for alpha in alphas:
        observations = np.array(estimates_per_alpha[alpha])
        n, mean, std = observations_to_n_mean_and_std(observations)
        degrees_of_freedom = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, degrees_of_freedom)
        # Calculate required sample size to achieve desired relative error, with a 10% buffer
        required_n = int(1.1 * (t_critical * std / (epsilon * abs(mean))) ** 2)
        required_sample_sizes[alpha] = max(required_n, n)

    # Determine the maximum required sample size among all alphas
    max_required_sample_size = max(required_sample_sizes.values())
    remaining_sample_size = max_required_sample_size - initial_sample_size
    print(f"Estimated remaining sample size = {remaining_sample_size}")

    # Step 2: Collect additional samples to meet required sample sizes
    total_sample_size = initial_sample_size
    if remaining_sample_size > 0:
        additional_seeds = [seed + total_sample_size + i for i in range(1, remaining_sample_size + 1)]
        additional_results = collect_samples(n, d, a, b, special_indices, additional_seeds, num_procs)
        estimates_per_alpha = update_estimates(additional_results, estimates_per_alpha)
        total_sample_size += remaining_sample_size

    # Final calculations
    estimates_summary, converged = compute_estimates_summary(estimates_per_alpha, alphas, epsilon)

    # If not converged, proceed with sample-size-doubling procedure
    while not converged:
        # Double the sample size
        required_sample_size = total_sample_size * 2
        additional_sample_size = required_sample_size - total_sample_size
        seeds = [seed + total_sample_size + i for i in range(1, additional_sample_size + 1)]
        additional_results = collect_samples(n, d, a, b, special_indices, seeds, num_procs)
        estimates_per_alpha = update_estimates(additional_results, estimates_per_alpha)
        total_sample_size = required_sample_size

        # Recalculate estimates
        estimates_summary, converged = compute_estimates_summary(estimates_per_alpha, alphas, epsilon)

    return estimates_summary

def write_to_file(filename: str, mode: str, init_line: str, results: dict):
    with open(filename, mode) as f:
        f.write(init_line)
        for k, v in results.items():
            d, a, b, alpha, sample_size, seed = k
            mean, ci_half_width, relative_error = v
            f.write(f"{d},{a},{b},{alpha},{sample_size},{seed},{mean},{ci_half_width},{relative_error}\n")

if __name__ == "__main__":
    alphas = [i / 1000 for i in range(1, 51) if 1000 % i == 0]
    indices = [
        (d, a, b, alphas, DEFAULT_SAMPLE_SIZE, DEFAULT_SEED)
        for d in DEFAULT_DS
        for a in DEFAULT_BETA_DIST_PARAMS
        for b in DEFAULT_BETA_DIST_PARAMS
    ]
    result: dict = dict()
    for index in tqdm(indices):
        d, a, b, alphas, sample_size, seed = index
        estimates_summary = efficiently_sample_log_harmonic_mean_critical_values(
            epsilon=DEFAULT_EPSILON,
            d=d,
            a=a,
            b=b,
            alphas=alphas,
            initial_sample_size=DEFAULT_SAMPLE_SIZE,
            seed=seed,
        )
        # Process the estimates_summary as needed
        for alpha, (mean, ci_half_width, relative_error, num_observations) in estimates_summary.items():
            key = (d, a, b, alpha, num_observations, seed)
            result[key] = (mean, ci_half_width, relative_error)
    write_to_file(
        filename=DEFAULT_FILENAME,
        mode=DEFAULT_MODE,
        init_line=DEFAULT_INIT_LINE,
        results=result,
    )
