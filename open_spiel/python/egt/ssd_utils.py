"""Utility functions for Stochastically Stable Distribution (SSD) analysis.

This module provides utility functions that complement the main SSD algorithm,
including compatibility functions for Alpha-Rank integration, ranking analysis,
and result formatting functions.
"""

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Tuple, Union, Optional, Any
import itertools

from open_spiel.python.egt import utils


def check_payoffs_are_ssd_compatible(payoff_tables: List[Any]) -> bool:
  """Check if payoff tables are compatible with SSD analysis.
  
  Args:
    payoff_tables: List of payoff tables (numpy arrays or HPT format).
    
  Returns:
    Boolean indicating compatibility.
  """
  if not payoff_tables:
    return False
    
  # Check if all tables have same dimensionality structure
  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  
  if payoffs_are_hpt_format:
    # For HPT format, check that all tables are compatible
    try:
      num_strats_per_population = utils.get_num_strats_per_population(
          payoff_tables, payoffs_are_hpt_format)
      return len(num_strats_per_population) > 0
    except:
      return False
  else:
    # For numpy arrays, check shapes are consistent
    if len(payoff_tables) == 1:
      # Single population game
      table = payoff_tables[0]
      return len(table.shape) == 2 and table.shape[0] == table.shape[1]
    else:
      # Multi-population game - check dimensions match
      base_shape = payoff_tables[0].shape
      return all(table.shape == base_shape for table in payoff_tables)


def convert_alpharank_to_ssd_format(payoff_tables: List[Any],
                                   payoffs_are_hpt_format: bool) -> Dict[str, Any]:
  """Convert Alpha-Rank payoff tables to SSD-compatible format.
  
  Args:
    payoff_tables: Payoff tables in Alpha-Rank format.
    payoffs_are_hpt_format: Whether tables are in HPT format.
    
  Returns:
    Dictionary containing SSD-compatible payoff information.
  """
  result = {
    'payoff_tables': payoff_tables,
    'payoffs_are_hpt_format': payoffs_are_hpt_format,
    'num_populations': len(payoff_tables),
    'num_strats_per_population': utils.get_num_strats_per_population(
        payoff_tables, payoffs_are_hpt_format)
  }
  
  if result['num_populations'] == 1:
    result['num_profiles'] = result['num_strats_per_population'][0]
    result['game_type'] = 'single_population'
  else:
    result['num_profiles'] = utils.get_num_profiles(
        result['num_strats_per_population'])
    result['game_type'] = 'multi_population'
    
  return result


def get_ssd_strat_profile_labels(payoff_tables: List[Any],
                                payoffs_are_hpt_format: bool,
                                custom_labels: Optional[Dict[int, str]] = None) -> Dict[int, str]:
  """Generate strategy profile labels compatible with SSD analysis.
  
  Args:
    payoff_tables: Payoff tables.
    payoffs_are_hpt_format: Whether tables are in HPT format.
    custom_labels: Optional custom labels for strategies.
    
  Returns:
    Dictionary mapping profile indices to human-readable labels.
  """
  if custom_labels is not None:
    return custom_labels
    
  # Use existing Alpha-Rank labeling system for compatibility
  return utils.get_strat_profile_labels(payoff_tables, payoffs_are_hpt_format)


def compute_ranking_from_distribution(distribution: np.ndarray) -> np.ndarray:
  """Compute strategy rankings from a stationary distribution.
  
  Args:
    distribution: Stationary distribution over strategies.
    
  Returns:
    Array of rankings (0 = highest ranked strategy).
  """
  # Higher probability = better rank (lower rank number)
  return stats.rankdata(-distribution, method='ordinal') - 1


def ranking_correlation(dist1: np.ndarray, 
                       dist2: np.ndarray,
                       method: str = 'spearman') -> float:
  """Compute ranking correlation between two distributions.
  
  Args:
    dist1: First distribution.
    dist2: Second distribution.
    method: Correlation method ('spearman', 'kendall', 'pearson').
    
  Returns:
    Correlation coefficient.
  """
  if len(dist1) != len(dist2):
    raise ValueError("Distributions must have same length")
    
  if method == 'spearman':
    corr, _ = stats.spearmanr(dist1, dist2)
  elif method == 'kendall':
    corr, _ = stats.kendalltau(dist1, dist2)
  elif method == 'pearson':
    corr, _ = stats.pearsonr(dist1, dist2)
  else:
    raise ValueError(f"Unknown correlation method: {method}")
    
  return corr if not np.isnan(corr) else 0.0


def top_k_overlap(dist1: np.ndarray, 
                 dist2: np.ndarray, 
                 k: int = 10) -> float:
  """Compute overlap in top-k strategies between two distributions.
  
  Args:
    dist1: First distribution.
    dist2: Second distribution.
    k: Number of top strategies to consider.
    
  Returns:
    Fraction of top-k strategies that overlap.
  """
  if len(dist1) != len(dist2):
    raise ValueError("Distributions must have same length")
    
  k = min(k, len(dist1))
  
  top_k_1 = set(np.argsort(dist1)[-k:])
  top_k_2 = set(np.argsort(dist2)[-k:])
  
  return len(top_k_1.intersection(top_k_2)) / k


def distribution_distance(dist1: np.ndarray, 
                         dist2: np.ndarray,
                         metric: str = 'tvd') -> float:
  """Compute distance between two probability distributions.
  
  Args:
    dist1: First distribution.
    dist2: Second distribution.  
    metric: Distance metric ('tvd', 'kl', 'hellinger', 'l2').
    
  Returns:
    Distance between distributions.
  """
  if len(dist1) != len(dist2):
    raise ValueError("Distributions must have same length")
    
  # Ensure distributions are normalized
  dist1 = dist1 / np.sum(dist1)
  dist2 = dist2 / np.sum(dist2)
  
  if metric == 'tvd':
    # Total variation distance
    return 0.5 * np.sum(np.abs(dist1 - dist2))
  elif metric == 'kl':
    # Kullback-Leibler divergence
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    dist1_safe = dist1 + epsilon
    dist2_safe = dist2 + epsilon
    return np.sum(dist1_safe * np.log(dist1_safe / dist2_safe))
  elif metric == 'hellinger':
    # Hellinger distance
    return np.sqrt(0.5 * np.sum((np.sqrt(dist1) - np.sqrt(dist2))**2))
  elif metric == 'l2':
    # L2 (Euclidean) distance
    return np.sqrt(np.sum((dist1 - dist2)**2))
  else:
    raise ValueError(f"Unknown metric: {metric}")


def print_ssd_rankings_table(payoff_tables: List[Any],
                           ssd_distribution: np.ndarray,
                           strat_labels: Dict[int, str],
                           num_top_strats_to_print: int = 8) -> None:
  """Print SSD rankings table in format similar to Alpha-Rank.
  
  Args:
    payoff_tables: Payoff tables.
    ssd_distribution: SSD stationary distribution.
    strat_labels: Strategy labels.
    num_top_strats_to_print: Number of top strategies to display.
  """
  print('\n=====================================')
  print('Stochastically Stable Distribution Rankings')
  print('=====================================\n')
  
  # Sort strategies by probability (descending)
  sorted_indices = np.argsort(ssd_distribution)[::-1]

  print("actual ssd dist:", ssd_distribution)
  
  print(f"{'Rank':<6} {'Strategy':<20} {'Probability':<12} {'Percentage':<12}")
  print('-' * 52)
  
  for i, strategy_idx in enumerate(sorted_indices[:num_top_strats_to_print]):
    rank = i + 1
    if isinstance(strat_labels, list):
      strategy_name = f"Strategy_{strategy_idx}"
    else:
      strategy_name = strat_labels.get(strategy_idx, f"Strategy_{strategy_idx}")
    probability = ssd_distribution[strategy_idx]
    percentage = 100.0 * probability
    
    print("strat index:", strategy_idx)
    print(f"{rank:<6} {probability:<12.6f} {percentage:<12.2f}%")
  
  if len(sorted_indices) > num_top_strats_to_print:
    remaining_mass = np.sum(ssd_distribution[sorted_indices[num_top_strats_to_print:]])
    remaining_count = len(sorted_indices) - num_top_strats_to_print
    print(f"...    ({remaining_count} others)       {remaining_mass:<12.6f} {100.0*remaining_mass:<12.2f}%")
  
  print()


def compare_rankings_ssd_alpharank(ssd_dist: np.ndarray,
                                  alpharank_dist: np.ndarray,
                                  strat_labels: Dict[int, str],
                                  num_top_strats: int = 10) -> Dict[str, Any]:
  """Compare and analyze ranking differences between SSD and Alpha-Rank.
  
  Args:
    ssd_dist: SSD stationary distribution.
    alpharank_dist: Alpha-Rank stationary distribution.
    strat_labels: Strategy labels.
    num_top_strats: Number of top strategies to analyze.
    
  Returns:
    Dictionary containing comparison metrics and analysis.
  """
  if len(ssd_dist) != len(alpharank_dist):
    raise ValueError("Distributions must have same length")
  
  # Compute various comparison metrics
  results = {
    'spearman_correlation': ranking_correlation(ssd_dist, alpharank_dist, 'spearman'),
    'kendall_correlation': ranking_correlation(ssd_dist, alpharank_dist, 'kendall'), 
    'pearson_correlation': ranking_correlation(ssd_dist, alpharank_dist, 'pearson'),
    'top_k_overlap': top_k_overlap(ssd_dist, alpharank_dist, num_top_strats),
    'tvd_distance': distribution_distance(ssd_dist, alpharank_dist, 'tvd'),
    'hellinger_distance': distribution_distance(ssd_dist, alpharank_dist, 'hellinger'),
    'l2_distance': distribution_distance(ssd_dist, alpharank_dist, 'l2')
  }
  
  # Rankings analysis
  ssd_rankings = compute_ranking_from_distribution(ssd_dist)
  alpharank_rankings = compute_ranking_from_distribution(alpharank_dist)
  
  # Find strategies with biggest ranking differences
  ranking_diffs = np.abs(ssd_rankings - alpharank_rankings)
  biggest_diffs_idx = np.argsort(ranking_diffs)[::-1]
  
  results['ranking_differences'] = []
  for i in range(min(5, len(biggest_diffs_idx))):
    strategy_idx = biggest_diffs_idx[i]
    diff = ranking_diffs[strategy_idx]
    if diff > 0:  # Only report non-zero differences
      print("ALPHARANK pROB:", alpharank_dist[strategy_idx])
      results['ranking_differences'].append({
        'strategy': strat_labels[strategy_idx] if isinstance(strat_labels, list) else strat_labels.get(strategy_idx, f"Strategy_{strategy_idx}"),
        'ssd_rank': int(ssd_rankings[strategy_idx]),
        'alpharank_rank': int(alpharank_rankings[strategy_idx]),
        'rank_difference': int(diff),
        'ssd_probability': ssd_dist[strategy_idx],
        'alpharank_probability': alpharank_dist[strategy_idx]
      })
  
  # Top strategies comparison
  ssd_top_k = np.argsort(ssd_dist)[::-1][:num_top_strats]
  alpharank_top_k = np.argsort(alpharank_dist)[::-1][:num_top_strats]
  
  results['top_strategies'] = {
    'ssd_only': ["{strat_labels[idx]}" 
                 for idx in ssd_top_k if idx not in alpharank_top_k],
    'alpharank_only': ["{strat_labels[idx]}" 
                       for idx in alpharank_top_k if idx not in ssd_top_k],
    'both': ["{strat_labels[idx]}" 
             for idx in ssd_top_k if idx in alpharank_top_k]
  }
  
  return results


def print_comparison_summary(comparison_results: Dict[str, Any]) -> None:
  """Print a summary of SSD vs Alpha-Rank comparison results.
  
  Args:
    comparison_results: Results from compare_rankings_ssd_alpharank.
  """
  print('\n=====================================')
  print('SSD vs Alpha-Rank Comparison Summary')
  print('=====================================\n')
  
  print('Correlation Metrics:')
  print(f"  Spearman correlation:  {comparison_results['spearman_correlation']:.4f}")
  print(f"  Kendall correlation:   {comparison_results['kendall_correlation']:.4f}")
  print(f"  Pearson correlation:   {comparison_results['pearson_correlation']:.4f}")
  
  print('\nDistance Metrics:')
  print(f"  Total variation distance: {comparison_results['tvd_distance']:.4f}")
  print(f"  Hellinger distance:       {comparison_results['hellinger_distance']:.4f}")
  print(f"  L2 distance:              {comparison_results['l2_distance']:.4f}")
  
  print(f"\nTop-k overlap: {comparison_results['top_k_overlap']:.2%}")
  
  if comparison_results['ranking_differences']:
    print('\nBiggest Ranking Differences:')
    print(f"{'Strategy':<20} {'SSD Rank':<10} {'AR Rank':<10} {'Diff':<6} {'SSD Prob':<10} {'AR Prob':<10}")
    print('-' * 76)
    
    for diff_info in comparison_results['ranking_differences']:
      print(f"{diff_info['strategy']} "
            f"{diff_info['ssd_rank']:<10} "
            f"{diff_info['alpharank_rank']:<10} "
            f"{diff_info['rank_difference']:<6} "
            f"ssd_p: {diff_info['ssd_probability']:<10.4f} "
            f"ar_p: {diff_info['alpharank_probability']:<10.4f}")
  
  print('\nTop Strategy Analysis:')
  top_strats = comparison_results['top_strategies']
  print(f"  Strategies in both top-k:     {len(top_strats['both'])}")
  print(f"  Strategies only in SSD top-k: {len(top_strats['ssd_only'])}")
  print(f"  Strategies only in AR top-k:  {len(top_strats['alpharank_only'])}")
  
  if top_strats['ssd_only']:
    print(f"    SSD-only top strategies: {', '.join(top_strats['ssd_only'][:3])}")
  if top_strats['alpharank_only']:
    print(f"    AR-only top strategies:  {', '.join(top_strats['alpharank_only'][:3])}")


def validate_ssd_distribution(distribution: np.ndarray, 
                             tolerance: float = 1e-6) -> bool:
  """Validate that a distribution is a proper probability distribution.
  
  Args:
    distribution: Distribution to validate.
    tolerance: Tolerance for numerical errors.
    
  Returns:
    True if distribution is valid.
  """
  # Check non-negativity
  if np.any(distribution < -tolerance):
    return False
    
  # Check normalization
  if abs(np.sum(distribution) - 1.0) > tolerance:
    return False
    
  return True


def normalize_distribution(distribution: np.ndarray) -> np.ndarray:
  """Normalize a distribution to sum to 1.
  
  Args:
    distribution: Unnormalized distribution.
    
  Returns:
    Normalized distribution.
  """
  total = np.sum(distribution)
  if total == 0:
    # Uniform distribution if all entries are zero
    return np.ones(len(distribution)) / len(distribution)
  return distribution / total


def get_support_strategies(distribution: np.ndarray,
                          threshold: float = 1e-10) -> List[int]:
  """Get indices of strategies in the support of the distribution.
  
  Args:
    distribution: Probability distribution.
    threshold: Minimum probability to be considered in support.
    
  Returns:
    List of strategy indices with probability above threshold.
  """
  return [i for i, prob in enumerate(distribution) if prob > threshold]


def entropy(distribution: np.ndarray) -> float:
  """Compute Shannon entropy of a probability distribution.
  
  Args:
    distribution: Probability distribution.
    
  Returns:
    Shannon entropy in nats.
  """
  # Avoid log(0) by adding small epsilon
  epsilon = 1e-12
  safe_dist = distribution + epsilon
  return -np.sum(safe_dist * np.log(safe_dist))


def effective_strategies(distribution: np.ndarray) -> float:
  """Compute the effective number of strategies.
  
  This is the exponential of Shannon entropy, representing the
  "effective" number of strategies that would give the same entropy
  if uniformly distributed.
  
  Args:
    distribution: Probability distribution.
    
  Returns:
    Effective number of strategies.
  """
  return np.exp(entropy(distribution))


def concentration_ratio(distribution: np.ndarray, k: int = 1) -> float:
  """Compute the concentration ratio (sum of top-k probabilities).
  
  Args:
    distribution: Probability distribution.
    k: Number of top strategies to include.
    
  Returns:
    Sum of probabilities of top-k strategies.
  """
  if k <= 0 or k > len(distribution):
    raise ValueError(f"k must be between 1 and {len(distribution)}")
    
  sorted_probs = np.sort(distribution)[::-1]
  return np.sum(sorted_probs[:k])


def gini_coefficient(distribution: np.ndarray) -> float:
  """Compute Gini coefficient measuring inequality in the distribution.
  
  Args:
    distribution: Probability distribution.
    
  Returns:
    Gini coefficient (0 = perfect equality, 1 = maximum inequality).
  """
  n = len(distribution)
  sorted_dist = np.sort(distribution)
  
  # Gini coefficient formula
  index = np.arange(1, n + 1)
  return (2 * np.sum(index * sorted_dist)) / (n * np.sum(sorted_dist)) - (n + 1) / n