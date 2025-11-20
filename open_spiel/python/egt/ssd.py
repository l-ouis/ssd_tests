"""Implementation of Stochastically Stable Distribution (SSD) analysis.

This module implements the SSD algorithm for computing stochastically stable
distributions of perturbed Markov processes, with applications to evolutionary
game theory and multiagent learning. The implementation is compatible with
OpenSpiel's Alpha-Rank framework.

The algorithm is based on:
"An Algorithm for Computing Stochastically Stable Distributions with 
Applications to Multiagent Learning in Repeated Games" 
by John R. Wicks and Amy Greenwald.
"""

import numpy as np
from numpy.polynomial import Polynomial as P
import scipy.linalg as la
import scipy.sparse as sp
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import warnings
import pyspiel

# Import OpenSpiel utils if available, otherwise provide minimal implementations
from open_spiel.python.egt import utils
from open_spiel.python.egt import heuristic_payoff_table

from open_spiel.python.egt.test_SSD import SSD

# Try to import ssd_utils, otherwise provide minimal implementations
try:
    import ssd_utils
except ImportError:
    # Minimal ssd_utils implementation
    class ssd_utils:
        @staticmethod
        def check_payoffs_are_ssd_compatible(payoff_tables):
            return len(payoff_tables) > 0 and all(isinstance(table, np.ndarray) for table in payoff_tables)
        
        @staticmethod
        def convert_alpharank_to_ssd_format(payoff_tables, payoffs_are_hpt_format):
            return {
                'payoff_tables': payoff_tables,
                'payoffs_are_hpt_format': payoffs_are_hpt_format,
                'num_populations': len(payoff_tables),
                'num_strats_per_population': utils.get_num_strats_per_population(payoff_tables, payoffs_are_hpt_format),
                'num_profiles': len(payoff_tables[0]) if payoffs_are_hpt_format else 
                               (payoff_tables[0].shape[0] if len(payoff_tables) == 1 else 
                                utils.get_num_profiles([table.shape[0] for table in payoff_tables]))
            }
        
        @staticmethod
        def get_ssd_strat_profile_labels(payoff_tables, payoffs_are_hpt_format, custom_labels=None):
            if custom_labels is not None:
                return custom_labels
            return utils.get_strat_profile_labels(payoff_tables, payoffs_are_hpt_format)
        
        @staticmethod
        def validate_ssd_distribution(distribution, tolerance=1e-6):
            return (np.all(distribution >= -tolerance) and 
                    abs(np.sum(distribution) - 1.0) <= tolerance)
        
        @staticmethod
        def normalize_distribution(distribution):
            total = np.sum(distribution)
            if total == 0:
                return np.ones(len(distribution)) / len(distribution)
            return distribution / total
        
        @staticmethod
        def print_ssd_rankings_table(payoff_tables, ssd_distribution, strat_labels, num_top_strats_to_print=8):
            print('\n=====================================')
            print('Stochastically Stable Distribution Rankings (simple mode)')
            print('=====================================\n')
            
            sorted_indices = np.argsort(ssd_distribution)[::-1]
            
            for i, strategy_idx in enumerate(sorted_indices[:num_top_strats_to_print]):
                rank = i + 1
                # strategy_name = strat_labels.get(strategy_idx, f"Strategy_{strategy_idx}")
                # strategy_name = strat_labels[strategy_idx]
                strategy_name = "strat index: " + str(strategy_idx)
                # print("STRAT LABELS:", strat_labels)
                probability = ssd_distribution[strategy_idx]
                percentage = 100.0 * probability
                
                print(f"{rank}. {strategy_name}: {probability:.6f} ({percentage:.2f}%)")
        
        @staticmethod
        def ranking_correlation(dist1, dist2, method='spearman'):
            from scipy import stats
            if method == 'spearman':
                corr, _ = stats.spearmanr(dist1, dist2)
            elif method == 'kendall':
                corr, _ = stats.kendalltau(dist1, dist2)
            elif method == 'pearson':
                corr, _ = stats.pearsonr(dist1, dist2)
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            return corr if not np.isnan(corr) else 0.0
        
        @staticmethod
        def distribution_distance(dist1, dist2, metric='tvd'):
            if metric == 'tvd':
                return 0.5 * np.sum(np.abs(dist1 - dist2))
            elif metric == 'hellinger':
                return np.sqrt(0.5 * np.sum((np.sqrt(dist1) - np.sqrt(dist2))**2))
            elif metric == 'l2':
                return np.sqrt(np.sum((dist1 - dist2)**2))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        @staticmethod
        def compare_rankings_ssd_alpharank(ssd_dist, alpharank_dist, strat_labels, num_top_strats=10):
            # Ensure strat_labels is a dictionary
            if isinstance(strat_labels, list):
                strat_labels = {i: f"Strategy_{i}" for i in range(len(strat_labels))}
            elif not isinstance(strat_labels, dict):
                strat_labels = {i: f"Strategy_{i}" for i in range(len(ssd_dist))}
                
            return {
                'spearman_correlation': ssd_utils.ranking_correlation(ssd_dist, alpharank_dist, 'spearman'),
                'kendall_correlation': ssd_utils.ranking_correlation(ssd_dist, alpharank_dist, 'kendall'),
                'tvd_distance': ssd_utils.distribution_distance(ssd_dist, alpharank_dist, 'tvd'),
                'hellinger_distance': ssd_utils.distribution_distance(ssd_dist, alpharank_dist, 'hellinger'),
                'top_strategies': {
                    'ssd_only': [],
                    'alpharank_only': [],
                    'both': []
                }
            }
        
        @staticmethod
        def print_comparison_summary(comparison_results):
            print('\nSSD vs Alpha-Rank Comparison Summary:')
            for key, value in comparison_results.items():
                print(f"  {key}: {value}")


class PolynomialMatrix:
  """Matrix of polynomials for representing perturbed Markov processes.

  This class can store its entries in either a dense numpy object array or a
  scipy.sparse matrix with dtype=object. When `use_sparse=True`, evaluated
  numeric matrices are returned as scipy.sparse.csr_matrix to save memory.
  """

  def __init__(self, matrix: np.ndarray | sp.spmatrix, use_sparse: bool = False):
    """Initialize polynomial matrix.

    Args:
      matrix: 2D array-like where each entry is a numpy.poly1d polynomial or
        an existing scipy sparse matrix with dtype=object.
      use_sparse: if True, store the polynomial matrix internally as a
        scipy.sparse.lil_matrix(dtype=object) and return evaluated matrices
        as scipy.sparse.csr_matrix.
    """
    self.use_sparse = bool(use_sparse)

    # Normalize input storage: convert to sparse object matrix if requested.
    if self.use_sparse:
      # If the input is already a scipy sparse matrix, keep it (ensure object
      # dtype). Otherwise create an empty sparse LIL matrix of the right
      # shape and populate any non-empty entries if provided.
      if sp.issparse(matrix):
        self.matrix = matrix.tolil()
      else:
        # Expect matrix to have a shape attribute (e.g., numpy array)
        shape = getattr(matrix, 'shape', None)
        if shape is None:
          raise ValueError('Cannot infer shape for sparse PolynomialMatrix')
        self.matrix = sp.lil_matrix(shape, dtype=object)
        # If input is an ndarray-like object, populate non-zero entries.
        try:
          it = np.ndenumerate(matrix)
          for (i, j), val in it:
            # Only set entries that are non-zero/None; users commonly pass
            # np.zeros(..., dtype=object) as template so skip zeros.
            if val is not None and not (isinstance(val, (int, float)) and val == 0):
              self.matrix[i, j] = val
        except Exception:
          # If enumeration fails, leave as empty sparse matrix.
          pass
    else:
      # Dense storage as numpy object array
      if sp.issparse(matrix):
        # Convert sparse object matrix to dense ndarray
        self.matrix = np.array(matrix.todense(), dtype=object)
      else:
        self.matrix = np.array(matrix, copy=True, dtype=object)

    self.shape = self.matrix.shape

  def __getitem__(self, key):
    return self.matrix[key]

  def __setitem__(self, key, value):
    self.matrix[key] = value

  def copy(self):
    """Create a deep copy of the polynomial matrix, preserving sparsity."""
    if self.use_sparse:
      return PolynomialMatrix(self.matrix.copy(), use_sparse=True)
    return PolynomialMatrix(np.array(self.matrix, copy=True), use_sparse=False)

  def evaluate_at(self, epsilon: float):
    """Evaluate all polynomials at given epsilon value.

    Returns a numpy.ndarray for dense storage, or a scipy.sparse.csr_matrix
    when this PolynomialMatrix was created with use_sparse=True.
    """
    if self.use_sparse:
      # Build a sparse numeric matrix by evaluating only stored polynomial
      # entries. Use LIL for incremental construction, convert to CSR for use.
      result = sp.lil_matrix(self.shape, dtype=float)

      # If internal storage is sparse, iterate its non-zero entries; if it's
      # accidentally dense, fall back to full iteration.
      if sp.issparse(self.matrix):
        coo = self.matrix.tocoo()
        for i, j, poly in zip(coo.row, coo.col, coo.data):
          try:
            if isinstance(poly, np.poly1d):
              val = float(poly(epsilon))
            else:
              val = float(poly)
          except Exception:
            # Try to extract scalar from 1-length containers
            if hasattr(poly, '__len__') and len(poly) == 1:
              val = float(poly[0])
            else:
              val = 0.0
          if val != 0.0:
            result[i, j] = val
      else:
        # Dense object array: iterate all entries (may be slow for large size)
        rows, cols = self.shape
        for i in range(rows):
          for j in range(cols):
            poly = self.matrix[i, j]
            try:
              if isinstance(poly, np.poly1d):
                val = float(poly(epsilon))
              else:
                val = float(poly)
            except Exception:
              if hasattr(poly, '__len__') and len(poly) == 1:
                val = float(poly[0])
              else:
                val = 0.0
            if val != 0.0:
              result[i, j] = val

      return result.tocsr()

    # Dense path: return numpy array
    rows, cols = self.shape
    result = np.zeros(self.shape, dtype=float)
    for i in range(rows):
      for j in range(cols):
        poly = self.matrix[i, j]
        if isinstance(poly, np.poly1d):
          result[i, j] = poly(epsilon)
        else:
          try:
            result[i, j] = float(poly)
          except Exception:
            if hasattr(poly, '__len__') and len(poly) == 1:
              result[i, j] = float(poly[0])
            else:
              result[i, j] = 0.0
    return result

  def get_constant_term_matrix(self):
    """Extract matrix of constant terms (epsilon=0)."""
    return self.evaluate_at(0.0)


def _create_polynomial(coefficients: Union[List[float], float, np.poly1d]) -> np.poly1d:
  """Create a polynomial from various input formats."""
  if isinstance(coefficients, np.poly1d):
    return coefficients
  elif isinstance(coefficients, (int, float)):
    return np.poly1d([coefficients])
  elif isinstance(coefficients, (list, tuple, np.ndarray)):
    return np.poly1d(coefficients)
  else:
    raise ValueError(f"Cannot create polynomial from {type(coefficients)}")


def _get_resistance_and_cost(polynomial: np.poly1d) -> Tuple[int, float]:
  """Get resistance (degree of lowest non-zero term) and cost (coefficient).
  
  Args:
    polynomial: Polynomial to analyze.
    
  Returns:
    Tuple of (resistance, cost) where resistance is the power of epsilon
    and cost is the coefficient.
  """
  if polynomial == np.poly1d([0]):
    return float('inf'), 0.0
    
  coeffs = polynomial.coefficients
  # Find first non-zero coefficient (lowest degree term)
  for i in range(len(coeffs) - 1, -1, -1):
    if abs(coeffs[i]) > 1e-12:
      resistance = len(coeffs) - 1 - i
      cost = coeffs[i]
      return resistance, cost
  
  return float('inf'), 0.0


def _has_zero_diagonal(matrix: PolynomialMatrix) -> bool:
  """Check if matrix has any zero diagonal entries."""
  for i in range(min(matrix.shape)):
    poly = matrix[i, i]
    if isinstance(poly, np.poly1d):
      if poly == np.poly1d([0]):
        return True
    elif poly == 0:
      return True
  return False


def _uniform_scale(matrix: PolynomialMatrix, scale_factor: float = 0.5) -> PolynomialMatrix:
  """Apply uniform scaling to eliminate zero diagonal entries."""
  if not _has_zero_diagonal(matrix):
    return matrix
    
  scaled = matrix.copy()
  identity_poly = np.poly1d([1])  # Constant polynomial 1
  
  for i in range(min(matrix.shape)):
    for j in range(matrix.shape[1]):
      if i == j:
        # Diagonal: scale_factor * (M_ij + I_ij)
        scaled[i, j] = scale_factor * (matrix[i, j] + identity_poly)
      else:
        # Off-diagonal: scale_factor * M_ij
        scaled[i, j] = scale_factor * matrix[i, j]
        
  return scaled


def _find_communicating_classes(adjacency_matrix: np.ndarray) -> Dict[int, List[int]]:
  """Find strongly connected components using Kosaraju's algorithm."""
  n = adjacency_matrix.shape[0]
  
  # First DFS to get finish times
  visited = [False] * n
  finish_stack = []
  
  def dfs1(v):
    visited[v] = True
    for u in range(n):
      # Fix: Handle potential array comparison issues
      edge_weight = adjacency_matrix[v, u]
      if np.isscalar(edge_weight):
        has_edge = edge_weight > 1e-12
      else:
        has_edge = np.any(edge_weight > 1e-12)
      
      if has_edge and not visited[u]:
        dfs1(u)
    finish_stack.append(v)
  
  for v in range(n):
    if not visited[v]:
      dfs1(v)
  
  # Create transpose graph
  transpose = adjacency_matrix.T
  
  # Second DFS on transpose in reverse finish order
  visited = [False] * n
  components = {}
  component_id = 0
  
  def dfs2(v, comp_id):
    visited[v] = True
    if comp_id not in components:
      components[comp_id] = []
    components[comp_id].append(v)
    
    for u in range(n):
      edge_weight = transpose[v, u]
      if np.isscalar(edge_weight):
        has_edge = edge_weight > 1e-12
      else:
        has_edge = np.any(edge_weight > 1e-12)
        
      if has_edge and not visited[u]:
        dfs2(u, comp_id)
  
  for v in reversed(finish_stack):
    if not visited[v]:
      dfs2(v, component_id)
      component_id += 1
  
  return components


def _find_closed_classes(matrix: np.ndarray, 
                        communicating_classes: Dict[int, List[int]]) -> Dict[int, List[int]]:
  """Find closed communicating classes."""
  closed_classes = {}
  
  for class_id, vertices in communicating_classes.items():
    is_closed = True
    vertex_set = set(vertices)
    
    # Check if any vertex has outgoing edges to other classes
    for v in vertices:
      for u in range(matrix.shape[1]):
        # Fix: Handle potential array comparison issues
        edge_weight = matrix[v, u]
        if np.isscalar(edge_weight):
          has_edge = edge_weight > 1e-12
        else:
          has_edge = np.any(edge_weight > 1e-12)
          
        if has_edge and u not in vertex_set:
          is_closed = False
          break
      if not is_closed:
        break
    
    if is_closed:
      closed_classes[class_id] = vertices
  
  return closed_classes


def _ssd_step(matrix: PolynomialMatrix) -> Tuple[Optional[np.ndarray], 
                                                Optional[PolynomialMatrix], 
                                                Optional[np.ndarray]]:
  """Perform one step of the SSD algorithm.
  
  Returns:
    Tuple of (stable_distribution, reduced_matrix, inclusion_operator).
    If stable_distribution is not None, algorithm has converged.
  """
  # Get constant term matrix (epsilon = 0)
  M0 = matrix.get_constant_term_matrix()
  
  # Find communicating classes
  # Create adjacency matrix (non-zero entries indicate transitions)
  adjacency = (M0 > 1e-12).astype(float)
  communicating_classes = _find_communicating_classes(adjacency)
  
  # Check if we have a single communicating class (convergence)
  if len(communicating_classes) == 1:
    # Compute stable distribution of M0
    try:
      eigenvals, eigenvects = np.linalg.eig(M0.T)
      # Find eigenvector for eigenvalue 1
      idx = np.argmin(np.abs(eigenvals - 1.0))
      stable_dist = np.real(eigenvects[:, idx])
      stable_dist = np.abs(stable_dist)  # Ensure non-negative
      stable_dist = stable_dist / np.sum(stable_dist)  # Normalize
      return stable_dist, None, None
    except:
      # Fallback: uniform distribution
      n = M0.shape[0]
      return np.ones(n) / n, None, None
  
  # Find closed classes
  closed_classes = _find_closed_classes(M0, communicating_classes)
  
  if closed_classes:
    # Eliminate all but one representative from the largest closed class
    max_size = 0
    largest_class = None
    
    for class_id, vertices in closed_classes.items():
      if len(vertices) > max_size:
        max_size = len(vertices)
        largest_class = vertices
    
    if largest_class and len(largest_class) > 1:
      # Eliminate all but the first vertex of the largest class
      states_to_eliminate = largest_class[1:]
      
      # Simplified quotient construction for this implementation
      # Just remove the states and renormalize
      remaining_states = [i for i in range(matrix.shape[0]) if i not in states_to_eliminate]
      
      # Create reduced matrix
      new_size = len(remaining_states)
      reduced_matrix = PolynomialMatrix(np.zeros((new_size, new_size), dtype=object))
      
      for i, orig_i in enumerate(remaining_states):
        for j, orig_j in enumerate(remaining_states):
          reduced_matrix[i, j] = matrix[orig_i, orig_j]
      
      # Create inclusion operator
      inclusion_op = np.zeros((matrix.shape[0], new_size))
      for i, orig_idx in enumerate(remaining_states):
        inclusion_op[orig_idx, i] = 1.0
      
      return None, reduced_matrix, inclusion_op
  
  # No closed classes found - apply scaling to make progress
  scaled_matrix = _uniform_scale(matrix)
  return None, scaled_matrix, np.eye(matrix.shape[0])


def _ssd_iterate(matrix: PolynomialMatrix, 
                max_iterations: int = 1000,
                verbose: bool = False) -> np.ndarray:
  """Iteratively apply SSD steps until convergence.
  
  The key insight is that we need to find the limit as ε→0, not just
  evaluate at ε=0 directly.
  
  Args:
    matrix: Initial polynomial matrix.
    max_iterations: Maximum number of iterations.
    verbose: Whether to print progress.
    
  Returns:
    Stochastically stable distribution.
  """
  mtx = matrix
  if hasattr(matrix, "matrix"):
    mtx = matrix.matrix
  res = SSD(mtx)
  return res


def construct_perturbed_markov_matrix_ev_dyn(payoff_tables: List[Any],
                                    payoffs_are_hpt_format: bool,
                                    perturbation_type: str = "uniform",
                                    perturbation_strength: float = 1.0,
                                    use_sparse: bool = False) -> PolynomialMatrix:
  game_info = ssd_utils.convert_alpharank_to_ssd_format(
      payoff_tables, payoffs_are_hpt_format)

  num_profiles = game_info['num_profiles']
  num_populations = game_info['num_populations']
  num_strats_per_population = game_info['num_strats_per_population']

  epsilon = np.poly1d([1, 0])  # epsilon
  one = np.poly1d([1])
  zero = np.poly1d([0])

  m = 10.0
  alpha = 50.0

  from open_spiel.python.egt import alpharank
  if num_populations == 1:
    game_is_constant_sum, payoff_sum = utils.check_is_constant_sum(
        payoff_tables[0], payoffs_are_hpt_format)
    # Use alpharank helper to get the base transition matrix c and rhos
    c_base, rhos = alpharank._get_singlepop_transition_matrix(
        payoff_tables[0], payoffs_are_hpt_format,
        m=m, alpha=alpha,
        game_is_constant_sum=game_is_constant_sum,
        use_local_selection_model=True,
        payoff_sum=payoff_sum,
        use_inf_alpha=False,
        inf_alpha_eps=0.1)
    # c_base is a (num_strats x num_strats) matrix with numeric entries.
    base = np.array(c_base, dtype=float)
    size = base.shape[0]
  else:
    c_base, rhos = alpharank._get_multipop_transition_matrix(
        payoff_tables, payoffs_are_hpt_format,
        m=m, alpha=alpha,
        use_inf_alpha=False,
        inf_alpha_eps=0.1)
    base = np.array(c_base, dtype=float)
    size = base.shape[0]

  # AlphaRank's `c` is row-stochastic with c[current, next].
  # SSD expects matrices of the form M[next, current] (columns sum to 1).
  base = base.T
  # Construct polynomial matrix: inject epsilon-uniform perturbation around the
  # base Alpha-Rank transition matrix. We produce polynomials of the form
  #   p(eps) = (1 - eps * perturbation_strength) * base + eps * perturbation_strength * U
  # where U is the uniform noise matrix (1/size).
  U_val = 1.0 / float(size)
  # Prefer an actual sparse object matrix as backing storage when requested.
  if use_sparse:
    obj_mat = sp.lil_matrix((size, size), dtype=object)
  else:
    obj_mat = np.zeros((size, size), dtype=object)
  matrix = PolynomialMatrix(obj_mat, use_sparse=use_sparse)

  for i in range(size):
    for j in range(size):
      a0 = float(base[i, j])  # constant term
      a1 = perturbation_strength * (U_val - a0)  # coefficient of epsilon
      matrix[i, j] = np.poly1d([a1, a0])
      # matrix[i, j] = a0

  # Set diagonal polynomials so that columns sum to 1 (SSD expects M[next,cur]).
  # Because alpharank's c is row-stochastic in general, ensure we normalize by
  # columns here for compatibility with SSD code which uses columns as from-states.
  for j in range(size):
    # Sum polynomials in column j
    col_sum = zero
    for i in range(size):
      if i == j:
        continue
      col_sum = col_sum + matrix[i, j]
    # diagonal = 1 - sum_offdiag
    matrix[j, j] = one - col_sum

  return matrix

def construct_br_learning_dynamics(payoff_tables: List[Any],
                                   payoffs_are_hpt_format: bool,
                                   perturbation_type: str = "uniform",
                                   perturbation_strength: float = 1.0,
                                   use_sparse: bool = False) -> PolynomialMatrix:
  """Construct perturbed Markov matrix from payoff tables.
  
  Args:
    payoff_tables: Payoff tables in Alpha-Rank format.
    payoffs_are_hpt_format: Whether tables are in HPT format.
    perturbation_type: Type of perturbation ("uniform", "trembling_hand").
    perturbation_strength: Strength of perturbation.
    
  Returns:
    Polynomial matrix representing perturbed Markov process.
  """

  # Convert to SSD format to get sizes
  game_info = ssd_utils.convert_alpharank_to_ssd_format(
      payoff_tables, payoffs_are_hpt_format)

  num_profiles = game_info['num_profiles']
  num_populations = game_info['num_populations']
  num_strats_per_population = game_info['num_strats_per_population']

  # Polynomial helpers
  epsilon = np.poly1d([1, 0])  # epsilon
  one = np.poly1d([1])
  zero = np.poly1d([0])

  m = 10.0
  alpha = 50.0

  from open_spiel.python.egt import alpharank
  # Use convenient defaults for finite-population parameters (these match
  # alpharank.compute defaults). SSD perturbation uses epsilon as an
  # independent small noise parameter, so we extract the underlying
  # transition structure from Alpha-Rank and then inject epsilon.
  if num_populations == 1:
    # Need to check whether single-population game is constant-sum to call
    # the same helper signature as alpharank.compute
    game_is_constant_sum, payoff_sum = utils.check_is_constant_sum(
        payoff_tables[0], payoffs_are_hpt_format)
    # Use alpharank helper to get the base transition matrix c and rhos
    c_base, rhos = alpharank._get_singlepop_transition_matrix(
        payoff_tables[0], payoffs_are_hpt_format,
        m=m, alpha=alpha,
        game_is_constant_sum=game_is_constant_sum,
        use_local_selection_model=True,
        payoff_sum=payoff_sum,
        use_inf_alpha=False,
        inf_alpha_eps=0.1)
    # c_base is a (num_strats x num_strats) matrix with numeric entries.
    base = np.array(c_base, dtype=float)
    size = base.shape[0]
  else:
    c_base, rhos = alpharank._get_multipop_transition_matrix(
        payoff_tables, payoffs_are_hpt_format,
        m=m, alpha=alpha,
        use_inf_alpha=False,
        inf_alpha_eps=0.1)
    base = np.array(c_base, dtype=float)
    size = base.shape[0]
  # AlphaRank's `c` is row-stochastic with c[current, next].
  # SSD expects matrices of the form M[next, current] (columns sum to 1).
  # Transpose the base matrix so indices align: matrix[next, current] = base[current, next].
  try:
    base = np.array(base, dtype=float).T
  except Exception:
    base = np.array(base, dtype=float)
  # Construct polynomial matrix: inject epsilon-uniform perturbation around the
  # base Alpha-Rank transition matrix. We produce polynomials of the form
  #   p(eps) = (1 - eps * perturbation_strength) * base + eps * perturbation_strength * U
  # where U is the uniform noise matrix (1/size).
  U_val = 1.0 / float(size)
  # Create actual sparse object backing when requested.
  if use_sparse:
    obj_mat = sp.lil_matrix((size, size), dtype=object)
  else:
    obj_mat = np.zeros((size, size), dtype=object)
  matrix = PolynomialMatrix(obj_mat, use_sparse=use_sparse)

  for i in range(size):
    for j in range(size):
      a0 = float(base[i, j])  # constant term
      a1 = perturbation_strength * (U_val - a0)  # coefficient of epsilon
      matrix[i, j] = np.poly1d([a1, a0])

  # Set diagonal polynomials so that columns sum to 1 (SSD expects M[next,cur]).
  # Because alpharank's c is row-stochastic in general, ensure we normalize by
  # columns here for compatibility with SSD code which uses columns as from-states.
  for j in range(size):
    # Sum polynomials in column j
    col_sum = zero
    for i in range(size):
      if i == j:
        continue
      col_sum = col_sum + matrix[i, j]
    # diagonal = 1 - sum_offdiag
    matrix[j, j] = one - col_sum

  return matrix


def compute_ssd(payoff_tables: List[Any],
               payoffs_are_hpt_format: bool = None,
               perturbation_type: str = "uniform",
               perturbation_strength: float = 1,
               max_iterations: int = 1000,
               verbose: bool = False,
               original_dynamics: bool = False,
               **kwargs) -> np.ndarray:
  """Compute stochastically stable distribution for given payoff tables.
  
  Args:
    payoff_tables: List of game payoff tables (same format as Alpha-Rank).
    payoffs_are_hpt_format: Whether tables are in HPT format (auto-detected if None).
    perturbation_type: Type of perturbation ("uniform", "trembling_hand").
    perturbation_strength: Strength of perturbation parameter.
    max_iterations: Maximum iterations for SSD algorithm.
    verbose: Whether to print progress information.
    simple_mode: Use simplified direct computation instead of full SSD algorithm.
    
  Returns:
    Stochastically stable distribution as numpy array.
  """
  # Auto-detect format if not specified
  if payoffs_are_hpt_format is None:
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)

  # Validate inputs
  # if not ssd_utils.check_payoffs_are_ssd_compatible(payoff_tables):
  #   print(payoff_tables)
  #   raise ValueError("Payoff tables are not compatible with SSD analysis")
  
  if verbose:
    print("Constructing perturbed Markov matrix...")
  

  num_strats_per_population = utils.get_num_strats_per_population(
      payoff_tables, payoffs_are_hpt_format)
  if np.array_equal(num_strats_per_population,
                    np.ones(len(num_strats_per_population))):
    rhos = np.asarray([[1]])
    return rhos
  
  # Construct perturbed Markov matrix
  matrix = None
  original_dynamics = False
  use_sparse = bool(kwargs.get('use_sparse', False))
  if not original_dynamics:
    matrix = construct_perturbed_markov_matrix_ev_dyn(
        payoff_tables, payoffs_are_hpt_format, perturbation_type, perturbation_strength, use_sparse=use_sparse)
  if original_dynamics:
    from open_spiel.python.egt.test_SSD import construct_perturbed_markov_matrix_hpt
    matrix = construct_perturbed_markov_matrix_hpt(payoff_tables, 1, 1)


  if verbose:
    print(f"Matrix size: {matrix.shape}")
    print("Running SSD algorithm...")
  
  # Compute SSD
  ssd_distribution = _ssd_iterate(matrix, max_iterations, verbose)
  
  # Validate result
  if not ssd_utils.validate_ssd_distribution(ssd_distribution):
    warnings.warn("Computed SSD may not be a valid probability distribution")
    ssd_distribution = ssd_utils.normalize_distribution(ssd_distribution)
  
  if verbose:
    print("SSD computation completed")
    support_size = len([x for x in ssd_distribution if x > 1e-10])
    print(f"Support size: {support_size}")
  
  return ssd_distribution

# Additional functions for compatibility with tests
def ssd_from_alpharank_format(payoff_tables: List[Any],
                             payoffs_are_hpt_format: bool,
                             **ssd_kwargs) -> np.ndarray:
  """Wrapper to compute SSD using Alpha-Rank formatted inputs."""
  return compute_ssd(payoff_tables, payoffs_are_hpt_format, **ssd_kwargs)


def compare_ssd_alpharank(payoff_tables: List[Any],
                         payoffs_are_hpt_format: bool = None,
                         m: int = 50,
                         alpha: float = 100.0,
                         ssd_kwargs: Optional[Dict[str, Any]] = None,
                         verbose: bool = False) -> Dict[str, Any]:
  """Compare SSD and Alpha-Rank results on the same game."""
  if payoffs_are_hpt_format is None:
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  
  if ssd_kwargs is None:
    ssd_kwargs = {}
  
  if verbose:
    print("Computing SSD distribution...")
  
  # Compute SSD - avoid passing verbose twice
  ssd_kwargs_copy = ssd_kwargs.copy()
  if 'verbose' not in ssd_kwargs_copy:
    ssd_kwargs_copy['verbose'] = verbose
  
  ssd_dist = compute_ssd(payoff_tables, payoffs_are_hpt_format, **ssd_kwargs_copy)
  
  # Try to compute Alpha-Rank if available
  alpharank_dist = None
  try:
    from open_spiel.python.egt import alpharank
    if verbose:
        print("Computing Alpha-Rank distribution...")
    
    _, _, alpharank_dist, _, _ = alpharank.compute(
        payoff_tables, m=m, alpha=alpha, verbose=False)
  except Exception as e:
    if verbose:
        print(f"Alpha-Rank computation failed: {e}")
  
  # Get strategy labels
  strat_labels = ssd_utils.get_ssd_strat_profile_labels(
      payoff_tables, payoffs_are_hpt_format)
  
  # Ensure strat_labels is a dictionary for comparison functions
  if isinstance(strat_labels, list):
    strat_labels = {i: f"Strategy_{i}" for i in range(len(strat_labels))}
  elif not isinstance(strat_labels, dict):
    strat_labels = {i: f"Strategy_{i}" for i in range(len(ssd_dist))}
  
  results = {
    'ssd_distribution': ssd_dist,
    'strategy_labels': strat_labels,
    'parameters': {
      'ssd': ssd_kwargs
    }
  }
  
  if alpharank_dist is not None:
    if verbose:
      print("Analyzing comparison...")
    print("ALPHARANK DIST!:", alpharank_dist)
    
    # Compare results
    comparison_results = ssd_utils.compare_rankings_ssd_alpharank(
        ssd_dist, alpharank_dist, strat_labels)
    
    results['alpharank_distribution'] = alpharank_dist
    results['comparison_metrics'] = comparison_results
    results['parameters']['alpharank'] = {'m': m, 'alpha': alpha}
    
    if verbose:
      print("\n" + "="*60)
      print("COMPARISON RESULTS")
      print("="*60)
      ssd_utils.print_comparison_summary(comparison_results)
  else:
    if verbose:
      print("Alpha-Rank not available for comparison")
  
  return results


def compute_and_report_ssd(payoff_tables: List[Any],
                          payoffs_are_hpt_format: bool = None,
                          verbose: bool = True,
                          num_top_strats_to_print: int = 8,
                          **ssd_kwargs) -> np.ndarray:
  """Compute and report SSD results (similar to Alpha-Rank's interface)."""
  if payoffs_are_hpt_format is None:
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  
  # Compute SSD
  ssd_dist = compute_ssd(payoff_tables, payoffs_are_hpt_format, 
                        verbose=verbose, **ssd_kwargs)
  
  # Get strategy labels
  strat_labels = ssd_utils.get_ssd_strat_profile_labels(
      payoff_tables, payoffs_are_hpt_format)
  
  if verbose:
    # Print results
    ssd_utils.print_ssd_rankings_table(
        payoff_tables, ssd_dist, strat_labels, num_top_strats_to_print)
    
    # Print summary statistics
    print("Summary Statistics:")
    support_size = len([x for x in ssd_dist if x > 1e-10])
    print(f"  Support size: {support_size}")
    max_prob = np.max(ssd_dist)
    print(f"  Maximum probability: {max_prob:.4f}")
    entropy = -np.sum([p * np.log(p) for p in ssd_dist if p > 1e-12])
    print(f"  Entropy: {entropy:.4f}")
  
  return ssd_dist


def ssd_vs_alpharank(payoff_matrix: np.ndarray, 
                    m: int = 50, 
                    alpha: float = 100.0,
                    **ssd_kwargs) -> Dict[str, Any]:
  """Quick comparison of SSD vs Alpha-Rank for single payoff matrix."""
  payoff_tables = [payoff_matrix]
  return compare_ssd_alpharank(
      payoff_tables, payoffs_are_hpt_format=False,
      m=m, alpha=alpha, ssd_kwargs=ssd_kwargs, verbose=True)


def analyze_tournament_dynamics(agent_payoffs: np.ndarray,
                               method: str = "ssd",
                               verbose: bool = True) -> Dict[str, Any]:
  """Analyze tournament dynamics using SSD."""
  # Convert to OpenSpiel format
  payoff_tables = [agent_payoffs]
  
  if method == "both":
    return compare_ssd_alpharank(payoff_tables, verbose=verbose)
  else:
    return {"ssd": compute_and_report_ssd(payoff_tables, verbose=verbose)}


# Export internal functions for testing
__all__ = [
    'PolynomialMatrix', 'compute_ssd', 'quick_ssd', 'validate_ssd_implementation',
    'ssd_from_alpharank_format', 'compare_ssd_alpharank', 'compute_and_report_ssd',
    'ssd_vs_alpharank', 'analyze_tournament_dynamics', 'compute_stochastically_stable_distribution',
    'ssd_analysis', '_create_polynomial', '_get_resistance_and_cost', '_has_zero_diagonal',
    '_find_communicating_classes', '_find_closed_classes', '_ssd_step', '_uniform_scale',
    'compute_evolutionary_analysis'
]

def compute_evolutionary_analysis(payoff_tables: List[Any],
                                 methods: List[str] = None,
                                 alpharank_params: Optional[Dict[str, Any]] = None,
                                 ssd_params: Optional[Dict[str, Any]] = None,
                                 compare: bool = True,
                                 verbose: bool = True) -> Dict[str, Any]:
  """Unified interface for running multiple evolutionary analysis methods.
  
  Args:
    payoff_tables: Payoff tables.
    methods: List of methods to run ("alpharank", "ssd").
    alpharank_params: Parameters for Alpha-Rank.
    ssd_params: Parameters for SSD.
    compare: Whether to compare results when multiple methods are used.
    verbose: Whether to print detailed results.
    
  Returns:
    Dictionary containing results from all requested methods.
  """
  if methods is None:
    methods = ["ssd"]
  
  if alpharank_params is None:
    alpharank_params = {"m": 50, "alpha": 100.0}
  
  if ssd_params is None:
    ssd_params = {}
  
  # Auto-detect format
  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  print("Payoffs Hpt format:", payoffs_are_hpt_format)
  
  results = {
    'methods': methods,
    'payoffs_are_hpt_format': payoffs_are_hpt_format,
    'parameters': {
      'alpharank': alpharank_params,
      'ssd': ssd_params
    }
  }
  
  # Get strategy labels
  strat_labels = ssd_utils.get_ssd_strat_profile_labels(
      payoff_tables, payoffs_are_hpt_format)
  results['strategy_labels'] = strat_labels
  
  # Run Alpha-Rank if requested
  if "alpharank" in methods:
    if verbose:
      print("Computing Alpha-Rank analysis...")
    
    try:
      from open_spiel.python.egt import alpharank
      
      _, _, alpharank_dist, _, _ = alpharank.compute(
          payoff_tables, verbose=verbose, **alpharank_params)
      
      results['alpharank'] = {
        'distribution': alpharank_dist,
        'parameters': alpharank_params
      }
      
      if verbose:
        print("\nAlpha-Rank Results:")
        utils.print_rankings_table(
            payoff_tables, alpharank_dist, strat_labels)
    except Exception as e:
      if verbose:
        print(f"Alpha-Rank computation failed: {e}")
  
  # Run SSD if requested
  if "ssd" in methods:
    if verbose:
      print("Computing SSD analysis...")
    
    ssd_dist = compute_ssd(
        payoff_tables, payoffs_are_hpt_format, 
        verbose=verbose, **ssd_params)
    print("Here's my payoff tables:", payoff_tables)
    print("HEre's the SSD DIST!!!!!:", ssd_dist)
    
    results['ssd'] = {
      'distribution': ssd_dist,
      'parameters': ssd_params
    }
    
    if verbose:
      ssd_utils.print_ssd_rankings_table(
          payoff_tables, ssd_dist, strat_labels)
  
  # Compare methods if requested and multiple methods were run
  if (compare and len(methods) > 1 and "alpharank" in methods and "ssd" in methods 
      and 'alpharank' in results and 'ssd' in results):
    if verbose:
      print("Computing method comparison...")
    
    comparison_results = ssd_utils.compare_rankings_ssd_alpharank(
        results['ssd']['distribution'],
        results['alpharank']['distribution'],
        strat_labels
    )
    
    results['comparison'] = comparison_results
    
    if verbose:
      ssd_utils.print_comparison_summary(comparison_results)
  
  return results

# game = pyspiel.load_matrix_game("matrix_pd")
# payoff_tables = utils.game_payoffs_array(game)


# payoff_tables= [heuristic_payoff_table.from_matrix_game(payoff_tables[0]),
#                 heuristic_payoff_table.from_matrix_game(payoff_tables[1].T)]
# _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)


# # Run both SSD and Alpha-Rank
# results = compute_evolutionary_analysis(
#     payoff_tables=payoff_tables,
#     methods=["ssd", "alpharank"],
#     alpharank_params={"m": 12, "alpha": 50},
#     ssd_params={},
#     compare=True,
#     verbose=True
# )
# print()





# To put in LaTeX notebook:

# Make graph for alpharank m and alpha.


# ssd is heavily parameterized. 

# use scipy sparse matrix
# split dynamics and algo.

# have graphs for each combination of ssd/arank <-> dynamics
# have some intuition on what is *created* from these dynamics
#         (the markov chains essentially)
# Show the shape of the markov chains.

# Find out how PSRO adds strategies thru RL somehow.

# operate on strategies independently instead of on the cross product (?)
# build best response for myself, not for joint.
# Need some way to predict opponent's strategies.

# figure out the psro loop / how they're adding best responses

# best response graph dynamic, adding epsilon noise

# do some research on graph neural networks (just look into it)

# make outline for lab notebook first.
# send to amy, then we can fill in things.


# try to artifically make some matrix that does not converge in alpharank, then
# see if something like that can emerge from actual dynamics


# after kuhn poker try leduc poker

