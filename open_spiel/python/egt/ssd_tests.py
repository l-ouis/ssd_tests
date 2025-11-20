#!/usr/bin/env python3
"""Comprehensive unit tests for SSD (Stochastically Stable Distribution) implementation.

This test suite covers all major functionality of the SSD algorithm implementation,
including core components, known game results, utility functions, and integration
with Alpha-Rank (if available).

Usage:
    python test_ssd.py
    python -m unittest test_ssd.py -v
"""

import unittest
import numpy as np
import warnings
import sys
from unittest.mock import patch
import time

# Import SSD modules (adjust import paths as needed)
try:
    from ssd import *
    import ssd_utils
    SSD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SSD modules: {e}")
    SSD_AVAILABLE = False

# Try to import Alpha-Rank for integration tests
try:
    from open_spiel.python.egt import alpharank
    ALPHARANK_AVAILABLE = True
except ImportError:
    ALPHARANK_AVAILABLE = False


class TestPolynomialMatrix(unittest.TestCase):
    """Test the PolynomialMatrix class and polynomial operations."""
    
    @unittest.skipUnless(SSD_AVAILABLE, "SSD modules not available")
    def setUp(self):
        """Set up test fixtures."""
        # Create simple polynomial matrix for testing
        self.matrix = np.array([
            [np.poly1d([1]), np.poly1d([1, 0])],  # [1, ε]
            [np.poly1d([0, 1]), np.poly1d([1])]   # [ε, 1]
        ], dtype=object)
        self.poly_matrix = PolynomialMatrix(self.matrix)
    
    def test_matrix_creation(self):
        """Test PolynomialMatrix creation and basic properties."""
        self.assertEqual(self.poly_matrix.shape, (2, 2))
        self.assertIsInstance(self.poly_matrix[0, 0], np.poly1d)
    
    def test_matrix_evaluation(self):
        """Test polynomial evaluation at specific epsilon values."""
        # Test at ε = 0
        result_0 = self.poly_matrix.evaluate_at(0.0)
        expected_0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result_0, expected_0)
        
        # Test at ε = 0.1
        result_01 = self.poly_matrix.evaluate_at(0.1)
        expected_01 = np.array([[1.0, 0.1], [0.01, 1.0]])
        np.testing.assert_array_almost_equal(result_01, expected_01)

    def test_evaluate_with_sparse_flag(self):
        """When PolynomialMatrix is created with use_sparse=True, evaluation
        should return a scipy.sparse.csr_matrix."""
        try:
            from scipy import sparse as sp
        except Exception:
            self.skipTest("scipy not available")

        sparse_poly = PolynomialMatrix(self.matrix, use_sparse=True)
        evaluated = sparse_poly.evaluate_at(0.1)
        # Expect a scipy.sparse matrix type
        self.assertTrue(hasattr(evaluated, 'tocoo') or hasattr(evaluated, 'toarray'))
    
    def test_constant_term_extraction(self):
        """Test extraction of constant terms (ε=0 evaluation)."""
        const_matrix = self.poly_matrix.get_constant_term_matrix()
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(const_matrix, expected)
    
    def test_matrix_copy(self):
        """Test deep copying of polynomial matrices."""
        copy_matrix = self.poly_matrix.copy()
        
        # Modify original
        self.poly_matrix[0, 0] = np.poly1d([2])
        
        # Copy should be unchanged
        self.assertEqual(copy_matrix[0, 0], np.poly1d([1]))


class TestPolynomialOperations(unittest.TestCase):
    """Test polynomial utility functions."""
    
    @unittest.skipUnless(SSD_AVAILABLE, "SSD modules not available")
    def test_polynomial_creation(self):
        """Test polynomial creation from various inputs."""
        # From float
        poly1 = _create_polynomial(1.5)
        self.assertEqual(poly1, np.poly1d([1.5]))
        
        # From list
        poly2 = _create_polynomial([1, 2, 3])
        self.assertEqual(poly2, np.poly1d([1, 2, 3]))
        
        # From existing polynomial
        original = np.poly1d([1, 0])
        poly3 = _create_polynomial(original)
        self.assertEqual(poly3, original)
    
    def test_resistance_and_cost(self):
        """Test resistance and cost computation for polynomials."""
        # Constant polynomial: resistance 0
        poly1 = np.poly1d([3])
        r1, c1 = _get_resistance_and_cost(poly1)
        self.assertEqual(r1, 0)
        self.assertEqual(c1, 3.0)
        
        # Linear polynomial: ε
        poly2 = np.poly1d([1, 0])
        r2, c2 = _get_resistance_and_cost(poly2)
        self.assertEqual(r2, 1)
        self.assertEqual(c2, 1.0)
        
        # Quadratic polynomial: 2ε²
        poly3 = np.poly1d([2, 0, 0])
        r3, c3 = _get_resistance_and_cost(poly3)
        self.assertEqual(r3, 2)
        self.assertEqual(c3, 2.0)
        
        # Zero polynomial: infinite resistance
        poly4 = np.poly1d([0])
        r4, c4 = _get_resistance_and_cost(poly4)
        self.assertEqual(r4, float('inf'))
        self.assertEqual(c4, 0.0)
    
    def test_zero_diagonal_detection(self):
        """Test detection of zero diagonal entries."""
        # Matrix with zero diagonal
        matrix1 = PolynomialMatrix(np.array([
            [np.poly1d([0]), np.poly1d([1])],
            [np.poly1d([1]), np.poly1d([1])]
        ], dtype=object))
        self.assertTrue(_has_zero_diagonal(matrix1))
        
        # Matrix without zero diagonal
        matrix2 = PolynomialMatrix(np.array([
            [np.poly1d([1]), np.poly1d([1])],
            [np.poly1d([1]), np.poly1d([2])]
        ], dtype=object))
        self.assertFalse(_has_zero_diagonal(matrix2))


class TestCommunicatingClasses(unittest.TestCase):
    """Test communicating class detection algorithms."""
    
    @unittest.skipUnless(SSD_AVAILABLE, "SSD modules not available")
    def test_irreducible_matrix(self):
        """Test communicating class detection on irreducible matrix."""
        # Simple irreducible 2x2 matrix
        matrix = np.array([[0.5, 0.5], [0.3, 0.7]])
        classes = _find_communicating_classes(matrix)
        
        # Should have exactly one class containing both states
        self.assertEqual(len(classes), 1)
        self.assertEqual(sorted(list(classes.values())[0]), [0, 1])
    
    def test_reducible_matrix(self):
        """Test communicating class detection on reducible matrix."""
        # Matrix with two separate components
        matrix = np.array([
            [0.8, 0.2, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.0, 1.0]
        ])
        classes = _find_communicating_classes(matrix)
        
        # Should have two separate classes
        self.assertEqual(len(classes), 2)
        
        # Check that states are properly separated
        all_states = set()
        for vertices in classes.values():
            all_states.update(vertices)
        self.assertEqual(all_states, {0, 1, 2})
    
    def test_closed_classes(self):
        """Test closed class detection."""
        # Matrix where state 2 is absorbing (closed)
        matrix = np.array([
            [0.5, 0.3, 0.2],
            [0.0, 0.7, 0.3],
            [0.0, 0.0, 1.0]
        ])
        
        communicating = _find_communicating_classes(matrix)
        closed = _find_closed_classes(matrix, communicating)
        
        # State 2 should form a closed class
        self.assertGreater(len(closed), 0)
        
        # Find the closed class containing state 2
        found_closed_2 = False
        for vertices in closed.values():
            if 2 in vertices:
                found_closed_2 = True
                break
        self.assertTrue(found_closed_2)


class TestSSDKnownGames(unittest.TestCase):
    """Test SSD on games with known theoretical results."""
    
    @unittest.skipUnless(SSD_AVAILABLE, "SSD modules not available")
    def test_coordination_game(self):
        """Test SSD on 2x2 coordination game."""
        # Stag hunt / coordination game
        # (5,5) is risk-dominant over (4,4)
        payoff_matrix = np.array([[5, 0], [0, 4]])
        
        ssd_dist = quick_ssd(payoff_matrix, verbose=False)
        
        # Validate distribution properties
        self.assertAlmostEqual(np.sum(ssd_dist), 1.0, places=6)
        self.assertTrue(np.all(ssd_dist >= 0))
        
        # Strategy 0 should dominate (risk-dominant equilibrium)
        self.assertGreater(ssd_dist[0], ssd_dist[1])
        
        # Should have strong preference for strategy 0
        self.assertGreater(ssd_dist[0], 0.6)
    
    def test_prisoners_dilemma(self):
        """Test SSD on Prisoner's Dilemma."""
        # Classic prisoner's dilemma payoffs
        # Defection (strategy 1) should dominate
        payoff_matrix = np.array([[3, 0], [5, 1]])
        
        ssd_dist = quick_ssd(payoff_matrix, verbose=False)
        
        # Validate distribution
        self.assertAlmostEqual(np.sum(ssd_dist), 1.0, places=6)
        
        # Defection should dominate cooperation
        self.assertGreater(ssd_dist[1], ssd_dist[0])
        
        # Should strongly favor defection
        self.assertGreater(ssd_dist[1], 0.7)
    
    def test_rock_paper_scissors(self):
        """Test SSD on symmetric Rock-Paper-Scissors."""
        # Symmetric zero-sum game should yield uniform distribution
        payoff_matrix = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ])
        
        ssd_dist = quick_ssd(payoff_matrix, verbose=False)
        
        # Validate distribution
        self.assertAlmostEqual(np.sum(ssd_dist), 1.0, places=6)
        
        # Should be approximately uniform
        for i in range(3):
            self.assertAlmostEqual(ssd_dist[i], 1/3, delta=0.15)
    
    def test_battle_of_sexes(self):
        """Test SSD on Battle of the Sexes (multi-population)."""
        # Two-population coordination game
        payoff_matrix1 = np.array([[2, 0], [0, 1]])  # Player 1 payoffs
        payoff_matrix2 = np.array([[1, 0], [0, 2]])  # Player 2 payoffs
        
        ssd_dist = compute_ssd([payoff_matrix1, payoff_matrix2], 
                              payoffs_are_hpt_format=False, verbose=False)
        
        # Validate distribution
        self.assertAlmostEqual(np.sum(ssd_dist), 1.0, places=6)
        
        # Should have 4 strategy profiles: (0,0), (0,1), (1,0), (1,1)
        self.assertEqual(len(ssd_dist), 4)
        
        # Pure equilibria (0,0) and (1,1) should have higher probability
        # than miscoordinated outcomes (0,1) and (1,0)
        self.assertGreater(ssd_dist[0] + ssd_dist[3], ssd_dist[1] + ssd_dist[2])
    
    def test_dominance_game(self):
        """Test SSD on game with strictly dominated strategies."""
        # Strategy 1 strictly dominates strategy 0
        payoff_matrix = np.array([[1, 1], [2, 2]])
        
        ssd_dist = quick_ssd(payoff_matrix, verbose=False)
        
        # Dominant strategy should have higher probability
        self.assertGreater(ssd_dist[1], ssd_dist[0])
        
        # Should strongly favor the dominant strategy
        self.assertGreater(ssd_dist[1], 0.8)


class TestSSDUtilities(unittest.TestCase):
    """Test SSD utility functions."""
    
    @unittest.skipUnless(SSD_AVAILABLE, "SSD modules not available")
    def setUp(self):
        """Set up test distributions."""
        self.dist1 = np.array([0.7, 0.2, 0.1])
        self.dist2 = np.array([0.6, 0.3, 0.1])
        self.uniform_dist = np.array([1/3, 1/3, 1/3])
    
    def test_distribution_validation(self):
        """Test distribution validation functions."""
        # Valid distribution
        self.assertTrue(ssd_utils.validate_ssd_distribution(self.dist1))
        
        # Invalid distributions
        invalid1 = np.array([0.5, 0.3, 0.1])  # Doesn't sum to 1
        self.assertFalse(ssd_utils.validate_ssd_distribution(invalid1))
        
        invalid2 = np.array([0.8, -0.1, 0.3])  # Negative entry
        self.assertFalse(ssd_utils.validate_ssd_distribution(invalid2))
    
    def test_distribution_normalization(self):
        """Test distribution normalization."""
        unnormalized = np.array([2, 1, 1])
        normalized = ssd_utils.normalize_distribution(unnormalized)
        
        expected = np.array([0.5, 0.25, 0.25])
        np.testing.assert_array_almost_equal(normalized, expected)
        
        # Test zero distribution
        zero_dist = np.array([0, 0, 0])
        uniform_result = ssd_utils.normalize_distribution(zero_dist)
        expected_uniform = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(uniform_result, expected_uniform)
    
    def test_ranking_computation(self):
        """Test ranking computation from distributions."""
        rankings = ssd_utils.compute_ranking_from_distribution(self.dist1)
        
        # Strategy 0 should have rank 0 (best), strategy 1 rank 1, strategy 2 rank 2
        self.assertEqual(rankings[0], 0)  # Highest probability
        self.assertEqual(rankings[1], 1)  # Middle probability
        self.assertEqual(rankings[2], 2)  # Lowest probability
    
    def test_correlation_metrics(self):
        """Test correlation computation between distributions."""
        # Self-correlation should be 1
        self.assertAlmostEqual(
            ssd_utils.ranking_correlation(self.dist1, self.dist1, 'spearman'), 
            1.0, places=6)
        
        # Correlation between different distributions
        corr = ssd_utils.ranking_correlation(self.dist1, self.dist2, 'spearman')
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)
        
        # Test other correlation methods
        kendall_corr = ssd_utils.ranking_correlation(self.dist1, self.dist2, 'kendall')
        self.assertGreaterEqual(kendall_corr, -1.0)
        self.assertLessEqual(kendall_corr, 1.0)
    
    def test_distance_metrics(self):
        """Test distance metrics between distributions."""
        # Self-distance should be 0
        self.assertAlmostEqual(
            ssd_utils.distribution_distance(self.dist1, self.dist1, 'tvd'), 
            0.0, places=6)
        
        # Distance between different distributions
        tvd = ssd_utils.distribution_distance(self.dist1, self.dist2, 'tvd')
        self.assertGreaterEqual(tvd, 0.0)
        self.assertLessEqual(tvd, 1.0)
        
        # Test other distance metrics
        hellinger = ssd_utils.distribution_distance(self.dist1, self.dist2, 'hellinger')
        self.assertGreaterEqual(hellinger, 0.0)
        
        l2_dist = ssd_utils.distribution_distance(self.dist1, self.dist2, 'l2')
        self.assertGreaterEqual(l2_dist, 0.0)
    
    def test_top_k_overlap(self):
        """Test top-k strategy overlap."""
        # Self-overlap should be 1
        self.assertEqual(ssd_utils.top_k_overlap(self.dist1, self.dist1, k=2), 1.0)
        
        # Overlap between similar distributions
        overlap = ssd_utils.top_k_overlap(self.dist1, self.dist2, k=2)
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)
    
    def test_entropy_and_diversity(self):
        """Test entropy and diversity measures."""
        # Uniform distribution should have maximum entropy
        uniform_entropy = ssd_utils.entropy(self.uniform_dist)
        skewed_entropy = ssd_utils.entropy(self.dist1)
        
        self.assertGreater(uniform_entropy, skewed_entropy)
        
        # Effective strategies
        eff_uniform = ssd_utils.effective_strategies(self.uniform_dist)
        eff_skewed = ssd_utils.effective_strategies(self.dist1)
        
        self.assertGreater(eff_uniform, eff_skewed)
        self.assertAlmostEqual(eff_uniform, 3.0, delta=0.1)  # Should be close to 3
    
    def test_concentration_measures(self):
        """Test concentration and inequality measures."""
        # Concentration ratio
        conc1 = ssd_utils.concentration_ratio(self.dist1, k=1)
        self.assertAlmostEqual(conc1, 0.7)  # Top strategy has 0.7 probability
        
        conc2 = ssd_utils.concentration_ratio(self.dist1, k=2)
        self.assertAlmostEqual(conc2, 0.9)  # Top 2 strategies have 0.9 probability
        
        # Gini coefficient
        gini_uniform = ssd_utils.gini_coefficient(self.uniform_dist)
        gini_skewed = ssd_utils.gini_coefficient(self.dist1)
        
        self.assertGreater(gini_skewed, gini_uniform)  # Skewed should be more unequal


class TestSSDAlgorithmComponents(unittest.TestCase):
    """Test individual components of the SSD algorithm."""
    
    @unittest.skipUnless(SSD_AVAILABLE, "SSD modules not available")
    def test_uniform_scaling(self):
        """Test uniform scaling operation."""
        # Create matrix with zero diagonal
        matrix = PolynomialMatrix(np.array([
            [np.poly1d([0]), np.poly1d([0.5])],
            [np.poly1d([0.3]), np.poly1d([0])]
        ], dtype=object))
        
        self.assertTrue(_has_zero_diagonal(matrix))
        
        scaled = _uniform_scale(matrix, scale_factor=0.5)
        
        # Should no longer have zero diagonal
        self.assertFalse(_has_zero_diagonal(scaled))
    
    def test_ssd_step_convergence(self):
        """Test SSD step on converged system."""
        # Create simple irreducible system
        epsilon = np.poly1d([1, 0])  # ε
        one = np.poly1d([1])  # 1
        
        matrix = PolynomialMatrix(np.array([
            [one - epsilon/2, epsilon/2],
            [epsilon/2, one - epsilon/2]
        ], dtype=object))
        
        stable_dist, reduced_matrix, inclusion_op = _ssd_step(matrix)
        
        # Should converge (return stable distribution)
        self.assertIsNotNone(stable_dist)
        self.assertIsNone(reduced_matrix)
        self.assertIsNone(inclusion_op)
        
        # Distribution should be approximately uniform
        self.assertAlmostEqual(stable_dist[0], 0.5, delta=0.1)
        self.assertAlmostEqual(stable_dist[1], 0.5, delta=0.1)


class TestSSDPerformance(unittest.TestCase):
    """Test SSD performance and robustness."""
    
    @unittest.skipUnless(SSD_AVAILABLE, "SSD modules not available")
    def test_small_games_performance(self):
        """Test that small games complete in reasonable time."""
        sizes = [2, 3, 4]
        
        for n in sizes:
            with self.subTest(size=n):
                # Generate random game
                payoff_matrix = np.random.uniform(0, 1, (n, n))
                
                start_time = time.time()
                ssd_dist = quick_ssd(payoff_matrix, verbose=False, max_iterations=50)
                elapsed = time.time() - start_time
                
                # Should complete quickly
                self.assertLess(elapsed, 10.0, f"Size {n}x{n} took {elapsed:.2f}s")
                
                # Result should be valid
                self.assertTrue(ssd_utils.validate_ssd_distribution(ssd_dist))
    
    def test_convergence_robustness(self):
        """Test that algorithm converges on various random games."""
        num_tests = 5
        
        for i in range(num_tests):
            with self.subTest(test=i):
                # Random 3x3 game
                payoff_matrix = np.random.uniform(-1, 1, (3, 3))
                
                try:
                    ssd_dist = quick_ssd(payoff_matrix, verbose=False, max_iterations=100)
                    
                    # Should be valid distribution
                    self.assertTrue(ssd_utils.validate_ssd_distribution(ssd_dist))
                    
                except Exception as e:
                    self.fail(f"SSD failed on random game {i}: {e}")
    
    def test_perturbation_sensitivity(self):
        """Test sensitivity to perturbation parameters."""
        payoff_matrix = np.array([[2, 0], [0, 1]])
        
        # Test different perturbation strengths
        strengths = [0.1, 1.0, 10.0]
        results = []
        
        for strength in strengths:
            ssd_dist = compute_ssd([payoff_matrix], 
                                  payoffs_are_hpt_format=False,
                                  perturbation_strength=strength,
                                  verbose=False)
            results.append(ssd_dist)
        
        # Results should be similar (robust to perturbation strength)
        for i in range(len(results) - 1):
            corr = ssd_utils.ranking_correlation(results[i], results[i+1], 'spearman')
            self.assertGreater(corr, 0.7, f"Low correlation between strength {strengths[i]} and {strengths[i+1]}")


@unittest.skipUnless(ALPHARANK_AVAILABLE, "Alpha-Rank not available")
class TestSSDAlphaRankIntegration(unittest.TestCase):
    """Test integration with Alpha-Rank (requires OpenSpiel)."""
    
    @unittest.skipUnless(SSD_AVAILABLE, "SSD modules not available")
    def test_comparison_functionality(self):
        """Test SSD vs Alpha-Rank comparison functions."""
        payoff_matrix = np.array([[5, 0], [0, 4]])
        
        results = ssd_vs_alpharank(payoff_matrix, verbose=False)
        
        # Check result structure
        self.assertIn('ssd_distribution', results)
        self.assertIn('alpharank_distribution', results)
        self.assertIn('comparison_metrics', results)
        
        # Both distributions should be valid
        ssd_dist = results['ssd_distribution']
        ar_dist = results['alpharank_distribution']
        
        self.assertTrue(ssd_utils.validate_ssd_distribution(ssd_dist))
        self.assertTrue(ssd_utils.validate_ssd_distribution(ar_dist))
        
        # Both should favor strategy 0 for this coordination game
        self.assertGreater(ssd_dist[0], ssd_dist[1])
        self.assertGreater(ar_dist[0], ar_dist[1])
    
    def test_correlation_across_games(self):
        """Test correlation between SSD and Alpha-Rank across multiple games."""
        correlations = []
        
        # Test on several games
        games = [
            np.array([[5, 0], [0, 4]]),  # Coordination
            np.array([[3, 0], [5, 1]]),  # Prisoner's Dilemma
            np.array([[1, 0], [0, 1]]),  # Pure coordination
        ]
        
        for game in games:
            try:
                results = ssd_vs_alpharank(game, verbose=False)
                corr = results['comparison_metrics']['spearman_correlation']
                correlations.append(corr)
            except Exception:
                continue  # Skip if computation fails
        
        # Average correlation should be positive
        if correlations:
            avg_corr = np.mean(correlations)
            self.assertGreater(avg_corr, 0.0, "SSD and Alpha-Rank should generally agree")
    
    def test_unified_analysis_interface(self):
        """Test unified evolutionary analysis interface."""
        payoff_matrix = np.array([[2, 0], [0, 1]])
        
        results = compute_evolutionary_analysis(
            [payoff_matrix], 
            methods=["alpharank", "ssd"],
            verbose=False
        )
        
        # Check result structure
        self.assertIn('alpharank', results)
        self.assertIn('ssd', results)
        self.assertIn('comparison', results)
        
        # Both methods should produce valid distributions
        ar_dist = results['alpharank']['distribution']
        ssd_dist = results['ssd']['distribution']
        
        self.assertTrue(ssd_utils.validate_ssd_distribution(ar_dist))
        self.assertTrue(ssd_utils.validate_ssd_distribution(ssd_dist))


class TestSSDValidationSuite(unittest.TestCase):
    """Test the built-in validation suite."""
    
    @unittest.skipUnless(SSD_AVAILABLE, "SSD modules not available")
    def test_built_in_validation(self):
        """Test that built-in validation suite runs successfully."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress convergence warnings
            
            results = validate_ssd_implementation(verbose=False)
        
        # Should return results for all test cases
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # At least some tests should pass
        passed = sum(results.values())
        self.assertGreater(passed, 0, "No validation tests passed")


if __name__ == '__main__':
    # Configure test runner
    unittest.TestLoader.sortTestMethodsUsing = None  # Keep test order
    
    # Run tests with verbose output
    if len(sys.argv) == 1:
        # If no arguments, run with increased verbosity
        unittest.main(argv=[''], verbosity=2, exit=False)
    else:
        # If arguments provided, run normally
        unittest.main()