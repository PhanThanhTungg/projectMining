import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import math
import heapq
from scipy import sparse
from numba import jit, prange, njit
import warnings
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings("ignore")

@njit(parallel=True)
def parallel_digitize(values, bin_edges):
    """Parallel histogram binning using numba"""
    n = len(values)
    result = np.zeros(n, dtype=np.uint8)
    n_bins = len(bin_edges) - 1
    
    for i in prange(n):  # Parallel loop
        val = values[i]
        # Binary search for bin
        left, right = 0, n_bins
        while left < right:
            mid = (left + right) // 2
            if val <= bin_edges[mid]:
                right = mid
            else:
                left = mid + 1
        result[i] = min(left, n_bins - 1)
    return result

@njit(parallel=True)
def parallel_histogram_stats(bin_indices, gradients, hessians, n_bins):
    """Parallel histogram building using numba"""
    grad_sum = np.zeros(n_bins)
    hess_sum = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int32)
    
    # Use parallel reduction pattern
    for i in prange(len(bin_indices)):
        bin_idx = bin_indices[i]
        # Atomic operations for thread safety
        grad_sum[bin_idx] += gradients[i]
        hess_sum[bin_idx] += hessians[i]
        counts[bin_idx] += 1
    
    return grad_sum, hess_sum, counts

@njit
def ultra_fast_gain_calculation(grad_sums, hess_sums, counts, reg_lambda):
    """Ultra-fast gain calculation with early stopping"""
    n_bins = len(grad_sums)
    gains = np.full(n_bins - 1, -np.inf)
    
    total_grad = 0.0
    total_hess = 0.0
    total_count = 0
    
    # Pre-compute totals
    for i in range(n_bins):
        total_grad += grad_sums[i]
        total_hess += hess_sums[i]
        total_count += counts[i]
    
    if total_count < 2:  # Early exit
        return gains
    
    cumsum_grad = 0.0
    cumsum_hess = 0.0
    cumsum_count = 0
    
    for split_idx in range(n_bins - 1):
        cumsum_grad += grad_sums[split_idx]
        cumsum_hess += hess_sums[split_idx]
        cumsum_count += counts[split_idx]
        
        left_count = cumsum_count
        right_count = total_count - left_count
        
        if left_count >= 1 and right_count >= 1:
            left_grad = cumsum_grad
            left_hess = cumsum_hess
            right_grad = total_grad - left_grad
            right_hess = total_hess - left_hess
            
            # Vectorized gain calculation
            left_gain = (left_grad * left_grad) / (left_hess + reg_lambda)
            right_gain = (right_grad * right_grad) / (right_hess + reg_lambda)
            parent_gain = (total_grad * total_grad) / (total_hess + reg_lambda)
            
            gains[split_idx] = 0.5 * (left_gain + right_gain - parent_gain)
    
    return gains

@njit(parallel=True)
def parallel_softmax(x):
    """Parallel softmax computation"""
    n_samples, n_classes = x.shape
    result = np.zeros_like(x)
    
    for i in prange(n_samples):
        row = x[i]
        max_val = np.max(row)
        shifted = row - max_val
        exp_vals = np.exp(np.clip(shifted, -500, 500))
        sum_exp = np.sum(exp_vals)
        result[i] = exp_vals / sum_exp
    
    return result

class UltraFastHistogramBuilder:
    """Ultra-optimized histogram builder với parallel processing"""
    
    def __init__(self, max_bins=255, n_jobs=-1):
        self.max_bins = max_bins
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.bin_mappers = {}
        self.bin_edges_array = None
        self.bin_counts = None
        
    def build_feature_histograms_parallel(self, X):
        """Parallel histogram building"""
        n_samples, n_features = X.shape
        
        # Pre-allocate
        max_edges = self.max_bins + 1
        self.bin_edges_array = np.zeros((n_features, max_edges))
        self.bin_counts = np.zeros(n_features, dtype=np.int32)
        
        # Parallel processing for features
        def process_feature(args):
            feature_idx, feature_values = args
            unique_vals = np.unique(feature_values)
            
            if len(unique_vals) <= self.max_bins:
                bin_edges = np.concatenate([unique_vals, [unique_vals[-1] + 1e-10]])
            else:
                # Fast quantile calculation
                quantiles = np.linspace(0, 1, self.max_bins + 1)
                bin_edges = np.quantile(feature_values, quantiles)
                bin_edges[-1] += 1e-10
            
            return feature_idx, bin_edges
        
        # Process features in parallel
        feature_args = [(i, X[:, i]) for i in range(n_features)]
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(process_feature, feature_args))
        
        # Store results
        for feature_idx, bin_edges in results:
            n_edges = len(bin_edges)
            self.bin_edges_array[feature_idx, :n_edges] = bin_edges
            self.bin_counts[feature_idx] = n_edges
            self.bin_mappers[feature_idx] = bin_edges
    
    def map_to_bins_parallel(self, X):
        """Parallel binning"""
        n_samples, n_features = X.shape
        binned_X = np.zeros((n_samples, n_features), dtype=np.uint8)
        
        for feature_idx in range(n_features):
            n_edges = self.bin_counts[feature_idx]
            bin_edges = self.bin_edges_array[feature_idx, :n_edges]
            binned_X[:, feature_idx] = parallel_digitize(X[:, feature_idx], bin_edges)
        
        return binned_X

class UltraFastGradientHistogram:
    """Ultra-optimized gradient histogram"""
    
    def __init__(self, max_bins=255):
        self.max_bins = max_bins
        self.gradient_sum = np.zeros(max_bins)
        self.hessian_sum = np.zeros(max_bins)
        self.sample_count = np.zeros(max_bins, dtype=np.int32)
    
    def build_histogram_ultra_fast(self, bin_indices, gradients, hessians, n_bins):
        """Ultra-fast histogram building"""
        self.gradient_sum[:n_bins] = 0
        self.hessian_sum[:n_bins] = 0
        self.sample_count[:n_bins] = 0
        
        grad_sums, hess_sums, counts = parallel_histogram_stats(
            bin_indices, gradients, hessians, n_bins
        )
        
        self.gradient_sum[:n_bins] = grad_sums
        self.hessian_sum[:n_bins] = hess_sums
        self.sample_count[:n_bins] = counts
        
        return self
    
    def calculate_gain_ultra_fast(self, n_bins, reg_lambda):
        """Ultra-fast gain calculation"""
        return ultra_fast_gain_calculation(
            self.gradient_sum[:n_bins],
            self.hessian_sum[:n_bins], 
            self.sample_count[:n_bins],
            reg_lambda
        )

class SuperFastLightGBMTree:
    """Super-optimized LightGBM tree"""
    
    def __init__(self, max_depth=6, min_child_samples=20, reg_lambda=0.1, 
                 histogram_builder=None, max_leaves=31):
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.histogram_builder = histogram_builder
        self.max_leaves = max_leaves
        
        self.root = None
        self.leaf_nodes = []
        self.histogram = UltraFastGradientHistogram()
        
    def fit_ultra_fast(self, X_binned, gradients, hessians):
        """Ultra-fast training"""
        n_samples = len(gradients)
        
        # Initialize root
        self.root = LightGBMTreeNode(depth=0, samples_idx=np.arange(n_samples))
        self._update_node_stats_vectorized(self.root, gradients, hessians)
        
        self.leaf_nodes = [self.root]
        num_leaves = 1
        
        while num_leaves < self.max_leaves and self.leaf_nodes:
            # Find best split using optimized search
            best_leaf, best_split = self._find_best_split_ultra_fast(
                X_binned, gradients, hessians
            )
            
            if best_leaf is None or best_split['gain'] <= 1e-6:  # Stricter threshold
                break
            
            left_child, right_child = self._split_leaf_ultra_fast(
                best_leaf, best_split, X_binned, gradients, hessians
            )
            
            # Update efficiently
            self.leaf_nodes.remove(best_leaf)
            if left_child.depth < self.max_depth and left_child.sample_count >= self.min_child_samples:
                self.leaf_nodes.append(left_child)
            if right_child.depth < self.max_depth and right_child.sample_count >= self.min_child_samples:
                self.leaf_nodes.append(right_child)
            
            num_leaves += 1
        
        self._calculate_leaf_values_fast()
    
    def _update_node_stats_vectorized(self, node, gradients, hessians):
        """Ultra-fast stats update"""
        if len(node.samples_idx) > 0:
            indices = node.samples_idx
            node.gradient_sum = np.sum(gradients[indices])
            node.hessian_sum = np.sum(hessians[indices])
            node.sample_count = len(indices)
    
    def _find_best_split_ultra_fast(self, X_binned, gradients, hessians):
        """Ultra-fast split finding with early stopping"""
        best_leaf = None
        best_split = {'gain': -np.inf}
        
        # Sort leaves by potential gain (heuristic)
        leaf_potentials = []
        for leaf in self.leaf_nodes:
            if (leaf.sample_count < self.min_child_samples * 2 or 
                leaf.depth >= self.max_depth):
                continue
            
            # Quick potential estimate
            potential = abs(leaf.gradient_sum) / (leaf.hessian_sum + self.reg_lambda)
            leaf_potentials.append((potential, leaf))
        
        # Sort and process only top candidates
        leaf_potentials.sort(reverse=True)
        max_candidates = min(len(leaf_potentials), 5)  # Limit search
        
        for _, leaf in leaf_potentials[:max_candidates]:
            split_info = self._find_best_split_for_leaf_ultra_fast(
                leaf, X_binned, gradients, hessians
            )
            if split_info['gain'] > best_split['gain']:
                best_split = split_info
                best_leaf = leaf
        
        return best_leaf, best_split
    
    def _find_best_split_for_leaf_ultra_fast(self, leaf, X_binned, gradients, hessians):
        """Ultra-optimized split finding"""
        best_split = {'gain': -np.inf}
        n_features = X_binned.shape[1]
        samples_idx = leaf.samples_idx
        
        if len(samples_idx) < self.min_child_samples * 2:
            return best_split
        
        # Pre-compute all data
        X_samples = X_binned[samples_idx]
        grad_samples = gradients[samples_idx]
        hess_samples = hessians[samples_idx]
        
        # Limit feature search for speed
        feature_indices = np.arange(n_features)
        if n_features > 10:  # Random sample features for large datasets
            n_sample_features = max(int(np.sqrt(n_features)), 5)
            feature_indices = np.random.choice(n_features, n_sample_features, replace=False)
        
        for feature_idx in feature_indices:
            feature_bins = X_samples[:, feature_idx]
            unique_bins = np.unique(feature_bins)
            
            if len(unique_bins) < 2:
                continue
            
            max_bin = int(np.max(feature_bins)) + 1
            
            # Ultra-fast histogram
            self.histogram.build_histogram_ultra_fast(
                feature_bins, grad_samples, hess_samples, max_bin
            )
            
            # Ultra-fast gain calculation
            gains = self.histogram.calculate_gain_ultra_fast(max_bin, self.reg_lambda)
            
            if len(gains) > 0:
                best_bin_idx = np.argmax(gains)
                max_gain = gains[best_bin_idx]
                
                if max_gain > best_split['gain']:
                    if feature_idx in self.histogram_builder.bin_mappers:
                        bin_edges = self.histogram_builder.bin_mappers[feature_idx]
                        raw_threshold = bin_edges[min(best_bin_idx + 1, len(bin_edges) - 1)]
                        
                        best_split = {
                            'gain': max_gain,
                            'feature_idx': feature_idx,
                            'bin_threshold': best_bin_idx,
                            'raw_threshold': raw_threshold
                        }
        
        return best_split
    
    def _split_leaf_ultra_fast(self, leaf, split_info, X_binned, gradients, hessians):
        """Ultra-fast leaf splitting"""
        feature_idx = split_info['feature_idx']
        bin_threshold = split_info['bin_threshold']
        
        samples_idx = leaf.samples_idx
        feature_bins = X_binned[samples_idx, feature_idx]
        
        # Vectorized splitting
        left_mask = feature_bins <= bin_threshold
        left_samples = samples_idx[left_mask]
        right_samples = samples_idx[~left_mask]
        
        # Quick validation
        if len(left_samples) < self.min_child_samples or len(right_samples) < self.min_child_samples:
            # Return dummy nodes that won't be added
            left_child = LightGBMTreeNode(depth=leaf.depth + 1, samples_idx=np.array([]))
            right_child = LightGBMTreeNode(depth=leaf.depth + 1, samples_idx=np.array([]))
            return left_child, right_child
        
        # Create children
        left_child = LightGBMTreeNode(depth=leaf.depth + 1, samples_idx=left_samples)
        right_child = LightGBMTreeNode(depth=leaf.depth + 1, samples_idx=right_samples)
        
        # Fast stats update
        self._update_node_stats_vectorized(left_child, gradients, hessians)
        self._update_node_stats_vectorized(right_child, gradients, hessians)
        
        # Update parent
        leaf.is_leaf = False
        leaf.feature_idx = feature_idx
        leaf.bin_threshold = bin_threshold
        leaf.raw_threshold = split_info['raw_threshold']
        leaf.left = left_child
        leaf.right = right_child
        
        left_child.parent = leaf
        right_child.parent = leaf
        
        return left_child, right_child
    
    def _calculate_leaf_values_fast(self):
        """Fast leaf value calculation"""
        def calculate_recursive(node):
            if node.is_leaf:
                if node.hessian_sum > 1e-10:
                    node.value = -node.gradient_sum / (node.hessian_sum + self.reg_lambda)
                else:
                    node.value = 0.0
            else:
                calculate_recursive(node.left)
                calculate_recursive(node.right)
        
        calculate_recursive(self.root)
    
    def predict_ultra_fast(self, X_binned):
        """Ultra-fast prediction using vectorized operations"""
        n_samples = len(X_binned)
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            node = self.root
            while not node.is_leaf:
                if X_binned[i, node.feature_idx] <= node.bin_threshold:
                    node = node.left
                else:
                    node = node.right
            predictions[i] = node.value
        
        return predictions

class LightGBMTreeNode:
    def __init__(self, depth=0, samples_idx=None):
        self.feature_idx = None
        self.bin_threshold = None
        self.raw_threshold = None
        self.left = None
        self.right = None
        self.parent = None
        self.is_leaf = True
        self.depth = depth
        self.samples_idx = samples_idx if samples_idx is not None else []
        self.value = 0.0
        self.gradient_sum = 0.0
        self.hessian_sum = 0.0
        self.sample_count = 0
        self.split_gain = 0.0

class TurboLightGBMClassifier:
    """TURBO-CHARGED LightGBM với mọi optimization có thể"""
    
    def __init__(self, 
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=6,
                 num_leaves=31,
                 min_child_samples=20,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 reg_lambda=0.1,
                 max_bins=255,
                 n_jobs=-1,
                 random_state=None):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.max_bins = max_bins
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Ultra-optimized components
        self.histogram_builder = UltraFastHistogramBuilder(max_bins=max_bins, n_jobs=n_jobs)
        self.trees = []
        self.feature_indices = []
        
        # Multiclass setup
        self.n_classes = None
        self.classes_ = None
        self.initial_predictions = None
        
        # Massive pre-allocation
        self._F_cache = None
        self._grad_cache = None
        self._hess_cache = None
        self._y_onehot_cache = None
        
        if random_state:
            np.random.seed(random_state)
    
    def _compute_gradients_hessians_turbo(self, y_onehot, F):
        """Turbo gradient/hessian computation"""
        probabilities = parallel_softmax(F)
        gradients = probabilities - y_onehot
        hessians = probabilities * (1 - probabilities)
        return gradients, hessians
    
    def fit(self, X, y):
        start_time = time.time()
        
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int32)
        
        # Setup
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # TURBO histogram building
        hist_start = time.time()
        self.histogram_builder.build_feature_histograms_parallel(X)
        X_binned = self.histogram_builder.map_to_bins_parallel(X)
        hist_time = time.time() - hist_start
        print(f"TURBO histograms: {hist_time:.2f}s")
        
        # Massive pre-allocation
        self._F_cache = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        self._grad_cache = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        self._hess_cache = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        
        # Initialize predictions
        class_counts = np.bincount(y, minlength=self.n_classes)
        class_probs = class_counts / n_samples
        self.initial_predictions = np.log(class_probs + 1e-15)
        self.initial_predictions -= self.initial_predictions[0]
        
        # Ultra-fast F initialization
        self._F_cache[:] = self.initial_predictions[np.newaxis, :]
        
        # Pre-compute one-hot encoding
        self._y_onehot_cache = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        self._y_onehot_cache[np.arange(n_samples), y] = 1
        
        print(f"TURBO training {self.n_estimators} iterations...")
        train_start = time.time()
        
        # TURBO training loop
        for iteration in range(self.n_estimators):
            iteration_trees = []
            iteration_features = []
            
            # TURBO gradient computation
            gradients, hessians = self._compute_gradients_hessians_turbo(
                self._y_onehot_cache, self._F_cache
            )
            
            # Train trees for all classes
            for class_idx in range(self.n_classes):
                # Smart sampling
                if self.subsample < 1.0:
                    n_samples_tree = max(1, int(n_samples * self.subsample))
                    sample_indices = np.random.choice(n_samples, n_samples_tree, replace=False)
                else:
                    sample_indices = np.arange(n_samples)
                
                if self.colsample_bytree < 1.0:
                    n_features_tree = max(1, int(n_features * self.colsample_bytree))
                    feature_indices = np.random.choice(n_features, n_features_tree, replace=False)
                else:
                    feature_indices = np.arange(n_features)
                
                # Extract data efficiently
                class_gradients = gradients[sample_indices, class_idx]
                class_hessians = hessians[sample_indices, class_idx]
                X_binned_subset = X_binned[np.ix_(sample_indices, feature_indices)]
                
                # Create TURBO tree
                tree_histogram_builder = UltraFastHistogramBuilder(max_bins=self.max_bins)
                for new_idx, orig_idx in enumerate(feature_indices):
                    if orig_idx in self.histogram_builder.bin_mappers:
                        tree_histogram_builder.bin_mappers[new_idx] = \
                            self.histogram_builder.bin_mappers[orig_idx]
                
                # TURBO tree training
                tree = SuperFastLightGBMTree(
                    max_depth=self.max_depth,
                    min_child_samples=self.min_child_samples,
                    reg_lambda=self.reg_lambda,
                    histogram_builder=tree_histogram_builder,
                    max_leaves=self.num_leaves
                )
                
                tree.fit_ultra_fast(X_binned_subset, class_gradients, class_hessians)
                
                # TURBO prediction update
                tree_predictions = tree.predict_ultra_fast(X_binned[:, feature_indices])
                self._F_cache[:, class_idx] += self.learning_rate * tree_predictions
                
                iteration_trees.append(tree)
                iteration_features.append(feature_indices)
            
            self.trees.append(iteration_trees)
            self.feature_indices.append(iteration_features)
            
            # Fast progress reporting
            if (iteration + 1) % 20 == 0 or iteration < 10:
                elapsed = time.time() - train_start
                iter_per_sec = (iteration + 1) / elapsed
                print(f"🔥 Iter {iteration+1:3d}/{self.n_estimators}, Speed: {iter_per_sec:.1f} it/s")
        
        total_time = time.time() - start_time
        print(f"TURBO training completed in {total_time:.2f}s!")
        print(f"TURBO speed: {self.n_estimators/total_time:.1f} iterations/second")
        
        return self
    
    def predict_proba(self, X):
        """TURBO probability prediction"""
        X = np.ascontiguousarray(X, dtype=np.float64)
        X_binned = self.histogram_builder.map_to_bins_parallel(X)
        n_samples = len(X)
        
        F = np.tile(self.initial_predictions, (n_samples, 1))
        
        # TURBO tree predictions
        for iteration_trees, iteration_features in zip(self.trees, self.feature_indices):
            for class_idx, (tree, feature_indices) in enumerate(zip(iteration_trees, iteration_features)):
                tree_pred = tree.predict_ultra_fast(X_binned[:, feature_indices])
                F[:, class_idx] += self.learning_rate * tree_pred
        
        return parallel_softmax(F)
    
    def predict(self, X):
        """TURBO label prediction"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                               f1_score, classification_report, log_loss)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    logloss = log_loss(y_test, y_pred_proba)
    
    print(f"\n{model_name} EVALUATION RESULTS:")
    print("=" * 60)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Log Loss:    {logloss:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'log_loss': logloss
    }

    
    
    