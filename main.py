import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import math
import heapq
from scipy import sparse

class HistogramBuilder:
    
    def __init__(self, max_bins=255):
        self.max_bins = max_bins
        self.bin_mappers = {}  
        
    def build_feature_histograms(self, X):
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) <= self.max_bins:
                bin_edges = np.concatenate([unique_values, [unique_values[-1] + 1e-10]])
            else:
                # Tạo equal-density histogram
                bin_edges = np.histogram_bin_edges(feature_values, bins=self.max_bins)
            
            self.bin_mappers[feature_idx] = bin_edges
    
    def map_to_bins(self, X):
        n_samples, n_features = X.shape
        binned_X = np.zeros((n_samples, n_features), dtype=np.uint8)
        
        for feature_idx in range(n_features):
            bin_edges = self.bin_mappers[feature_idx]
            binned_X[:, feature_idx] = np.digitize(X[:, feature_idx], bin_edges) - 1
            binned_X[:, feature_idx] = np.clip(binned_X[:, feature_idx], 0, len(bin_edges) - 2)
        
        return binned_X

class GradientHistogram:
    
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.gradient_sum = np.zeros(n_bins)
        self.hessian_sum = np.zeros(n_bins)
        self.sample_count = np.zeros(n_bins)
    
    def add_sample(self, bin_idx, gradient, hessian):

        self.gradient_sum[bin_idx] += gradient
        self.hessian_sum[bin_idx] += hessian
        self.sample_count[bin_idx] += 1
    
    def calculate_gain(self, reg_lambda=0.1):
        gains = []
        
        # Cumulative sums for left side
        cumsum_grad = np.cumsum(self.gradient_sum)
        cumsum_hess = np.cumsum(self.hessian_sum)
        cumsum_count = np.cumsum(self.sample_count)
        
        total_grad = cumsum_grad[-1]
        total_hess = cumsum_hess[-1]
        total_count = cumsum_count[-1]
        
        for split_idx in range(self.n_bins - 1):
            left_grad = cumsum_grad[split_idx]
            left_hess = cumsum_hess[split_idx]
            left_count = cumsum_count[split_idx]
            
            right_grad = total_grad - left_grad
            right_hess = total_hess - left_hess
            right_count = total_count - left_count
            
            if left_count < 1 or right_count < 1:
                gains.append(-np.inf)
                continue
            
            # LightGBM gain formula
            left_gain = (left_grad ** 2) / (left_hess + reg_lambda)
            right_gain = (right_grad ** 2) / (right_hess + reg_lambda)
            parent_gain = (total_grad ** 2) / (total_hess + reg_lambda)
            
            gain = 0.5 * (left_gain + right_gain - parent_gain)
            gains.append(gain)
        
        return np.array(gains)

class LightGBMTreeNode:
    
    def __init__(self, depth=0, samples_idx=None):
        # Split information
        self.feature_idx = None
        self.bin_threshold = None  # Threshold ở dạng bin index
        self.raw_threshold = None  # Threshold ở dạng raw value
        
        # Tree structure
        self.left = None
        self.right = None
        self.parent = None
        
        # Node information
        self.is_leaf = True
        self.depth = depth
        self.samples_idx = samples_idx if samples_idx is not None else []
        self.value = 0.0  # Leaf value
        
        # Statistics for gain calculation
        self.gradient_sum = 0.0
        self.hessian_sum = 0.0
        self.sample_count = 0
        
        # Priority for leaf-wise growth
        self.split_gain = 0.0

class LightGBMTree:
    
    def __init__(self, max_depth=6, min_child_samples=20, reg_lambda=0.1, 
                 histogram_builder=None, max_leaves=31):
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.histogram_builder = histogram_builder
        self.max_leaves = max_leaves
        
        self.root = None
        self.leaf_nodes = []  # Danh sách các leaf có thể split
        
    def fit(self, X_binned, gradients, hessians):
        n_samples = len(gradients)
        
        # Initialize root
        self.root = LightGBMTreeNode(depth=0, samples_idx=np.arange(n_samples))
        self._update_node_stats(self.root, gradients, hessians)
        
        # Priority queue cho leaf-wise growth
        self.leaf_nodes = [self.root]
        
        num_leaves = 1
        while num_leaves < self.max_leaves and self.leaf_nodes:
            # Tìm leaf có highest gain để split
            best_leaf, best_split = self._find_best_leaf_to_split(X_binned, gradients, hessians)
            
            if best_leaf is None or best_split['gain'] <= 0:
                break
            
            # Split best leaf
            left_child, right_child = self._split_leaf(best_leaf, best_split, 
                                                     X_binned, gradients, hessians)
            
            # Remove split leaf và add new leaves
            self.leaf_nodes.remove(best_leaf)
            if left_child.depth < self.max_depth:
                self.leaf_nodes.append(left_child)
            if right_child.depth < self.max_depth:
                self.leaf_nodes.append(right_child)
            
            num_leaves += 1
        
        # Calculate leaf values
        self._calculate_leaf_values()
    
    def _update_node_stats(self, node, gradients, hessians):
        if len(node.samples_idx) > 0:
            node.gradient_sum = np.sum(gradients[node.samples_idx])
            node.hessian_sum = np.sum(hessians[node.samples_idx])
            node.sample_count = len(node.samples_idx)
    
    def _find_best_leaf_to_split(self, X_binned, gradients, hessians):
        best_leaf = None
        best_split = {'gain': -np.inf}
        
        for leaf in self.leaf_nodes:
            if (leaf.sample_count < self.min_child_samples * 2 or 
                leaf.depth >= self.max_depth):
                continue
            
            split_info = self._find_best_split_for_leaf(leaf, X_binned, gradients, hessians)
            if split_info['gain'] > best_split['gain']:
                best_split = split_info
                best_leaf = leaf
        
        return best_leaf, best_split
    
    def _find_best_split_for_leaf(self, leaf, X_binned, gradients, hessians):
        best_split = {'gain': -np.inf}
        
        n_features = X_binned.shape[1]
        samples_idx = leaf.samples_idx
        
        for feature_idx in range(n_features):
            # Build histogram cho feature này
            feature_bins = X_binned[samples_idx, feature_idx]
            max_bin = int(np.max(feature_bins)) + 1
            
            hist = GradientHistogram(max_bin)
            
            # Populate histogram
            for i, sample_idx in enumerate(samples_idx):
                bin_idx = feature_bins[i]
                hist.add_sample(bin_idx, gradients[sample_idx], hessians[sample_idx])
            
            # Calculate gains for all possible splits
            gains = hist.calculate_gain(self.reg_lambda)
            
            if len(gains) > 0:
                best_bin_idx = np.argmax(gains)
                max_gain = gains[best_bin_idx]
                
                if max_gain > best_split['gain']:
                    # Convert bin threshold to raw value
                    bin_edges = self.histogram_builder.bin_mappers[feature_idx]
                    raw_threshold = bin_edges[best_bin_idx + 1]
                    
                    best_split = {
                        'gain': max_gain,
                        'feature_idx': feature_idx,
                        'bin_threshold': best_bin_idx,
                        'raw_threshold': raw_threshold
                    }
        
        return best_split
    
    def _split_leaf(self, leaf, split_info, X_binned, gradients, hessians):
        feature_idx = split_info['feature_idx']
        bin_threshold = split_info['bin_threshold']
        
        # Split samples
        samples_idx = leaf.samples_idx
        feature_bins = X_binned[samples_idx, feature_idx]
        
        left_mask = feature_bins <= bin_threshold
        right_mask = ~left_mask
        
        left_samples = samples_idx[left_mask]
        right_samples = samples_idx[right_mask]
        
        # Create children
        left_child = LightGBMTreeNode(depth=leaf.depth + 1, samples_idx=left_samples)
        right_child = LightGBMTreeNode(depth=leaf.depth + 1, samples_idx=right_samples)
        
        # Update statistics
        self._update_node_stats(left_child, gradients, hessians)
        self._update_node_stats(right_child, gradients, hessians)
        
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
    
    def _calculate_leaf_values(self):
        self._calculate_leaf_values_recursive(self.root)
    
    def _calculate_leaf_values_recursive(self, node):
        if node.is_leaf:
            if node.hessian_sum > 0:
                node.value = -node.gradient_sum / (node.hessian_sum + self.reg_lambda)
            else:
                node.value = 0.0
        else:
            self._calculate_leaf_values_recursive(node.left)
            self._calculate_leaf_values_recursive(node.right)
    
    def predict(self, X_binned):
        return np.array([self._predict_sample(x) for x in X_binned])
    
    def _predict_sample(self, x_binned):
        node = self.root
        while not node.is_leaf:
            if x_binned[node.feature_idx] <= node.bin_threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class LightGBMClassifier:
    
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
                 random_state=None):
        
        # Hyperparameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.max_bins = max_bins
        self.random_state = random_state
        
        # Model components
        self.histogram_builder = HistogramBuilder(max_bins=max_bins)
        self.trees = []
        self.feature_indices = []
        
        # Multiclass setup
        self.n_classes = None
        self.classes_ = None
        self.initial_predictions = None
        
        if random_state:
            np.random.seed(random_state)
    
    def _softmax(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(np.clip(x_shifted, -500, 500))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y):
        """Convert labels to one-hot encoding"""
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot
    
    def _compute_gradients_hessians(self, y_true_onehot, F):
        """Compute gradients and hessians for multiclass"""
        probabilities = self._softmax(F)
        gradients = probabilities - y_true_onehot
        hessians = probabilities * (1 - probabilities)
        return gradients, hessians
    
    def _sample_features(self, n_features):
        """Sample features for each tree"""
        if self.colsample_bytree >= 1.0:
            return np.arange(n_features)
        
        n_selected = max(1, int(n_features * self.colsample_bytree))
        return np.random.choice(n_features, n_selected, replace=False)
    
    def _sample_data(self, n_samples):
        """Sample data indices for each tree"""
        if self.subsample >= 1.0:
            return np.arange(n_samples)
        
        n_selected = max(1, int(n_samples * self.subsample))
        return np.random.choice(n_samples, n_selected, replace=False)
    
    def fit(self, X, y):
        """Train LightGBM model"""
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        
        print(" LIGHTGBM-LIKE TRAINING STARTED")
        
        # Setup multiclass
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        print(f"Dataset: {n_samples} samples, {n_features} features, {self.n_classes} classes")
        
        # Build histograms
        print("Building feature histograms...")
        self.histogram_builder.build_feature_histograms(X)
        X_binned = self.histogram_builder.map_to_bins(X)
        print(f"Mapped to {self.max_bins} bins per feature")
        
        # Initialize predictions
        class_counts = np.bincount(y, minlength=self.n_classes)
        class_probs = class_counts / n_samples
        self.initial_predictions = np.log(class_probs + 1e-15)
        self.initial_predictions -= self.initial_predictions[0]  # First class as reference
        
        # Initialize F matrix
        F = np.tile(self.initial_predictions, (n_samples, 1))
        y_onehot = self._one_hot_encode(y)
        
        print(f"Training {self.n_estimators} iterations with leaf-wise growth...")
        
        # Training loop
        for iteration in range(self.n_estimators):
            iteration_trees = []
            iteration_features = []
            
            # Compute gradients and hessians
            gradients, hessians = self._compute_gradients_hessians(y_onehot, F)
            
            # Train one tree per class
            for class_idx in range(self.n_classes):
                # Sample data
                sample_indices = self._sample_data(n_samples)
                
                # Sample features  
                feature_indices = self._sample_features(n_features)
                
                # Get gradients/hessians for this class
                class_gradients = gradients[sample_indices, class_idx]
                class_hessians = hessians[sample_indices, class_idx]
                
                # Subset data
                X_binned_subset = X_binned[np.ix_(sample_indices, feature_indices)]
                
                # Create tree with histogram builder for subset
                tree_histogram_builder = HistogramBuilder(max_bins=self.max_bins)
                
                # Map feature indices for subset
                for new_idx, orig_idx in enumerate(feature_indices):
                    tree_histogram_builder.bin_mappers[new_idx] = \
                        self.histogram_builder.bin_mappers[orig_idx]
                
                # Train tree
                tree = LightGBMTree(
                    max_depth=self.max_depth,
                    min_child_samples=self.min_child_samples,
                    reg_lambda=self.reg_lambda,
                    histogram_builder=tree_histogram_builder,
                    max_leaves=self.num_leaves
                )
                
                tree.fit(X_binned_subset, class_gradients, class_hessians)
                
                # Update predictions
                tree_predictions = tree.predict(X_binned[:, feature_indices])
                F[:, class_idx] += self.learning_rate * tree_predictions
                
                # Store tree
                iteration_trees.append(tree)
                iteration_features.append(feature_indices)
            
            self.trees.append(iteration_trees)
            self.feature_indices.append(iteration_features)
            
            # Progress report
            if (iteration + 1) % 10 == 0:
                loss = self._calculate_loss(y_onehot, F)
                print(f"📈 Iteration {iteration+1:3d}/{self.n_estimators}, Loss: {loss:.6f}")
        
        print("Training completed!")
        return self
    
    def _calculate_loss(self, y_true_onehot, F):
        """Calculate categorical cross-entropy loss"""
        probabilities = self._softmax(F)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true_onehot * np.log(probabilities), axis=1))
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X, dtype=np.float64)
        X_binned = self.histogram_builder.map_to_bins(X)
        n_samples = len(X)
        
        # Initialize with base predictions
        F = np.tile(self.initial_predictions, (n_samples, 1))
        
        # Add predictions from all trees
        for iteration_trees, iteration_features in zip(self.trees, self.feature_indices):
            for class_idx, (tree, feature_indices) in enumerate(zip(iteration_trees, iteration_features)):
                tree_pred = tree.predict(X_binned[:, feature_indices])
                F[:, class_idx] += self.learning_rate * tree_pred
        
        return self._softmax(F)
    
    def predict(self, X):
        """Predict class labels"""
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

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    import pandas as pd
    
    # Load data
    print("Loading dataset...")
    data = pd.read_csv('./Data/iniDataset.csv')
    
    # Prepare data
    data.Disease = data.Disease.astype('category')
    disease_mapping = dict(enumerate(data['Disease'].cat.categories))
    print(f"🏥 Disease classes: {disease_mapping}")
    
    data.Disease = data.Disease.cat.codes.values
    X = data.drop('Disease', axis=1).values
    y = data['Disease'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nDataset shape: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test")
    
    # Train Custom LightGBM
    print("TRAINING CUSTOM LIGHTGBM")

    custom_lgbm = LightGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        max_bins=255,
        random_state=42
    )
    
    custom_lgbm.fit(X_train_scaled, y_train)
    custom_results = evaluate_model(custom_lgbm, X_test_scaled, y_test, "Custom LightGBM")
    
    