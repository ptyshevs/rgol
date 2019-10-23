cimport cython
cimport numpy as np
import numpy as np
from libc.limits cimport INT_MAX  # Default max_depth
from libc.math cimport log2
from cython.parallel import prange, parallel

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef double entropy(double[:, :] y):
    cdef float n = y.shape[0]
    
    _,  cnts = np.unique(y, return_counts=True)
    if len(cnts) == 1:
        return 0
    cdef double accum = 0;
    cdef int i = 0;
    cdef double p;
    for i in range(len(cnts)):
        p = cnts[i] / n
        accum += p * log2(p)
    return -accum

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef double gini(double[:, :] y):
    cdef float n = y.shape[0]
    
    _,  cnts = np.unique(y, return_counts=True)
    cdef double accum = 0;
    cdef int i = 0;
    cdef double p;
    for i in range(len(cnts)):
        p = cnts[i] / n
        accum += p * p
    return 1 - accum

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef double variance(double[:, :] y):
    return np.var(y)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef double mad_median(double[:, :] y):
    return np.mean(np.abs(y - np.median(y)))

def mode(a):
    if len(a) == 0:
        return np.nan
    (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    mode = a[index]
    return mode

class Node():
    
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None,
                 right=None, index_mask=None, agg=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.left_labels = None
        self.right = right
        self.right_labels = None
        self.agg = agg
        self.index_mask = index_mask
        self.left_mask = None
        self.right_mask = None
        self.n_samples = 0
        if self.labels is not None:
            self.n_samples = len(self.labels)
            
        self.impurity = -1
        self.predict_left = None
        self.predict_right = None
        
    @property
    def is_leaf(self):
        return self.left is None or self.right is None
    
    def __repr__(self):
        r = f'[{self.feature_idx}]'
        r += f"<={self.threshold}"
        r += f'|n={self.n_samples}'
        r += f'|imp={self.impurity}'
        return r

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef float entropy_lightning(np.int64_t[:] counts, int n):
    """
    Parallel cross-entropy calculation, using counts per class as well as
    total number of samples across all classes.
    
    This is the bottleneck.
    """
    cdef float accum = 0.0, p, c
    cdef int i = 0;
    for i in prange(counts.shape[0], nogil=True):
        c = counts[i]
        if c > 0:
            p = c / n
            accum -= p * log2(p)
    return accum


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef void clf_update_counts(np.int64_t[:] counts, np.uint8_t[:] y, int start_pos, int end_pos):
    """
    Parallel update to per-class sample counts, for classification
    
    This is the bottleneck.
    """
    cdef int i = start_pos
    for i in prange(start_pos, end_pos, nogil=True):
        counts[y[i]] += 1
        
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef void reg_update_counts(np.float64_t[:] counts, np.float64_t[:] y, int start_pos, int end_pos):
    """
    Update to statistics across splits, for regression.
    
    Can be parallelized using Chan et al. stable algorithm.
    
    This is the bottleneck.
    """
    # counts[0] = count_left
    # counts[1] = mean_left
    # counts[2] = variance_left * n
    # counts[3] = count_right
    # count[4] = mean_right
    # count[5] = variance_right * n
    cdef int i = start_pos
    cdef double delta = 0, delta2
    cdef double cnt_left = counts[0]
    cdef double mean_left = counts[1]
    cdef double var_left = counts[2]
    cdef double cnt_right = counts[3]
    cdef double mean_right = counts[4]
    cdef double var_right = counts[5]

    for i in range(start_pos, end_pos):
        cnt_left += 1
        delta = y[i] - mean_left
        mean_left += delta / cnt_left
        delta2 = y[i] - mean_left
        var_left += delta * delta2
        
        cnt_right -= 1
        delta = y[i] - mean_right
        mean_right -= delta / cnt_right
        delta2 = y[i] - mean_right
        var_right -= delta * delta2
        
    counts[0] = cnt_left
    counts[1] = mean_left
    counts[2] = var_left
    counts[3] = cnt_right
    counts[4] = mean_right
    counts[5] = var_right

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef fast_unique(np.float64_t[:] x):
    # Assuming x is sorted in increasing order, unique becomes simpler
    # but is still asimptotically linear
    thresholds = [x[0]]
    switches = [0]
    cdef double cur_v = x[0]
    cdef int i = 0
    for i in range(x.shape[0]):
        if cur_v != x[i]:
            cur_v = x[i]
            thresholds.append(cur_v)
            switches.append(i)
    return np.array(thresholds, dtype=np.float64), np.array(switches, dtype=np.int64)

cdef class FastDecisionTree:
    cdef int max_depth, min_samples_split, min_samples_leaf, random_state
    cdef public int debug, is_clf
    cdef public criterion, mode, d_mapper, p_mapper, d_func, p_func, tree, labels
    cdef int tree_max_depth, best_split_attempts, n_unique_labels
    
    def __init__(self, int max_depth=INT_MAX, int min_samples_split=2, int min_samples_leaf=1,
                 criterion='entropy', debug=False, random_state=0, agg=None):
        """
        Classification\Regression Decision Tree
        
        @note: Regression is 4-times slower than classification
        @note2: this class is 5-30x slower than sklearn
        @param criterion: if 'entropy', classification is performed, otherwise variance ('mse' in sklearn) is used for regression
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.p_mapper = {'mode': mode, 'mean': np.mean, 'median': np.median}
        self.debug = debug
        self.tree = None
        self.random_state = random_state
        self.is_clf = criterion == "entropy"
        if agg is None:
            if self.is_clf:
                agg = 'mode'
            else:
                agg = 'mean'
        self.p_func = self.p_mapper[agg]

        self.tree_max_depth = 0
        self.best_split_attempts = 0
        self.n_unique_labels = 0
        
    def predict(self, X):
        y = np.full(len(X), np.nan, dtype=self.labels.dtype)

        mask = np.full(len(X), True)
        cur_depth = 0
        queue = [(self.tree, mask)]
        while True:
            if len(queue) == 0:
                break
            node, mask = queue.pop(0)
            left_mask = (X[:, node.feature_idx] <= node.threshold) & mask
            right_mask = (X[:, node.feature_idx] > node.threshold) & mask
            if not node.left:
                y[left_mask] = node.left_predict
            if not node.right:
                y[right_mask] = node.right_predict
            if node.left:
                queue.append((node.left, left_mask))
            if node.right:
                queue.append((node.right, right_mask))
        return y
        
    def predict_proba(self, X):
        y = np.full((len(X), self.n_unique_labels), np.nan, dtype=self.labels.dtype)

        mask = np.full(len(X), True)
        cur_depth = 0
        queue = [(self.tree, mask)]
        while True:
            if not queue:
                break
            node, mask = queue.pop(0)
            left_mask = (X[:, node.feature_idx] <= node.threshold) & mask
            right_mask = (X[:, node.feature_idx] > node.threshold) & mask
            if not node.left:
                cnts = np.bincount(node.left_labels, minlength=self.n_unique_labels)
                
                y[left_mask, :] = cnts / cnts.sum()
            if not node.right:
                cnts = np.bincount(node.right_labels, minlength=self.n_unique_labels)
                y[right_mask] = cnts / cnts.sum()
            if node.left:
                queue.append((node.left, left_mask))
            if node.right:
                queue.append((node.right, right_mask))
        return y
            
    def fit(self, X, y):
        np.random.seed(self.random_state)
        if self.is_clf:
            y = y.astype(np.uint8)
            self.n_unique_labels = len(np.unique(y))
        self.labels = y

        self.tree_max_depth = 0
        self.best_split_attempts = 0

        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        y = y.squeeze()
        
        mask = np.full(len(X), True)
        split_queue = [(mask, None, 0, -1, None)]
        
        while True:
            if not split_queue:
                break
            split_mask, split_root, depth, impurity, total_counts = split_queue.pop(0)
            if depth >= self.max_depth:
                break
            
            Xs, ys = X[split_mask], y[split_mask]

            if self.is_clf:
                ret = self._best_ft_clf(Xs, ys, impurity, total_counts)
            else:
                ret = self._best_ft_reg(Xs, ys, impurity, total_counts)

                
            best_idx, best_t, left_imp, right_imp, left_gc, right_gc = ret
            if self.debug:
                print("Cur_node:", str(split_root), "Best split:", best_idx, "t=", best_t, "left_imp=", left_imp, "right_imp=", right_imp)
            self.best_split_attempts += 1
            if best_idx is None or best_idx < 0:
                continue
            self.tree_max_depth = max(depth, self.tree_max_depth)
            
            node = Node(feature_idx=best_idx, threshold=best_t)
            left_mask = (X[:, best_idx] <= best_t) & split_mask
            node.left_predict = self.p_func(y[left_mask])
            node.left_labels = y[left_mask]
            right_mask = (X[:, best_idx] > best_t) & split_mask
            node.right_predict = self.p_func(y[right_mask])
            node.right_labels = y[right_mask]

            
            if not np.isclose(left_imp, 0):
                split_queue.append((left_mask, node, depth + 1, left_imp, left_gc))
            elif self.debug:
                print("Skipping left node (pure leaf according to impurity)")
            if not np.isclose(right_imp, 0):
                split_queue.append((right_mask, node, depth + 1, right_imp, right_gc))
            elif self.debug:
                print("Skipping right node (pure leaf according to impurity)")
            
            if split_root is None:
                if self.is_clf:
                    node.entropy = entropy_lightning(np.bincount(ys), len(ys))
                else:
                    node.entropy = np.var(ys)
                self.tree = node
                continue
            is_left = True
            if split_root.feature_idx == node.feature_idx:
                if split_root.threshold < node.threshold:
                    if split_root.right:
                        raise ValueError("WTF, This should not happen")
                    is_left = False
                elif split_root.threshold > node.threshold:
                    if split_root.left:
                        raise ValueError("WTF, This should not happen 2")
                    is_left = True
                else:
                    continue
            elif split_root.left is not None:
                is_left = False
            if is_left:
                node.impurity = left_imp
                split_root.left = node
            else:
                node.impurity = right_imp
                split_root.right = node


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def _best_ft_clf(self, X, y, float total_impurity, total_counts):
        cdef float d_l, d_r, d, best_d, t, best_t, left_impurity, right_impurity
        cdef int best_i, best_idx, feature_idx, n_l, n_r, n_labels, n_samples, n_features, n_switches, min_samples_leaf
        cdef int i = 0
        
        cdef double best_ft_t
        cdef float best_ft_d, ft_lt_imp, ft_rt_imp, best_lt_imp, best_rt_imp
        

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_labels = self.n_unique_labels
        min_samples_leaf = self.min_samples_leaf

        best_idx, best_d, best_t = -1, 0, 0
        cdef float n = float(n_samples)
        
        if n_samples < self.min_samples_leaf:
            if self.debug:
                print("Not enough samples to make a split")
            return best_idx, best_t, -1, 1
        
        
        if total_impurity < 0:
            total_counts = np.bincount(y, minlength=n_labels)
            total_impurity = entropy_lightning(total_counts, n_samples)
        
        best_lt_gc, best_rt_gc = None, None
        
        for feature_idx in range(n_features):
            x = X[:, feature_idx]
            idx = np.argsort(x)
            xs = x[idx]
            ys = y[idx]
            thresholds, switches = np.unique(xs, return_index=True)
            n_switches = switches.shape[0]
            if n_switches < 2:
                continue
            
            gc = np.zeros(n_labels, dtype=np.int64)
            
            best_ft_t, best_ft_d, ft_lt_imp, ft_rt_imp = 0, -1, -1, -1
            ft_lt_gc, ft_rt_gc = None, None
            d_l, d_r = -1, -1

            for i in range(n_switches - 1):
                clf_update_counts(gc, ys, switches[i], switches[i + 1])
                n_l = switches[i + 1]

                d_l = entropy_lightning(gc, n_l)
                
                if n_l < min_samples_leaf:
                    continue
                
                n_r = n_samples - n_l
                
                if n_r < min_samples_leaf:
                    continue
                if d_r == -1 or d_r > 0:
                    d_r = entropy_lightning(total_counts - gc, n_samples - n_l)
                
                d = total_impurity - n_l / n * d_l - n_r / n * d_r
                if d > best_ft_d:
                    best_ft_d = d
                    best_ft_t = thresholds[i]
                    ft_lt_imp = d_l
                    ft_rt_imp = d_r
                    ft_lt_gc = np.copy(gc)
                    ft_rt_gc = total_counts - gc
            if best_ft_d > best_d:
                best_d = best_ft_d
                best_idx = feature_idx
                best_t = best_ft_t
                best_lt_imp = ft_lt_imp
                best_lt_gc = ft_lt_gc
                best_rt_imp = ft_rt_imp
                best_rt_gc = ft_rt_gc
        return best_idx, best_t, best_lt_imp, best_rt_imp, best_lt_gc, best_rt_gc
    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def _best_ft_reg(self, X, y, float total_impurity, total_counts):
        cdef float d_l, d_r, d, best_d, t, best_t, left_impurity, right_impurity
        cdef int best_i, best_idx, feature_idx, n_l, n_r, n_labels, n_samples, n_features, n_switches, min_samples_leaf
        cdef int i = 0
        
        cdef double best_ft_t
        cdef float best_ft_d, ft_lt_imp, ft_rt_imp, best_lt_imp, best_rt_imp
        

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_labels = self.n_unique_labels
        min_samples_leaf = self.min_samples_leaf

        best_idx, best_d, best_t = -1, 0, 0
        cdef float n = float(n_samples)
        
        if n_samples <= min_samples_leaf:
            if self.debug:
                print("Not enough samples to make a split")
            return best_idx, best_t, -1, 1, -1, -1
        
        
        if total_impurity < 0:
            total_impurity = np.var(y)
        
        total_counts = np.array([len(y), y.mean(), np.var(y) * len(y)])
        best_lt_gc, best_rt_gc = None, None

        
        for feature_idx in range(n_features):
            x = X[:, feature_idx]
            idx = np.argsort(x)
            xs = x[idx]
            ys = y[idx]
            thresholds, switches = [-1, 0, 1], [xs]
            n_switches = switches.shape[0]
            if n_switches < 2:
                continue
            
            gc = np.zeros(6)
            gc[3:] = total_counts
            
            best_ft_t, best_ft_d, ft_lt_imp, ft_rt_imp = 0, -1, -1, -1
            ft_lt_gc, ft_rt_gc = None, None
            d_l, d_r = -1, -1
            for i in range(n_switches - 1):
                reg_update_counts(gc, ys, switches[i], switches[i + 1])
                n_l = switches[i + 1]

                d_l = gc[2] / gc[0]
                
                if n_l < min_samples_leaf:
                    continue
                
                n_r = n_samples - n_l
                
                if n_r < min_samples_leaf:
                    continue
                if d_r == -1 or d_r > 0:
                    d_r = gc[5] / gc[3]
                else:
                    break
                
                d = total_impurity - n_l / n * d_l - n_r / n * d_r
                if d > best_ft_d:
                    best_ft_d = d
                    best_ft_t = thresholds[i]
                    ft_lt_imp = d_l
                    ft_rt_imp = d_r
            if best_ft_d > best_d:
                best_d = best_ft_d
                best_idx = feature_idx
                best_t = best_ft_t
                best_lt_imp = ft_lt_imp
                best_lt_gc = ft_lt_gc
                best_rt_imp = ft_rt_imp
                best_rt_gc = ft_rt_gc
        return best_idx, best_t, best_lt_imp, best_rt_imp, best_lt_gc, best_rt_gc
