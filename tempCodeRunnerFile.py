from collections import defaultdict
import random
from typing import Sequence, Tuple, Any


def stratified_split(X: Sequence[Any], y: Sequence[Any], test_size: float = 0.2, random_state: int = None) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]]:
	"""Simple stratified train/test split.

	Parameters:
	- X: sequence of samples
	- y: sequence of labels (same length as X)
	- test_size: fraction of samples to include in test set
	- random_state: optional seed

	Returns: X_train, X_test, y_train, y_test
	"""
	if random_state is not None:
		random.seed(random_state)

	if len(X) != len(y):
		raise ValueError("X and y must have the same length")

	# group indices by class
	groups = defaultdict(list)
	for idx, label in enumerate(y):
		groups[label].append(idx)

	test_idx = set()
	for label, indices in groups.items():
		n_test = max(1, int(len(indices) * test_size)) if len(indices) > 0 else 0
		selected = random.sample(indices, n_test)
		test_idx.update(selected)

	X_train, X_test, y_train, y_test = [], [], [], []
	for idx, (xi, yi) in enumerate(zip(X, y)):
		if idx in test_idx:
			X_test.append(xi)
			y_test.append(yi)
		else:
			X_train.append(xi)
			y_train.append(yi)

	return X_train, X_test, y_train, y_test


if __name__ == "__main__":
	# tiny sanity check
	X = list(range(20))
	y = [0]*10 + [1]*10
	Xt, Xe, yt, ye = stratified_split(X, y, test_size=0.3, random_state=0)
	print(len(Xt), len(Xe), sum(1 for v in ye if v==0), sum(1 for v in ye if v==1))