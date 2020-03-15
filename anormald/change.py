"""
SingularSpectrumTransformation
"""
from typing import *
import numpy as np


class SSTChangeDetector(object):
	def __init__(self, n_windows: int, n_columns: Optional[int]=None, n_lags: Optional[int]=None, r: int=1, m: int=1):
		"""
		"""
		self._n_windows = n_windows
		self._n_columns = n_columns if n_columns is not None else self._n_windows // 2
		self._n_lags = n_lags if n_lags is not None else self._n_columns // 2
		self._r = r
		self._m = m

	def _rolling_window(self, ts: np.ndarray):
		shape = (len(ts) - self._n_windows + 1, self._n_windows)
		return np.lib.stride_tricks.as_strided(ts, shape=shape)

	def _trajetory_matrix(self, ts, t):
		start = t - self._n_windows - self._n_columns + 1
		end = t
		if start < 0 or end > len(ts):
			return None
		return self._rolling_window(ts)[start:end]

	def _test_matrix(self, ts, t):
		start = t - self._n_windows - self._n_columns + 1 + self._n_lags
		end = t + self._n_lags
		if start < 0 or end > len(ts):
			return None
		return self._rolling_window(ts)[start:end]

	def _svd(self, X, full_metrices=False, compute_uv=False):
		U, _, _ = np.linalg.svd(X, full_metrices=full_metrices)
		return U

	def fit(self, ts: np.ndarray):
		"""
		"""
		self._scores = np.zeros(len(ts))
		for t in range(len(ts)):
			# 履歴行列
			X = self._trajetory_matrix(ts, t)
			# テスト行列
			Z = self._test_matrix(ts, t)
			score = 0
			if X and Z:
				# 特異値分解(SVD)
				U, _, _ = np.linalg.svd(X, full_metrices=False)
				Q, _, _ = np.linalg.svd(X, full_metrices=False)
				U_r = U[:, :self._r]
				Q_m = Q[:, :self._m]
				score = 1 - np.linalg.svd(np.dot(U_r.T, Q_m), full_metrices=False, compute_uv=False)[0]
			self._scores[t] = score

	def scores(self, standarize=False):
		if standarize:
			return self._scores / sum(self._scores)
		return self._scores
