

#!/usr/bin/env python3
"""
plot curves (llm generated)
"""
import os
import sys
from typing import List, Tuple
import re
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_series(path: str) -> Tuple[List[float], List[float]]:
	"""Read a newline-separated series from `path`.

	Each non-empty line may contain either a single float (exploitability)
	or two floats separated by comma/whitespace: "exploitability, time_to_train".

	Returns a tuple (exploitabilities, times). `times` will be empty if no
	per-line times are found.
	"""
	values: List[float] = []
	times: List[float] = []
	if not os.path.exists(path):
		print(f"Warning: file not found: {path}")
		return values, times

	with open(path, 'r') as f:
		for i, line in enumerate(f):
			s = line.strip()
			if not s:
				continue

			# Split on comma or whitespace
			parts = [p for p in re.split('[,\s]+', s) if p]
			if not parts:
				continue

			# Try parsing first token as exploitability
			try:
				v = float(parts[0])
				values.append(v)
			except ValueError:
				print(f"Warning: could not parse exploitability on line {i+1} in {path}: '{line.strip()}'", file=sys.stderr)
				continue

			# If there is a second token, try parsing it as time-to-train
			if len(parts) >= 2:
				try:
					t = float(parts[1])
					times.append(t)
				except ValueError:
					# ignore malformed time values but keep exploitability
					print(f"Warning: could not parse time on line {i+1} in {path}: '{line.strip()}'", file=sys.stderr)

	return values, times


def plot_learning_curve(arank_series: Tuple[List[float], List[float]],
						ssd_series: Tuple[List[float], List[float]],
						out_path: str = 'learning_curve.png'):
	arank_exp, arank_time = arank_series
	ssd_exp, ssd_time = ssd_series

	x_ar = list(range(1, len(arank_exp) + 1))
	x_ssd = list(range(1, len(ssd_exp) + 1))

	# Avoid non-positive values for log scale by clipping to a tiny positive value
	eps_clip = 1e-12
	arank_plot = [max(v, eps_clip) for v in arank_exp]
	ssd_plot = [max(v, eps_clip) for v in ssd_exp]

	plt.figure(figsize=(9, 5))
	ax1 = plt.gca()
	if arank_plot:
		ax1.plot(x_ar, arank_plot, label='Alpha-Rank', marker='o', linewidth=1)
	if ssd_plot:
		ax1.plot(x_ssd, ssd_plot, label='SSD', marker='s', linewidth=1)

	ax1.set_xlabel('iteration')
	ax1.set_ylabel('exploitability')
	ax1.set_title('a-rank vs ssd, 3-player leduc poker')
	ax1.grid(True, linestyle='--', alpha=0.5, which='both')

	# Secondary axis for time-to-train if either series contains times
	has_time = bool(arank_time) or bool(ssd_time)
	if has_time:
		ax2 = ax1.twinx()
		# Plot times as simple point markers (one point per iteration)
		if arank_time:
			x_ar_time = list(range(1, len(arank_time) + 1))
			ax2.plot(x_ar_time, arank_time, marker='o', linestyle='None',
					 label='Alpha-Rank time', color='C2', alpha=0.9)
		if ssd_time:
			x_ssd_time = list(range(1, len(ssd_time) + 1))
			ax2.plot(x_ssd_time, ssd_time, marker='s', linestyle='None',
					 label='SSD time', color='C3', alpha=0.9)
		ax2.set_ylabel('time_to_train (s)')

		# Combine legends from both axes
		lines_labels = [ax.get_legend_handles_labels() for ax in (ax1, ax2)]
		lines = [l for (l, _) in lines_labels for l in l]
		labels = [lbl for (_, lbl) in lines_labels for lbl in lbl]
		ax1.legend(lines, labels)
	else:
		ax1.legend()

	# Optionally annotate final exploitability values
	if arank_plot:
		ax1.annotate(f"{arank_plot[-1]:.4e}", xy=(x_ar[-1], arank_plot[-1]), xytext=(5, 0), textcoords='offset points')
	if ssd_plot:
		ax1.annotate(f"{ssd_plot[-1]:.4e}", xy=(x_ssd[-1], ssd_plot[-1]), xytext=(5, 0), textcoords='offset points')

	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	print(f"Saved learning-curve plot to {out_path}")


def main():
	parser = argparse.ArgumentParser(description='Plot learning curves from experiment files')
	parser.add_argument('--start', type=int, default=1, help='Start iteration (1-indexed) to plot from, default 5')
	parser.add_argument('--arank', type=str, default=None, help='Path to exp_arank.txt')
	parser.add_argument('--ssd', type=str, default=None, help='Path to exp_ssd.txt')
	parser.add_argument('--out', type=str, default=None, help='Output PNG path')
	args = parser.parse_args()

	repo_root = os.path.dirname(os.path.abspath(__file__))
	arank_path = args.arank if args.arank is not None else os.path.join(repo_root, '..', 'exp_arank.txt')
	ssd_path = args.ssd if args.ssd is not None else os.path.join(repo_root, '..', 'exp_ssd.txt')

	# Support running from repository root too
	if not os.path.exists(arank_path):
		arank_path = os.path.join(repo_root, 'exp_arank.txt')
	if not os.path.exists(ssd_path):
		ssd_path = os.path.join(repo_root, 'exp_ssd.txt')

	arank_exp, arank_time = read_series(arank_path)
	ssd_exp, ssd_time = read_series(ssd_path)

	if not arank_exp and not ssd_exp:
		print('No data found in exp_arank.txt or exp_ssd.txt. Exiting.')
		return

	# Apply start slicing (1-indexed)
	start_idx = max(1, args.start) - 1
	arank_exp_slice = arank_exp[start_idx:]
	arank_time_slice = arank_time[start_idx:] if arank_time else []
	ssd_exp_slice = ssd_exp[start_idx:]
	ssd_time_slice = ssd_time[start_idx:] if ssd_time else []

	out_png = args.out if args.out is not None else os.path.join(repo_root, 'learning_curve.png')
	plot_learning_curve((arank_exp_slice, arank_time_slice), (ssd_exp_slice, ssd_time_slice), out_png)


if __name__ == '__main__':
	main()



# time for 13 iteration ssd new: Time so far: 7010.094840049744
