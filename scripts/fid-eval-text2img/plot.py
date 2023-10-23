"""
python plot_fid_vs_clip.py \
  --fid_scores_csv path/to/fid_scores.csv \
  --clip_scores_csv path/to/clip_scores.csv
Replace path/to/fid_scores.csv and path/to/clip_scores.csv with the paths
to the respective CSV files. The script will display the plot with FID
scores against CLIP scores, with cfg values annotated on each point.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_fid_vs_clip(fid_scores_csv, clip_scores_csv):
    fid_scores = pd.read_csv(fid_scores_csv)
    clip_scores = pd.read_csv(clip_scores_csv)
    merged_data = pd.merge(fid_scores, clip_scores, on='cfg')

    fig, ax = plt.subplots()
    ax.plot(merged_data['clip_score'], merged_data['fid'], marker='o', linestyle='-')  # Connect points with a line

    for i, txt in enumerate(merged_data['cfg']):
        ax.annotate(txt, (merged_data['clip_score'][i], merged_data['fid'][i]))

    ax.set_xlabel('CLIP Score')
    ax.set_ylabel('FID')
    ax.set_title('FID vs CLIP Score')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fid_scores_csv', required=True, type=str, help='Path to the FID scores CSV file')
    parser.add_argument('--clip_scores_csv', required=True, type=str, help='Path to the CLIP scores CSV file')
    args = parser.parse_args()

    plot_fid_vs_clip(args.fid_scores_csv, args.clip_scores_csv)
