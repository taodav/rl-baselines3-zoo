from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', **{'family': 'serif', 'serif': ['cmr10'], 'size': 14})
rc('axes', unicode_minus=False)

if __name__ == "__main__":
    buffer_dir = Path("/Users/ruoyutao/Documents/rl-baselines3-zoo/buffers")

    alignments = {}
    for env_dir in buffer_dir.glob("*"):
        alignments_file = env_dir / "alignments.npy"
        env_name = '-'.join(env_dir.name.split("-")[:2])
        alignments[env_name] = np.load(alignments_file, allow_pickle=True).item()
    
    fig, axes = plt.subplots(2, int(np.ceil(len(alignments) / 2)), figsize=(16, 8))

    for i, (env_name, alignments) in enumerate(alignments.items()):
        ax = axes[i % 2, i // 2]
        x = np.arange(alignments['k'])
        ax.plot(x, alignments['return_alignments'])
        ax.set_title(env_name)
        ax.set_ylim(0, 1)

    fig.supylabel('Alignment')
    fig.supxlabel('k')
    fig.suptitle('Alignment of discounted return')

    fig.tight_layout()
    plt.show()


