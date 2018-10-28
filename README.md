# unzip-nets
A greedy approach for finding the optimal architecture (in terms of speed-accuracy tradeoff) for Multi-Task Learning. Although such kind of delicate graph manipulations are easier on [PyTorch][pytorch], we chose to use [TensorFlow][tf] for its matureness on mobile platforms ([TF-Lite][lite]).

## Repository structure
This repo is divided into 3 main branches:

*  `master` - for the main Unzip-Nets Multi-Task Learning code
*  `prototyping` - for random Jupyter notebooks, small experiments, and prototyping snippets
*  `gh-pages` - for the documentations and [literature summaries][litsum] (with LaTeX)


[litsum]: https://hav4ik.github.io/unzip-nets/literature
[pytorch]: https://pytorch.org/
[tf]: https://www.tensorflow.org/
[lite]: https://www.tensorflow.org/lite/
