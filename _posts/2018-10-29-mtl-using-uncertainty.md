---
section: literature
category: Multi-Task Learning
title: "Multi-Task Learning Using Uncertainty to Weight Losses (CVPR'18)"
type: "MTL"
layout: nil
---

[Demo video][demo] \| [arXiv:1705.07115][arxiv] \| [PDF link][link]

This paper proposes an **heuristical** solution to loss weighing in Multi-Task Learning. Formally, it finds the good values of $$\lambda_i$$ coefficients in the Multi-Task Loss below:

$$
\begin{equation} \label{mtloss}
\begin{split}
\mathcal{L}_{\text{total}}\left(\cdot\right) = \sum_{i=1}^K {\lambda{}_i \mathcal{L}_i(\cdot) \rightarrow \text{min}}
\end{split}
\end{equation}
$$

where $$\mathcal{L}_i(\cdot)$$ is the loss for $$i$$-th task. Usually, setting $$\lambda_i$$ to $$1$$ is not a good idea: for different tasks, the magnitude of loss functions, as well as the magnitudes of gradients, might be very different.


## Multi-Task Likelihoods

Let $$f^W(x)$$ be the output of the network with weights $$W$$ on input $$x$$. For single-task, we model the network output uncertainty with a density function $$p\left( y \vert f^W(x) \right)$$ (how the true answer is likely to be $$y$$, given network's response). In the case of multiple network outputs $$y_1, \dots y_K$$, we obtain the following multi-task likelihood:

$$
\begin{equation} \label{mtlike}
p \left( y_1, \dots y_K \vert f^W(x) \right) = p\left(y_1 \vert f^W(x)\right) \dots p\left(y_K \vert f^W(x)\right)
\end{equation}
$$

In practice, we maximise the $$\text{log}( \cdot )$$ of the likelihood of the model.

### 1. Regression case

We model the likelihood as a Gaussian with mean given by the network output and observable scalar deviation $$\sigma$$:

$$
\begin{equation} \label{regcase}
p\left(y \vert f^W(x) \right) = \mathcal{N}\left( f^W(x), \sigma^2 \right)
\end{equation}
$$

the log likelihood can be written as:

$$
\begin{equation} \label{regcase_ll}
\text{log} p\left(y \vert f^W(x) \right) \propto -\frac{1}{2\sigma^2} \left\| y - f^W(x) \right\|^2 - \text{log}\,\sigma = -\frac{1}{2\sigma^2} \mathcal{L}(W) - \text{log}\,\sigma
\end{equation}
$$

In the Multi-Task case, it is obvious that applying $$(\ref{regcase_ll})$$ to $$(\ref{mtlike})$$ leads to a minimization objective $$\mathcal{L}(W, \sigma_1, \dots \sigma_K)$$ (our loss) for our Multi-Task Network:

$$
\begin{equation} \label{regcase_loss}
\mathcal{L}\left(W, \sigma_1, \dots \sigma_K \right) = -\text{log}\,p\left(y_1, \dots y_K \vert f^W(x)\right) \propto \sum_{i=1}^K{ \frac{1}{2\sigma_i^2} \mathcal{L}_i(W)} + \text{log}\, \prod_{i=1}^K{\sigma_i}
\end{equation}
$$

so the we are learning the objectives adaptively based on coefficients $$\sigma_i$$ (that are updated based on data). As $$\sigma$$ -- the noise parameter of $$y$$ increases, we have that the weights of $$\mathcal{L}(W)$$ decreases, and vise-versa. The noise is discouraged from increasing too much (effectively ignoring the data) by the last term objective, that acts like a regularizer.

### 2. Classification case

For classification, we squash the *scaled* network outputs through a softmax function, and sample from resulting probability vector:

$$
\begin{equation} \label{clcase}
p\left(y \vert f^W(x) \right) = \text{Softmax}\left( \frac{1}{\sigma^2} f^W(x) \right)
\end{equation}
$$

with a positive scalar $$\sigma$$, which can be interpreted as Boltzmann distribution (or Gibbs distribition). This scalar can be learnt, where the magnitude determines how 'uniform' (flat) the discrete distribution is. This relates to its uncertainty, as measured in entropy. The log likelihood then can be written as:

$$
\begin{equation} \label{clcase_ll}
\text{log}\, p\left(y = c\vert f^W(x),\,\sigma \right) = \frac{1}{\sigma^2} f_c^W(x) - \text{log} \sum_{c'} \exp{\left( \frac{1}{\sigma^2} f_{c'}^W(x) \right)}
\end{equation}
$$

where $$f_{c}^W$$ is the $$c$$'th element of the vector $$f^W(x)$$.




[demo]: https://www.youtube.com/watch?v=1OaIB-h-0Ws
[arxiv]: https://arxiv.org/abs/1705.07115
[link]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf

