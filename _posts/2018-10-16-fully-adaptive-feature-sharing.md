---
section: literature
category: Multi-Task Learning
title: "Fully-Adaptive Feature Sharing in MTNs (CVPR'17)"
type: "MTL"
layout: nil
---

[Youtube Talk][talk] \| [arXiv:1611.05377][arxiv] \| [PDF link][link]

This paper gives an answer to the question "when to split a layer into branches" while training a Multi-Tasking Neural Network. It uses a naive approach, based on so called "affinity score" between branches.

<p align="center">
  <img src="{{site.baseurl}}/static/images/task_splitting.svg" alt="" style="width:70%"/>
</p>

The downside of this naive approach is that it based off the data solely, and does not account the values of the neurons weights or activations or inner representations.


## Task Affinity Score

Let $$p_i^n$$ and $$g_i^n$$ -- network prediction and ground truth of the $$n$$-th sample for $$i$$-th task, respectively. The ***error margin*** of the sample is denoted as $$m_i^n = \| p_i^n - g_i^n \|$$.

Let's also define an ***indicator variable*** $$e_i^n$$ that takes the value $$1$$ if $$m_i^n \ge \mathbb{E}\{m_i\}$$, otherwise $$0$$.

Then, for a pair of tasks $$i$$ and $$j$$, their mutual ***affinity*** is defined as:

$$
\begin{equation} \label{affinity}
\begin{split}
    A(i, j) & = \mathbb{P} \left( e_i^n = 1, e_j^n = 1 \right) + \mathbb{P} \left( e_i^n = 1, e_j^n = 1 \right) \\
            & = \mathbb{E} \left\{ e_i^n e_j^n + (1 - e_i^n)(1 - e_j^n) \right\}
\end{split}
\end{equation}
$$

So, the *affinity* between task $$i$$ and $$j$$ can the thought of some measure of similarty between these tasks. The higher the *affinity*, the more similar they are.


## Branch Affinity Score

For network branches $$k$$ and $$l$$, with outputs to the tasks $$\{ T_k^0, T_k^1, \dots T_k^n \}$$ and $$\{ T_l^0, T_l^1, \dots T_l^m \}$$ respectively, the ***directed branch affinity*** from branch $$k$$ to branch $$l$$ and vice-versa are defined as following:

$$
\begin{equation} \label{dbaffinity}
\begin{split}
    \widetilde{A}_b (k, l) =  \underset{i_k}{\text{mean}} \left( \min_{j_l}{A\left(i_k, j_l\right)}\right) \\
    \widetilde{A}_b (l, k) =  \underset{j_l}{\text{mean}} \left( \min_{i_k}{A\left(j_l, i_k\right)}\right)
\end{split}
\end{equation}
$$

i.e. for every output $$j_k$$ from branch $$k$$, we consider its minimal affinity to the outputs from branch $$l$$ (so consider the maximum unlikelyhood), then the *directed branch affinity* from $$k$$ to $$l$$ is the mean of such values over outputs $$j_k$$.

Then, we can define the ***branch affinity*** between branches $$k$$ and $$l$$ as following:

$$
\begin{equation} \label{baffinity}
    A_b\left( k, l \right) = \frac{ \widetilde{A}_b(k,l) + \widetilde{A}_b(l, k) }{2}
\end{equation}
$$

The higher the *branch affinity* between branches, the more "similar" they are w.r.t. their tasks.


## Best branching between groups of tasks?

The idea is to penaltize the situation, where very similar tasks (with high affinity) are in the same branch. This part will be described later.





[link]: https://arxiv.org/pdf/1611.05377.pdf
[arxiv]: https://arxiv.org/abs/1611.05377
[talk]: https://www.youtube.com/watch?v=P5xPB9a55HA
