# Supplementary Material: Dataset Construction and Implementation Details

This document provides the dataset construction procedures and implementation
details that are referenced in the main paper but omitted from the manuscript
for length reasons.

---

## 1. Datasets

We conduct two sets of experiments to evaluate the effectiveness of the proposed
method. First, we use a controlled **synthetic dataset** to enable a clear
comparison with existing approaches under well-defined distribution shifts.
Second, we evaluate the method on multiple real-world **benchmark datasets**
constructed to capture various distribution shifts. Here, we provide details on
how to construct the datasets used in the experiments.

### 1.1 Synthetic Dataset

In this study, we extend the synthetic dataset construction procedure proposed by
Garg et al. (2023) to additionally introduce class-prior shift.

Let $\mathbf{x}$ denote an instance consisting of two feature subsets,
$\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2]$, where $\mathbf{x}_1 \in \mathbb{R}^{d_1}$
and $\mathbf{x}_2 \in \mathbb{R}^{d_2}$. The first component $\mathbf{x}_1$
represents features that remain consistent between the training and test data,
while the second component $\mathbf{x}_2$ corresponds to features whose
distributions differ between the two. In the training data, $\mathbf{x}_2$ is
generated to be correlated with the label $y$, whereas in the test data, it is
replaced with Gaussian noise so that the feature distributions differ between the
training and test data. To further simulate PU learning under class-prior shift,
we sample positive and unlabeled instances with different class-prior proportions
across the training and test distributions.

Formally, we denote the positive training instance, unlabeled training instance,
and unlabeled test instance as

$$
\begin{aligned}
\mathbf{x}_{\text{tr}}^+ &= [\mathbf{x}_{1(\text{tr})}^+,\ \mathbf{x}_{2(\text{tr})}^+], \\
\mathbf{x}_{\text{tr}}   &= [\mathbf{x}_{1(\text{tr})},\ \mathbf{x}_{2(\text{tr})}], \\
\mathbf{x}_{\text{te}}   &= [\mathbf{x}_{1(\text{te})},\ \mathbf{x}_{2(\text{te})}],
\end{aligned}
$$

where $\mathbf{x}_{\text{tr}}^+$, $\mathbf{x}_{\text{tr}}$, and
$\mathbf{x}_{\text{te}}$ represent the positive training, unlabeled training, and
unlabeled test instances, respectively. $\mathbf{x}_{1(\cdot)}$ represents the
feature subset shared between the training and test data, and
$\mathbf{x}_{2(\cdot)}$ represents the feature subset whose distribution differs
between them.

#### Shared feature subset $\mathbf{x}_1$

The feature subset $\mathbf{x}_1$, which appears in both training and test
distributions, is generated as

$$
\begin{aligned}
\mathbf{x}_{1(\text{tr})}^+ &\sim P_1^+, \\
\mathbf{x}_{1(\text{tr})}   &\sim \pi_{\text{tr}}\, P_1^+ + (1-\pi_{\text{tr}})\, P_1^-, \\
\mathbf{x}_{1(\text{te})}   &\sim \pi_{\text{te}}\, P_1^+ + (1-\pi_{\text{te}})\, P_1^-,
\end{aligned}
$$

where $P_1^y$ denotes a Gaussian distribution
$\mathcal{N}(\gamma \cdot y w,\ \Sigma_1)$ with covariance
$\Sigma_1 \coloneqq \sigma_1^2(\mathbf{I}_{d_1} - w w^\top)$, and
$w \sim \text{Unif}(\mathbb{S}^{d_1-1})$.

This construction ensures that $\mathbf{x}_1$ encodes label-related information
shared across both training and test data. The covariance $\Sigma_1$ removes
variance along the label direction, retaining only information relevant to $y$.
Although all instances share the same class-conditional Gaussian distributions
$P_1^+$ and $P_1^-$, the overall marginal distributions differ due to the PU
learning setup and the class-prior shift, which is reflected by different
class-prior ratios $\pi_{\text{tr}}$ and $\pi_{\text{te}}$. Note that if the class
priors are identical ($\pi_{\text{tr}} = \pi_{\text{te}}$), the marginal
distributions also become identical, and no distribution shift occurs in
$\mathbf{x}_1$.

#### Distribution-specific feature subset $\mathbf{x}_2$

For the feature subset $\mathbf{x}_2$, which contains information specific to the
training distribution, we do not perform random sampling for
$\mathbf{x}_{2(\text{tr})}^+$ and $\mathbf{x}_{2(\text{tr})}$ but instead assign
them fixed values. In contrast, $\mathbf{x}_{2(\text{te})}$ is generated using
Gaussian noise drawn from a distribution that does not carry any information about
the training distribution. Formally,

$$
\begin{aligned}
\mathbf{x}_{2(\text{tr})}^+ &\coloneqq +\mathbf{1}_{d_2}, \\
\mathbf{x}_{2(\text{tr})}   &\coloneqq \pi_{\text{tr}}\cdot(+1)\mathbf{1}_{d_2} + (1-\pi_{\text{tr}})\cdot(-1)\mathbf{1}_{d_2}, \\
\mathbf{x}_{2(\text{te})}   &\sim \mathcal{N}(\mathbf{0},\ \Sigma_2),
\end{aligned}
$$

where $\Sigma_2 \coloneqq \sigma_2^2 \mathbf{I}_{d_2}$. Since $\mathbf{x}_2$ also
reflects the PU learning setup and the class-prior shift, we distinguish
$\mathbf{x}_{2(\text{tr})}^+$, $\mathbf{x}_{2(\text{tr})}$, and
$\mathbf{x}_{2(\text{te})}$ accordingly.

To systematically examine the impact of class-prior shift, we vary the positive
class ratio $\pi$ and consider three combinations of training and test priors:

$$
(\pi_{\text{tr}},\ \pi_{\text{te}}) \in \{(0.5, 0.5),\ (0.3, 0.7),\ (0.7, 0.3)\}.
$$

### 1.2 Benchmark Dataset

Recently, a benchmark dataset for PU learning under distribution shift has been
proposed by Kumagai et al. (2025). In their study, the benchmark datasets are
constructed based on input/output shift (Gama et al., 2014). For example, in the
CIFAR10 dataset, classes (0, 1, 8, 9) are assigned as positive and classes
(2, 3, 4, 5, 6, 7) as negative during training, while in testing, classes 0 and 1
are changed to negative and classes 2 and 3 to positive. This setting is
challenging because the semantic meaning of the positive label changes across
splits, where the positive label at test time corresponds to a different subset of
base classes than at training time. As a result, a classifier trained on the
training semantics optimizes for a different target task at test time, and
adapting the model to the new label semantics is infeasible without access to
target samples. To address this issue, Kumagai et al. (2025) partially utilize
test data to compensate for the distribution difference between training and
testing. In contrast, our study assumes that access to test data is not available,
and therefore, the benchmark datasets constructed in Kumagai et al. (2025) cannot
be directly used in our experiments.

As a result, we follow the setup proposed in another line of prior work
(Garg et al., 2022) to construct our benchmark datasets. Specifically, we employ
four datasets in our experiments: CIFAR10 (Krizhevsky & Hinton, 2009), ENTITY13
and LIVING17 from the BREEDS benchmark (Santurkar et al., 2021), and Camelyon17
from the WILDS benchmark (Koh et al., 2021). For CIFAR10, we adopt CIFAR10v2
(Lu et al., 2020), CIFAR10-C (Hendrycks & Dietterich, 2019), and CINIC10
(Darlow et al., 2018) as test datasets that share the same label set but differ in
their data distributions, thereby introducing distribution shift across the
datasets. To further account for class-prior shift, we adjust the class ratios in
each dataset so that they differ between the training and test sets. ENTITY13,
LIVING17, and Camelyon17 are originally designed to exhibit distribution shift
between their training and test sets. To further account for class-prior shift, we
introduce an additional shift in these datasets in the same manner as in CIFAR10.
All data preprocessing procedures follow the setups adopted in prior studies for
each dataset.

Note that CIFAR10 (including CIFAR10v2, CIFAR10-C, and CINIC10), ENTITY13, and
LIVING17 originally consist of 10, 13, and 17 classes, respectively. Therefore,
they are not directly applicable to PU learning, which assumes a binary
classification setting. CIFAR10 datasets are adopted directly from previous work
(Kumagai et al., 2025), while others are constructed with reference to the same
methodology using our proposed procedure:

- **CIFAR10:** Vehicle classes (0, 1, 8, 9) are labeled as positive, and animal
  classes (2, 3, 4, 5, 6, 7) as negative.
- **ENTITY13:** Natural objects (1, 2, 3, 4, 12) are labeled as positive, and
  artificial objects (0, 5, 6, 7, 8, 9, 10, 11) as negative.
- **LIVING17:** Mammals (8, 9, 10, 11, 12, 15, 16) are labeled as positive, and
  non-mammals (0, 1, 2, 3, 4, 5, 6, 7, 13, 14) as negative.

---

## 2. Implementation Details

For CIFAR10 datasets, the encoder consists of three convolutional blocks followed
by an adaptive average pooling and flattening operation that yields a
256-dimensional embedding. Each block applies two consecutive $3 \times 3$
convolutions with Batch Normalization and ReLU activation, while the spatial
resolution and channel dimensions evolve as

$$
32 \times 32 \times 3
\ \rightarrow\ 32 \times 32 \times 64
\ \rightarrow\ 16 \times 16 \times 128
\ \rightarrow\ 8 \times 8.
$$

The classifier is implemented as a two-layer MLP with 64 hidden units and ReLU
activations. For ENTITY13, LIVING17, and Camelyon17, we use the convolutional
backbone of ResNet-18 as the encoder, and a two-layer MLP with 512 hidden units
and ReLU activations as the classifier.

In the proposed method, data augmentation is applied in both the contrastive
learning and self-training stages. During contrastive learning, two augmented
views of each image are generated using RandomResizedCrop, RandomHorizontalFlip,
ColorJitter, GaussianBlur, and Solarization, with varying transformation
strengths. In self-training, following FixMatch (Sohn et al., 2020), weak
augmentations (RandomHorizontalFlip, normalization) and strong augmentations
(geometric and color transformations such as rotation, flip, brightness, and
sharpness) are used to ensure pseudo-label consistency.

---

## References

Darlow, L. N., Crowley, E. J., Antoniou, A., & Storkey, A. J. (2018). CINIC-10 is not ImageNet or CIFAR-10. *arXiv preprint arXiv:1810.03505*.

Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1–37.

Garg, S., Balakrishnan, S., Lipton, Z. C., Neyshabur, B., & Sedghi, H. (2022). Leveraging Unlabeled Data to Predict Out-of-Distribution Performance. In *ICLR*.

Garg, S., Setlur, A., Lipton, Z. C., Balakrishnan, S., Smith, V., & Raghunathan, A. (2023). Complementary Benefits of Contrastive Learning and Self-Training Under Distribution Shift. In *NeurIPS*.

Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. In *ICLR*.

Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., Hu, W., Yasunaga, M., Phillips, R. L., Gao, I., Lee, T., David, E., Stavness, I., Guo, W., Earnshaw, B. A., Haque, I. S., Beery, S., Leskovec, J., Kundaje, A., Pierson, E., Levine, S., Finn, C., & Liang, P. (2021). WILDS: A Benchmark of in-the-Wild Distribution Shifts. In *ICML*.

Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.

Kumagai, A., Iwata, T., Takahashi, H., Nishiyama, T., & Fujiwara, Y. (2025). Importance-weighted Positive-unlabeled Learning for Distribution Shift Adaptation. In *AISTATS*, PMLR 258:1576–1584.

Lu, S., Nott, B., Olson, A., Todeschini, A., Vahabi, H., Carmon, Y., & Schmidt, L. (2020). Harder or Different? A Closer Look at Distribution Shift in Dataset Reproduction. In *ICML Workshop on Uncertainty and Robustness in Deep Learning*.

Santurkar, S., Tsipras, D., & Madry, A. (2021). BREEDS: Benchmarks for Subpopulation Shift. In *ICLR*.

Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H., & Raffel, C. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. In *NeurIPS*.
