# FedHSA
The paper proposes a heterogeneous federated learning framework named FedHSA, which aims to align discriminative features and improve the generalization performance of heterogeneous client models. FedHSA introduces two key components, both of which are innovative contributions.

Strengths:
1. We have done a lot of experiments on multiple datasets, compared the performance of our FedHSA and other popular methods.
2. we have enhanced our understanding of the main ideas of our paper through visualization.
3. we perform ablation experiments on each module in the fedhsa framework for verification.

Weaknesses:
1. In this paper, we use euclidean distance instead of cosine similarity to cluster prototypes in the part of subclass division.
2. Baselines is not introduced in detail in this article.
3. The threshold value is set by the percentile of L2 distance distribution between prototypes, and it is not clear that each category is divided into several subcategories.

Future Work:

In future work, we will consider exploring incentive mechanisms to encourage high-quality clients to participate in heterogeneous federated learning.
