# FedHSA
The paper proposes a heterogeneous federated learning framework named FedHSA, which aims to align discriminative features and improve the generalization performance of heterogeneous client models. FedHSA introduces two key components, both of which are innovative contributions.

Strengths:
1. We propose a novel heterogeneous federated learning framework named FedHSA. We design a differentiated prototype aggregation (DPA) method on the server side to obtain fine-grained and semantically consistent global prototypes. It can enhance the semantic representational capacity of the shared knowledge.
2. We propose a hierarchical prototype alignment method (HPA) on the client side. It can achieve fine-grained alignment between the local representation of the client and the global prototype of the subclass. In addition, it can also capture global semantic information and improve the generalization of heterogeneous client models by constraining the representation of the client through superclass.
3. We conduct extensive comparative and experiments across three datasets and eight heterogeneous model. The experimental results demonstrate that our proposed FedHSA method consistently outperforms existing HtFL methods. In particular, compared to FedProto, FedHSA achieves performance improvements of up to 9.65%.

Weaknesses:
1. In subection B of the related work, we discuss approaches to address model heterogeneity using knowledge distillation. However, the subsection title only mentions “knowledge distillation.” It is recommended that we revise the title to more accurately reflect the content and scope of the discussion.
2. Baselines is not introduced in detail in this paper.
3. In Section V, regarding prototype clustering, we set the value of Num to 100. The rationale behind this choice is unclear. We should provide a detailed explanation or justification for selecting this specific value.

Future Work:

Later updates will make up for the above three weaknesses. In future work, we will consider exploring incentive mechanisms to encourage high-quality clients to participate in heterogeneous federated learning.
