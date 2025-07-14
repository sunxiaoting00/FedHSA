# FedHSA
We propose a novel HtFL framework FedHSA, which is based on multi-prototype subclass hierarchical semantic alignment. Within FedHSA, we design two key components: a differentiated prototype aggregation (DPA) module and a hierarchical prototype alignment (HPA) module.

Strengths:
1. We propose a novel heterogeneous federated learning framework named FedHSA. We design a differentiated prototype aggregation (DPA) method on the server side to obtain fine-grained and semantically consistent global prototypes. It can enhance the semantic representational capacity of the shared knowledge.
2. We propose a hierarchical prototype alignment method (HPA) on the client side. It can achieve fine-grained alignment between the local representation of the client and the global prototype of the subclass. In addition, it can also capture global semantic information and improve the generalization of heterogeneous client models by constraining the representation of the client through superclass.
3. We conduct extensive comparative and experiments across three datasets and eight heterogeneous model. The experimental results demonstrate that our proposed FedHSA method consistently outperforms existing HtFL methods. In particular, compared to FedProto, FedHSA achieves performance improvements of up to 9.65%.

Weaknesses:
1. In subection B of the related work, we discuss approaches to address model heterogeneity using knowledge distillation. However, the subsection title only mentions “knowledge distillation.” It is recommended that we revise the title to more accurately reflect the content and scope of the discussion.
2. Baselines is not introduced in detail in this paper.
3. In Section V, regarding prototype clustering, we set the value of Num to 100. The rationale behind this choice is unclear. We should provide a detailed explanation or justification for selecting this specific value.

Future Work:

We will make up for the above three weaknesses in the later updated version. In future work, we will consider exploring incentive mechanisms to encourage high-quality clients to participate in heterogeneous federated learning.
