# Third-party attribution

The retriever architecture, directional distance encoding, frozen-encoder feature construction, and shortest-path weak supervision are adapted from:

**SubgraphRAG**, Graph-COM, copyright 2024 Graph-COM, MIT License.

- Repository: https://github.com/Graph-COM/SubgraphRAG
- Reviewed source revision: `48eb05e56d11293fa780849512310462c45b956b`
- Mufei Li, Siqi Miao, and Pan Li. *Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation*. ICLR 2025. https://arxiv.org/abs/2410.20724
- License: [licenses/SubgraphRAG.txt](licenses/SubgraphRAG.txt)

This implementation uses native PyTorch mean aggregation instead of PyTorch Geometric, retains parallel-edge supervision, and adds portable devices, per-example preparation, resumable training, graph storage, HTTP serving, and TRACE integration. It does not bundle original trained weights or claim compatibility with arbitrary upstream pickle/checkpoint files.

The path-search procedure and parsing prompt follow *TRACE: An Interactive Visual Paradigm for Knowledge Graph Question-Answering*, Vayd Sai Ramkumar, Srilekha Mamidala, Esha Ranade, Ilaria Liccardi, and Lalana Kagal, ISWC 2026 submission, Section 3 and Appendices A, B, D. The earlier MEng thesis's community visualization and prospective SPIRAL integration are not part of this backend.

WebQSP data is obtained separately from `ml1996/webqsp`, at revision `54a938897b1ef1e8dbd2fddb6985fd1e688172e8`, as in the released SubgraphRAG embedding workflow. Its provenance and any upstream terms remain applicable; the repository's code license does not relicense datasets or external model weights. The GTE model and its external custom-code revision are pinned in `configs/webqsp.yaml`. Transformers/PyTorch and other dependencies retain their respective licenses.
