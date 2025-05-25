# Scalable Graph Retrieval
# Hyperbolic Transformer
----fill here and push files---

# HashGAN: Adversarial Hashing with Hyperbolic Embeddings for Fast Semantic Retrieval

This repository contains the implementation of **HashGAN**, a self-supervised adversarial hashing model that learns robust and semantically meaningful hash codes from hyperbolic embeddings. Combined with a two-stage top-k retrieval strategy, it enables highly efficient and accurate semantic search on large-scale datasets.

##  Overview

HashGAN integrates:
- Hyperbolic projection and hashing via tangent/log maps.
- An adversarial generator-discriminator setup to improve robustness.
- Riemannian K-Means clustering for pseudo-labeling.
- Joint optimization to align learned hashes with semantic structure.

At inference, binary hash codes enable fast pre-filtering using Hamming distance, followed by precise re-ranking using hyperbolic (Lorentzian) distance.

##  Architecture

- **Generator:** Projects hyperbolic embeddings to tangent space, then outputs both continuous and binary hashes.
- **Discriminator:** Learns to distinguish between real and noisy hash codes.
- **Cluster Loss:** Encourages semantic consistency via pseudo-labels derived from hyperbolic K-means.



