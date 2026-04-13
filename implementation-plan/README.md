# Implementation Plan

This folder contains the detailed technical implementation plan for the project described in [Ideas for project.md](../Ideas%20for%20project.md).

Read in this order:

1. [01-system-breakdown.md](01-system-breakdown.md)
2. [02-project-folder-structure.md](02-project-folder-structure.md)
3. [03-detailed-module-design.md](03-detailed-module-design.md)
4. [04-code-skeleton.md](04-code-skeleton.md)
5. [05-training-strategy.md](05-training-strategy.md)
6. [06-practical-tips-and-pitfalls.md](06-practical-tips-and-pitfalls.md)
7. [07-next-steps.md](07-next-steps.md)

The recommended starting baseline for a single-GPU environment is:

* `convnext_tiny` for image encoding
* `xlm-roberta-base` for text encoding
* concatenation-based fusion before attempting cross-attention
* image size `224 x 224`
* text `max_length=96` or `128`
* mixed precision training with checkpointing and early stopping