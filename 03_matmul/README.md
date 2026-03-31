# Matrix Multiplication

## Overview

This module implements matrix multiplication on GPU with progressively optimized kernels, focusing on how computation is organized across thread, warp, and block levels.

---

## Implementations

### 1. Naive Version

- Each thread computes one output element
- No data reuse
- Memory-bound due to excessive global memory access

---

### 2. Tiled (Shared Memory) Version

- Uses shared memory to cache tiles of A and B
- Reduces global memory traffic
- Introduces data reuse within a block

---

### 3. Register Blocking Version

- Each thread computes multiple output elements (e.g., 2×2)
- Uses registers to store intermediate results
- Increases compute intensity
- Reduces redundant memory access

---

### 4. Warp-level Version

- Organizes computation at the warp level
- Each warp is responsible for a sub-tile
- Introduces hierarchical decomposition: block → warp → thread

---

## Key Concepts

### Data Reuse

- Shared memory enables reuse across threads
- Registers enable reuse within a thread

---

### Tiling

- Divides computation into tiles to improve locality
- Reduces pressure on global memory bandwidth

---

### Register Blocking

- Each thread computes multiple outputs
- Improves arithmetic intensity

---

### Warp-level Execution

- Warp is the basic execution unit on GPU
- Threads in a warp execute in lockstep
- Enables coordinated computation without explicit synchronization

---

### Memory Hierarchy

- Global memory → large but slow
- Shared memory → faster, block-level
- Registers → fastest, thread-level

---

## Summary

- Performance improvements mainly come from increasing data reuse
- Optimization progresses from thread-level to warp-level coordination
- Register blocking and tiling are key techniques for efficient matrix multiplication