# Reduction Optimization

## Overview

This module implements different versions of parallel reduction on GPU.

I progressively optimize from shared memory based reduction to warp-level primitives and atomic aggregation.

---

## Implementations

### 1. Shared Memory Reduction

- Tree-based reduction
- Uses `__syncthreads()` for synchronization
- High synchronization overhead

---

### 2. Warp-level Reduction

- Uses `__shfl_down_sync`
- Eliminates synchronization within warp
- Shared memory only used for inter-warp communication

---

### 3. Warp + Atomic Reduction

- Uses warp-level reduction
- Aggregates results using `atomicAdd`
- No shared memory required

---

## Key Concepts

### Grid-stride Loop

Allows threads to process multiple elements:

```cpp
for (int i = idx; i < n; i += blockDim.x * gridDim.x)