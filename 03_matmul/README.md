# Matrix Multiplication

## Overview

This module implements matrix multiplication with progressive optimization.

---

## Implementations

### 1. Naive Version

- Direct computation
- No data reuse
- Memory-bound

---

### 2. Tiled Version

- Uses shared memory
- Improves data reuse
- Reduces global memory access

---

## Key Concepts

### Data Reuse

- Each element loaded once used multiple times

---

### Tiling

- Divide computation into blocks
- Load tiles into shared memory

---

### Memory Hierarchy

- Global memory → slow
- Shared memory → faster

---

## Insights

- Naive version is bandwidth limited
- Tiling significantly improves performance
- Memory access pattern is critical