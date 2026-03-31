# Elementwise Operations

## Overview

This module implements basic elementwise operations on GPU.

---

## Implementations

### 1. Naive Version

- One thread processes one element
- Simple mapping

---

### 2. Grid-stride Version

- Each thread processes multiple elements
- Scales to arbitrary input size

---

## Key Concepts

### Elementwise Parallelism

- Each thread works independently
- No communication needed

---

### Grid-stride Loop

```cpp
for (int i = idx; i < N; i += blockDim.x * gridDim.x)