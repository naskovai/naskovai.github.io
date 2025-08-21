+++
date = '2025-08-21T00:10:41-07:00'
draft = false
title = 'The Sandwich Framework for Understanding Linear Algebra'
description = 'Coordinate Translations, Scaling, and State Transitions - A unified approach to linear algebra decompositions'
tags = ['linear-algebra', 'mathematics', 'eigenvalues', 'svd', 'decompositions']
math = true
+++

## Core Philosophy: Translate → Transform → Translate Back

Every fundamental linear algebra operation follows the same pattern:
1. **Translate**: Change to coordinates where the problem becomes simple
2. **Transform**: Apply scaling in those coordinates  
3. **Translate Back**: Change back to the original coordinate system

## State Notation & Matrix Types

**System State**: What coordinate system the computation is currently operating in (e.g., Standard, Eigen, Singular)

**Operation Language**: What coordinate system a matrix/operation is natively written in

**Matrix Types** (the key insight!):
- 🔄 **Rotations/Reflections**: Orthogonal matrices - preserve geometry, just change viewpoint
- 📏 **Scaling**: Diagonal matrices - stretch/shrink along axes, the "actual work"
- 🔀 **General coordinate changes**: Non-orthogonal matrices - might distort geometry
- 🎯 **Projections**: Combine rotation + truncation

---

## 1) Orthogonal Diagonalization: $M = Q D Q^T$ (symmetric matrices)

**What this is about**: When you have a symmetric matrix (like a covariance matrix, Hessian, or quadratic form), this decomposition finds the natural "principal axes" where the matrix becomes diagonal. This is the cleanest possible decomposition because symmetric matrices have orthogonal eigenvectors, making all coordinate changes pure rotations.

**When to use**: Symmetric matrices, quadratic forms, principal component analysis, optimization (finding principal axes of curvature).

**Legend for this case:**

*Symbols:*
- $M$: **symmetric matrix written in Standard** 
- $Q$: **orthogonal matrix of eigenvectors written in Standard** (so $Q^T = Q^{-1}$)
- $D$: **diagonal matrix of eigenvalues written in Eigen** (diagonal in eigenbasis)

*Operations:*
1. $Q^T$: **Standard → Eigen** (rotation - preserves geometry)
2. $D$: **diagonal scaling in Eigen coords**
3. $Q$: **Eigen → Standard** (rotation - preserves geometry)

**Geometric story**: Rotate to natural axes, scale, rotate back.

**Operation Language**: M, Q written in Standard; D written in Eigen  
**System State Transitions**:
1. Start: Standard
2. $Q^T$: Standard → Eigen 🔄 rotation to eigen coords
3. $D$: acts in Eigen 📏 pure scaling in eigen coords  
4. $Q$: Eigen → Standard 🔄 rotation back
5. End: Standard

**Why it works**: Symmetric matrices have orthogonal eigenvectors, so the coordinate change is a pure rotation.

---

## 2) "Matrix in eigen language" (3Blue1Brown frame): $P^{-1} M P = D$

**What this is about**: This is Grant Sanderson's (3Blue1Brown) perspective on eigendecomposition. Instead of applying a matrix to vectors, we're asking: "What would this matrix look like if we changed our coordinate system to use eigenvectors as basis vectors?" It's about relabeling the matrix itself rather than transforming vectors.

**When to use**: When you want to understand what a linear transformation "really does" in its most natural coordinate system. Conceptual understanding of eigendecomposition.

**Legend for this case:**

*Symbols:*
- $M$: **diagonalizable matrix written in Standard**
- $P$: **matrix of eigenvectors written in Standard** (columns are eigenvectors, may not be orthogonal)
- $D$: **diagonal matrix of eigenvalues written in Eigen**

*Operations:*
1. $P$: **converts Eigen inputs → Standard** (so M can receive them)
2. $M$: **acts in its native Standard language** 
3. $P^{-1}$: **converts Standard outputs → Eigen** (so result is in Eigen)

**Geometric story**: Relabeling the operator itself, not applying to a vector.

**Operation Language**: M, P written in Standard  
**System Interpretation Changes**:
1. Start: System interprets operations in Standard
2. $P$: converts Eigen inputs → Standard 🔀 so M can receive them
3. $M$: acts in its native Standard language 🔀 processes in Standard
4. $P^{-1}$: converts Standard outputs → Eigen 🔀 result interpreted as Eigen
5. End: System sees M as acting Eigen → Eigen (becomes D)

**Key insight**: We're changing how we *interpret* the matrix's inputs/outputs, not transforming the system state.

---

## 3) General Diagonalization: $M = P D P^{-1}$ (non-symmetric)

**What this is about**: When you have a non-symmetric matrix that's still diagonalizable, the eigenvectors are no longer orthogonal. This means the coordinate changes can distort geometry (stretching, shearing) rather than just rotating. It's the "messy" version of eigendecomposition where we lose the clean geometric properties.

**When to use**: Non-symmetric matrices that are still diagonalizable, dynamic systems, Markov chains, some optimization problems.

**Legend for this case:**

*Symbols:*
- $M$: **non-symmetric diagonalizable matrix written in Standard**
- $P$: **matrix of eigenvectors written in Standard** (columns are eigenvectors, may not be orthogonal)
- $D$: **diagonal matrix of eigenvalues written in Eigen**

*Operations:*
1. $P^{-1}$: **Standard → Eigen** (may distort geometry)
2. $D$: **diagonal scaling in Eigen coords** 
3. $P$: **Eigen → Standard** (may distort geometry)

**Geometric story**: Change to skewed eigenbasis, scale, change back.

**Operation Language**: M, P written in Standard; D written in Eigen  
**System State Transitions**:
1. Start: Standard
2. $P^{-1}$: Standard → Eigen 🔀 general coordinate change (may distort)
3. $D$: acts in Eigen 📏 scaling in eigen coords
4. $P$: Eigen → Standard 🔀 change back
5. End: Standard

**Key difference**: $P^{-1}$ is typically NOT orthogonal, so this coordinate change can distort geometry.

---

## 4) QR Decomposition & Projections: $P_{\text{proj}} = Q Q^T$

**What this is about**: This section focuses on the beautiful case where you already have orthonormal columns (matrix Q). Whether you got Q from QR decomposition of some original matrix A, or you started with orthonormal columns, projecting onto that subspace becomes elegantly simple: just $QQ^T$. This is the "clean" projection case.

**When to use**: When you have orthonormal basis vectors for your subspace. After running QR decomposition and extracting Q. Projections in contexts where bases are already orthogonal.

**Legend for this case:**

*Symbols:*
- $Q$: **orthogonal matrix written in Standard** (columns are orthonormal basis for subspace, so $Q^T = Q^{-1}$)
- $P_{\text{proj}}$: **orthogonal projection operator written in Standard**

*Operations:*
1. $Q^T$: **Standard → Subspace** (rotation - preserves geometry)
2. $Q$: **Subspace → Standard** (rotation - preserves geometry)

**Geometric story**: Rotate to subspace coordinates, keep those components, rotate back.

**Operation Language**: Q written in Standard  
**System State Transitions**:
1. Start: Standard
2. $Q^T$: Standard → Subspace 🔄 rotation to subspace coords
3. (I): identity in Subspace 📏 identity = no scaling
4. $Q$: Subspace → Standard 🔄 rotation back  
5. End: Standard

**Insight**: Orthogonal projection uses pure rotations - that's why it's geometrically clean.

---

## 5) General Projection: $P = A(A^T A)^{-1}A^T$

**What this is about**: When you want to project onto a subspace but you only have non-orthogonal basis vectors (columns of A), you can't use the simple $AA^T$ formula. Instead, you need the pseudoinverse machinery to correct for the overlaps between non-orthogonal columns. This is the "messy but general" projection formula.

**When to use**: Projecting onto subspaces when your basis isn't orthogonal, least squares with non-orthogonal regressors, data fitting to non-orthogonal function spaces.

**Legend for this case:**

*Symbols:*
- $A$: **matrix with potentially non-orthogonal columns written in Standard** (columns are standard coordinate vectors)
- $A^T$: **transpose of A, written in Standard** (matrix entries stored normally)
- $(A^T A)^{-1}$: **inverse Gram matrix written in Standard** (matrix entries stored normally)
- $P$: **projection operator written in Standard** (result matrix stored normally)

*Operations:*
1. $A^T$: **Standard → A_coords** (extract coordinates w.r.t. A's columns)
2. $(A^T A)^{-1}$: **A_coords → A_coords** (correct for non-orthogonality in A_coords)
3. $A$: **A_coords → Standard** (reconstruct in Standard using A's columns)

**Geometric story**: Extract overlaps, correct for non-orthogonality, then reconstruct.

**Operation Language**: A, $A^T$, $(A^T A)^{-1}$ all written in Standard  
**System State Transitions**:
1. Start: Standard
2. $A^T$: Standard → A_coords 🔀 extract coordinates w.r.t. A's columns
3. $(A^T A)^{-1}$: A_coords → A_coords 🔀 correct for non-orthogonal columns  
4. $A$: A_coords → Standard 🔀 reconstruct in standard coords
5. End: Standard

**Why it's complex**: When $A$ has non-orthogonal columns, we can't just use $AA^T$ like in the orthogonal case. Instead:

**Why it's complex**: When $A$ has non-orthogonal columns, we can't just use $AA^T$ like in the orthogonal case. Instead:

1. **$A^T$**: Computes "raw overlaps" - how much the input vector overlaps with each column of $A$
2. **$(A^T A)^{-1}$**: The **Gram matrix correction** - fixes the fact that A's columns aren't orthogonal
3. **$A$**: Reconstructs the vector using only A's column space

**The key insight**: $(A^T A)^{-1}A^T$ together form the **pseudoinverse** of $A$, which is the generalization of inversion for non-square or non-orthogonal matrices.

---

## 6) SVD: The Crown Jewel $A = U \Sigma V^T$

**What this is about**: SVD is the most general matrix decomposition - it works for ANY matrix (even non-square!). It finds the optimal coordinate systems for both input and output spaces simultaneously, revealing the fundamental structure of any linear transformation as "rotate → scale → rotate". It's what eigendecomposition wishes it could be.

**When to use**: Principal component analysis, data compression, image processing, collaborative filtering, any time you need the "best" low-rank approximation to data.

**Key insight: Singular values vs. Eigenvalues**

Before diving in, let's clarify a crucial distinction:

- **Eigenvalues**: Only exist for square matrices, can be negative or complex, come from $Av = \lambda v$
- **Singular values**: Exist for ANY matrix (even non-square!), always non-negative real numbers, are the square roots of eigenvalues of $A^TA$ (or $AA^T$)

**Why singular values are more universal:**
- Every matrix has singular values, but not every matrix has eigenvalues
- Singular values tell you about the "stretching factors" of a matrix along its singular vector directions
- The singular value decomposition works even when eigendecomposition fails (non-square or defective matrices)

**The beautiful connection:**
- Singular values $\sigma_i$ are $\sqrt{\text{eigenvalues of } A^TA}$
- The singular vectors (columns of $U$ and $V$) are the directions where $A$ achieves pure scaling
- These are NOT the eigenvectors of $A$ itself (unless $A$ is symmetric), but rather the eigenvectors of $A^TA$ and $AA^T$

**Legend for this case:**

*Symbols:*
- $A$: **any matrix written in Standard** (columns are standard coordinate vectors)
- $U$: **orthogonal matrix written in Standard** (columns are standard coordinate vectors)
- $\Sigma$: **diagonal matrix bridging Right_singular → Left_singular** (doesn't live in any single coordinate system)
- $V$: **orthogonal matrix written in Standard** (columns are standard coordinate vectors)
- $\Sigma^+$: **pseudoinverse bridging Left_singular → Right_singular** (inverse of Σ)

*Operations:*
1. $V^T$: **Standard_input → Right_singular** (rotation - preserves geometry)
2. $\Sigma$: **Right_singular → Left_singular** (diagonal scaling between DIFFERENT singular coordinate spaces)
3. $U$: **Left_singular → Standard_output** (rotation - preserves geometry)

*For pseudoinverse:*
1. $U^T$: **Standard_output → Left_singular** (rotation - preserves geometry)
2. $\Sigma^+$: **Left_singular → Right_singular** (inverse scaling between different coordinate spaces)
3. $V$: **Right_singular → Standard_input** (rotation - preserves geometry)

**Geometric story**: Optimal rotation in input space, pure scaling between spaces, optimal rotation in output space.

**Critical insight**: Right_singular and Left_singular are **different coordinate spaces**:
- **Right_singular**: Optimal basis for the input space (eigenvectors of $A^TA$)
- **Left_singular**: Optimal basis for the output space (eigenvectors of $AA^T$)
- **SVD decomposition**: Reveals how any linear transformation connects these optimal coordinate systems

### 6a) Forward: $y = A x = U \Sigma V^T x$

**What this is about**: This is the basic SVD application - using the decomposition to apply the original matrix $A$ to a vector. It shows how any linear transformation can be broken down into three simple steps: rotate in input space, scale between spaces, rotate in output space.

**When to use**: Understanding what a matrix "really does" geometrically, implementing matrix multiplication efficiently when you already have the SVD.

**Operation Language**: A, U, V written in Standard; $\Sigma$ bridges coordinate systems  
**System State Transitions (Input Space → Output Space)**:
1. Start: Standard_input
2. $V^T$: Standard_input → Right_singular 🔄 rotation in input space
3. $\Sigma$: Right_singular → Left_singular 📏 scaling between different singular spaces
4. $U$: Left_singular → Standard_output 🔄 rotation to output space
5. End: Standard_output

### 6b) Pseudoinverse: $x^+ = A^+ y = V \Sigma^+ U^T y$

**What this is about**: The pseudoinverse gives you the "best possible inverse" for any matrix, even non-square ones. It's the universal solution to "given output $y$, what input $x$ most likely produced it?" When $A$ is invertible, this gives the exact inverse. When it's not, it gives the least-squares best approximation.

**When to use**: Solving linear systems that have no exact solution (overdetermined), data fitting, finding the "closest" solution to inconsistent systems, inverting non-square matrices.

**Operation Language**: U, V written in Standard; $\Sigma^+$ bridges coordinate systems  
**System State Transitions (Output Space → Input Space)**:
1. Start: Standard_output
2. $U^T$: Standard_output → Left_singular 🔄 rotation in output space
3. $\Sigma^+$: Left_singular → Right_singular 📏 inverse scaling between different spaces
4. $V$: Right_singular → Standard_input 🔄 rotation back to input space
5. End: Standard_input

### 6c) Projections via SVD

**What this is about**: SVD gives you clean projections onto both the column space and row space of any matrix. Since $U$ and $V$ are already orthonormal, you get the beautiful $UU^T$ and $VV^T$ projection formulas without needing any pseudoinverse corrections. This connects SVD to the fundamental subspaces of linear algebra.

**When to use**: Principal component analysis (projecting onto top singular directions), data compression, noise filtering, dimensionality reduction.

**Column space projection**: $P_{\text{col}} = U_r U_r^T$.
- System transitions: Standard_output → Left_singular → Standard_output (keep first $r$ components)

**Row space projection**: $P_{\text{row}} = V_r V_r^T$ (acts in input space).
- System transitions: Standard_input → Right_singular → Standard_input (keep first $r$ components)

### 6d) Whitening / preconditioning: $(A^T A)^{-1/2} = V \Sigma^{-1} V^T$

**What this is about**: Whitening transforms your coordinate system so that the matrix $A^T A$ becomes the identity matrix. This "undoes" any stretching or correlation structure, making the space isotropic (same in all directions). It's like taking a stretched, tilted ellipse and turning it back into a perfect circle.

**When to use**: Optimization (preconditioning gradient descent), machine learning (whitening data before training), signal processing (removing correlations), preparing data so that all dimensions are treated equally.

- $A^T A = V \Sigma^2 V^T$ (domain curvature)
- $(A^T A)^{-1/2} = V \Sigma^{-1} V^T$
  - Apply to $g$: $V^T g$ → $\Sigma^{-1}$ → $V$.
    Same **translate → simple scale → translate back** sandwich.

**Operation Language**: V written in Standard; $\Sigma^{-1}$ acts within Right_singular coordinate system  
**System State Transitions**:
1. Start: Standard
2. $V^T$: Standard → Right_singular 🔄 rotation to right singular coords
3. $\Sigma^{-1}$: Right_singular → Right_singular 📏 inverse scaling within right singular space
4. $V$: Right_singular → Standard 🔄 rotation back
5. End: Standard

**Geometric meaning**: Transform space so that $A^T A$ becomes identity (isotropic).

---

## 7) Cholesky Decomposition: $A = L L^T$ (positive definite)

**What this is about**: When you have a positive definite symmetric matrix (like a covariance matrix or a "nice" quadratic form), you can factor it as the product of a lower triangular matrix with itself. Think of L as the "square root" of A. This is computationally efficient and numerically stable.

**When to use**: Covariance matrices, solving systems with positive definite matrices, generating correlated random variables, optimization with quadratic objectives.

**Legend for this case:**

*Symbols:*
- $A$: **positive definite symmetric matrix written in Standard** (like covariance matrix)
- $L$: **lower triangular matrix written in Standard** (the "square root" of A)

*Operations:*
1. $L^T$: **Standard → Triangular** (coordinate change - not rotation!)
2. $L$: **Triangular → Standard** (coordinate change - not rotation!)

**Geometric story**: Factor into "square root" operations.

**Operation Language**: A, L written in Standard  
**System State Transitions**:
1. $A = L L^T$ where L is lower triangular
2. Start: Standard
3. $L^T$: Standard → Triangular_coords 🔀 coordinate change (not rotation!)
4. $L$: Triangular_coords → Standard 🔀 coordinate change back
5. End: Standard

**Special property**: For correlation/covariance matrices, this gives the "portfolio" decomposition.

---

## Universal Principles

### 🔄 **Rotations** (Orthogonal matrices)
- **Preserve**: distances, angles, geometric relationships
- **Change**: only the coordinate system viewpoint
- **Property**: $Q^T Q = I$, $\|Qx\| = \|x\|$

### 📏 **Scaling** (Diagonal matrices)  
- **Preserve**: coordinate directions (axes)
- **Change**: magnitudes along each axis
- **Property**: Acts independently on each coordinate

### 🔀 **General Coordinate Changes**
- **May distort**: distances, angles, geometric relationships  
- **Needed when**: basis vectors aren't orthogonal

### 🎯 **Projections**
- **Combine**: rotation to subspace + truncation + rotation back
- **Orthogonal projections**: pure rotations around truncation
- **General projections**: include metric corrections

---

## The Meta-Pattern

**Every decomposition answers**: "What's the simplest way to think about this operation?"

1. **Find the natural coordinate system** (where the operation becomes diagonal/simple)
2. **Identify what the operation actually does** (scale? project? rotate?)  
3. **Package as**: coordinate change + simple operation + coordinate change back

**Reading rule**: For any product $L S R$, read right-to-left:
- $R$: **TRANSLATE** - change system state to natural coordinates (may be rotation or general coordinate change)
- $S$: **TRANSFORM** - scaling in those coordinates
- $L$: **TRANSLATE BACK** - change system state back to original coordinates (inverse of $R$)

**Key insight**: The translate step (rightmost) may be orthogonal (rotation) or general (potentially distorting).

The magic is that this **same pattern** explains eigendecomposition, SVD, projections, least squares, whitening, and more. The differences are just in *which* coordinate systems are natural and *what* simple operation happens there.
