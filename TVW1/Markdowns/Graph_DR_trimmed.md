# Graph Douglas–Rachford — Codex-oriented technical spec

Source paper: **Graph and distributed extensions of the Douglas–Rachford method**  
Authors: Kristian Bredies, Enis Chenchene, Emanuele Naldi  
arXiv:2211.04782v1, 2022.

This is a compressed implementation-oriented version of the paper. It keeps the notation, mathematical objects, update rules, special graph cases, and diagnostics that are most useful for reading or writing code. It intentionally removes most proofs, historical discussion, experiments, and bibliographic detail.

---

## 0. Problem and global notation

Goal: solve the monotone inclusion

$$
\text{find } x\in\mathcal H \quad\text{such that}\quad
0\in (A_1+\cdots + A_N)x,
$$

where each $A_i:\mathcal H\to 2^{\mathcal H}$ is maximal monotone.

The method is a frugal resolvent splitting with minimal lifting: it stores $N-1$ variables in $\mathcal H$ and evaluates each resolvent once per iteration.

Important spaces and objects:

| Symbol | Meaning |
|---|---|
| $\mathcal H$ | component Hilbert space |
| $\mathcal D:=\mathcal H^{N-1}$ | reduced lifted space |
| $\mathbf H:=\mathcal H^{2N-1}=\mathcal H^N\times\mathcal H^{N-1}$ | full lifted PPP space |
| $x=(x_1,\dots,x_N)\in\mathcal H^N$ | per-operator primal/state estimates |
| $w=(w_1,\dots,w_{N-1})\in\mathcal D$ | lifted dual/state variables |
| $A=(A_1,\dots,A_N)$ | original monotone operators |
| $\mathbf A$ | diagonal operator: $\mathbf A x=(A_1x_1,\dots,A_Nx_N)$ |
| $J_{\gamma A_i}$ | resolvent $(I+\gamma A_i)^{-1}$ |
| $\sigma>0$ | step size; algorithm uses $J_{\frac{\sigma}{d_i}A_i}$ |
| $\theta_k\in(0,2]$ | relaxation; require $\sum_k\theta_k(2-\theta_k)=+\infty$ |
| $I$ | identity on $\mathcal H$ |
| bold linear maps | Kronecker lifts, e.g. $\mathbf Z=Z\otimes I$ |

If $A_i=\partial f_i$, the inclusion corresponds, under usual regularity assumptions, to minimizing

$$
f(x)=\sum_{i=1}^N f_i(x).
$$

---

## 1. Graphs

A **bilevel graph** is

$$
\mathrm{biG}=(\mathcal N,\mathcal E,\mathcal E'),
\qquad \mathcal N=\{1,\dots,N\}.
$$

It has two layers:

1. **State graph** $G=(\mathcal N,\mathcal E)$.
2. **Base graph** $G'=(\mathcal N,\mathcal E')$, a connected subgraph used to build the lifting matrix $Z$.

The state graph is directed and topologically ordered: if $(h,i)\in\mathcal E$, then $h<i$. Thus the sequential update of $x_1,x_2,\dots,x_N$ can use already-computed values $x_h^{k+1}$.

For any directed graph, adjacency is treated undirected when forming the Laplacian. The degree of node $i$ in the state graph is

$$
d_i = \# \operatorname{adj}(i;G).
$$

For directed degree notation used in diagnostics:

- $d_i^+$ = in-degree of node $i$ in the ordered directed graph.
- $d_i^-$ = out-degree of node $i$.

The state-graph unbalance is

$$
U_G=\sqrt{\frac1N\sum_{i=1}^N(d_i^- - d_i^+)^2}.
$$

---

## 2. Base graph Laplacian and the matrix \(Z\)

Given the base graph $G'=(\mathcal N,\mathcal E')$, define its graph Laplacian $L\in\mathbb R^{N\times N}$ by

$$
L_{ij}=
\begin{cases}
d_i' & i=j,\\
-1 & i\neq j \text{ and } i,j \text{ are adjacent in }G',\\
0 & \text{otherwise},
\end{cases}
$$

where $d_i'$ is the degree of $i$ in the base graph.

Choose $Z\in\mathbb R^{N\times(N-1)}$ such that

$$
L=ZZ^*,\qquad \operatorname{rank}Z=N-1,\qquad \ker Z^*=\operatorname{span}\{\mathbf 1\}.
$$

Equivalently, if $z_i$ denotes row $i$ of $Z$,

$$
z_1+\cdots+z_N=0,\qquad
\operatorname{span}\{z_1,\dots,z_N\}=\mathbb R^{N-1},
$$

and $z_i\cdot z_j\neq0$ iff $i,j$ are adjacent in $G'$.

### Practical construction of \(Z\)

For a general connected base graph:

1. Form the Laplacian $L$.
2. Compute a rank-$N-1$ factorization $L=ZZ^*$, e.g. spectral decomposition.
3. Use any valid $Z$; different choices are equivalent up to an orthogonal transform of $w$.

For a tree base graph:

- $|\mathcal E'|=N-1$.
- $Z$ can be the oriented incidence matrix.
- If edge $e_j=(h,i)$ is the $j$-th base edge, then

$$
Z_{rj}=
\begin{cases}
1 & r=h \quad\text{edge leaves }h,\\
-1 & r=i \quad\text{edge enters }i,\\
0 & \text{otherwise}.
\end{cases}
$$

Then $ZZ^*=L$.

---

## 3. Full lifted operator construction

The construction is useful for understanding why Algorithm 1 is a reduced degenerate PPP method.

Define the onto decomposition

$$
C^*=\begin{pmatrix}\mathbf Z^* & \mathbf I\end{pmatrix},
$$

where $\mathbf Z=Z\otimes I$ and $\mathbf I$ is identity on $\mathcal D$. Then

$$
M=CC^*
=
\begin{pmatrix}
\mathbf L & \mathbf Z\\
\mathbf Z^* & \mathbf I
\end{pmatrix},
\qquad
\mathbf L=\mathbf Z\mathbf Z^*=L\otimes I.
$$

Define the skew matrix $\Sigma\in\mathbb R^{N\times N}$ from the base-graph Laplacian $L$ by

$$
\Sigma_{ij}=
\begin{cases}
-L_{ij} & i<j,\\
L_{ij} & i>j,\\
0 & i=j.
\end{cases}
$$

Let $\mathbf\Sigma=\Sigma\otimes I$. Define

$$
B_L=\mathbf A+\mathbf\Sigma.
$$

For each state edge not in the base graph, $(i,j)\in\mathcal E\setminus\mathcal E'$, define $P^{ij}\in\mathbb R^{N\times N}$ by

$$
(P^{ij})_{hk}=
\begin{cases}
1 & h=i,\ k=i,\\
1 & h=j,\ k=j,\\
-2 & h=j,\ k=i,\\
0 & \text{otherwise}.
\end{cases}
$$

Set

$$
\mathbf P=\sum_{(i,j)\in\mathcal E\setminus\mathcal E'} P^{ij}\otimes I,
\qquad
A_L=B_L+\mathbf P.
$$

Finally define the maximal monotone lifted operator

$$
\mathcal A=
\begin{pmatrix}
A_L & -\mathbf Z\\
\mathbf Z^* & 0
\end{pmatrix}.
$$

Key property:

$$
0\in\mathcal A(x,v)
\quad\Longleftrightarrow\quad
x_1=\cdots=x_N=:x
\quad\text{and}\quad
0\in\sum_{i=1}^N A_i x.
$$

Thus zeros of $\mathcal A$ encode solutions of the original problem.

---

## 4. Reduced PPP form

The reduced variable is

$$
w^k=C^*u^k=\mathbf Z^*x^k+v^k.
$$

With relaxation $\theta_k=1$,

$$
\begin{cases}
x^{k+1}=(\mathbf L+A_L)^{-1}\mathbf Z w^k,\\
w^{k+1}=w^k-\mathbf Z^*x^{k+1}.
\end{cases}
$$

For general relaxation,

$$
\begin{cases}
x^{k+1}=(\mathbf L+A_L)^{-1}\mathbf Z w^k,\\
w^{k+1}=w^k-\theta_k\mathbf Z^*x^{k+1}.
\end{cases}
$$

The inverse $(\mathbf L+A_L)^{-1}$ is lower triangular with respect to the topological order of the state graph, giving the explicit resolvent updates in Algorithm 1.

---

## 5. Algorithm 1: Graph Douglas–Rachford

Inputs:

- maximal monotone operators $A_1,\dots,A_N$;
- bilevel graph $(\mathcal N,\mathcal E,\mathcal E')$;
- $Z\in\mathbb R^{N\times(N-1)}$ with $ZZ^*=L(G')$;
- state-graph degrees $d_i$;
- step size $\sigma>0$;
- relaxation sequence $\theta_k\in(0,2]$ with $\sum_k\theta_k(2-\theta_k)=+\infty$.

Initialize

$$
w_1^0,\dots,w_{N-1}^0\in\mathcal H.
$$

For $k=0,1,2,\dots$:

1. Sequentially for $i=1,\dots,N$ compute

$$
x_i^{k+1}
=
J_{\frac{\sigma}{d_i}A_i}
\left(
\frac{2}{d_i}\sum_{(h,i)\in\mathcal E}x_h^{k+1}
+
\frac{1}{d_i}\sum_{j=1}^{N-1}Z_{ij}w_j^k
\right).
$$

2. For $j=1,\dots,N-1$ update

$$
w_j^{k+1}
=
w_j^k
-
\theta_k
\sum_{i=1}^{N}Z_{ij}x_i^{k+1}.
$$

Implementation notes:

- The first sum uses **state edges** $(h,i)\in\mathcal E$.
- Because $\mathcal E$ is topologically ordered, all $x_h^{k+1}$ needed for $x_i^{k+1}$ have already been computed.
- Each resolvent $J_{\frac{\sigma}{d_i}A_i}$ is evaluated exactly once per iteration.
- If $A_i=\partial f_i$, then $J_{\gamma A_i}=\operatorname{prox}_{\gamma f_i}$.
- The lifted variable $w$ has only $N-1$ components in $\mathcal H$.

Convergence guarantee:

- If $\operatorname{zer}(A_1+\cdots+A_N)\neq\emptyset$, then all $w_j^k$ converge weakly to $w_j^*$.
- The induced $x_i^*$ coincide:

$$
x_1^*=\cdots=x_N^*,
$$

and this common point solves the original inclusion.
- The sequences $x_i^k$ converge weakly to that solution.

---

## 6. Tree base graph specialization

Assume $G'$ is a tree and $Z$ is its oriented incidence matrix. The $w$ variables can be indexed by base edges $(h,i)\in\mathcal E'$:

$$
w_j \equiv w_{(h,i)}.
$$

Initialize

$$
w_{(h,i)}^0\in\mathcal H
\qquad\text{for all }(h,i)\in\mathcal E'.
$$

For $k=0,1,2,\dots$:

1. Sequentially for $i=1,\dots,N$ compute

$$
x_i^{k+1}
=
J_{\frac{\sigma}{d_i}A_i}
\left(
\frac{2}{d_i}\sum_{(h,i)\in\mathcal E}x_h^{k+1}
+
\frac{1}{d_i}
\left[
\sum_{(i,j)\in\mathcal E'}w_{(i,j)}^k
-
\sum_{(h,i)\in\mathcal E'}w_{(h,i)}^k
\right]
\right).
$$

2. For each base edge $(h,i)\in\mathcal E'$ update

$$
w_{(h,i)}^{k+1}
=
w_{(h,i)}^k
+
\theta_k\left(x_i^{k+1}-x_h^{k+1}\right).
$$

Implementation notes:

- Tree base graphs avoid numerical factorization of the Laplacian.
- The sign convention above matches the incidence convention: $+1$ at the edge tail, $-1$ at the edge head.
- The update of $w_{(h,i)}$ measures the disagreement across the base edge.

---

## 7. Important special cases

### 7.1 Ryu-type \(N\)-operator splitting

Use:

- complete state graph;
- star-shaped tree base graph rooted at node $N$:

$$
\mathcal E'=\{(i,N):i=1,\dots,N-1\}.
$$

Then, with standard indexing $w_i=w_{(i,N)}$,

$$
\begin{cases}
x_i^{k+1}
=
J_{\frac{\sigma}{N-1}A_i}
\left(
\dfrac{2}{N-1}\displaystyle\sum_{h=1}^{i-1}x_h^{k+1}
+
\dfrac{1}{N-1}w_i^k
\right),
& i=1,\dots,N-1,
\\[2ex]
x_N^{k+1}
=
J_{\frac{\sigma}{N-1}A_N}
\left(
\dfrac{2}{N-1}\displaystyle\sum_{h=1}^{N-1}x_h^{k+1}
-
\dfrac{1}{N-1}\displaystyle\sum_{j=1}^{N-1}w_j^k
\right),
\\[2ex]
w_j^{k+1}
=
w_j^k+\theta_k(x_N^{k+1}-x_j^{k+1}),
& j=1,\dots,N-1.
\end{cases}
$$

For $N=3$, after a rescaling of $w$, this recovers Ryu's three-operator splitting.

### 7.2 Malitsky–Tam splitting

Use:

$$
\mathcal E'=\{(i,i+1):i=1,\dots,N-1\}
$$

and

$$
\mathcal E=\mathcal E'\cup\{(1,N)\}.
$$

Then Algorithm 2 becomes the Malitsky–Tam splitting, interpreted here as a reduced proximal point method.

### 7.3 Three-operator complete base graph

For $N=3$ and complete base graph, one valid factorization is

$$
Z^*
=
\begin{pmatrix}
\sqrt2 & -\sqrt{1/2} & -\sqrt{1/2}\\
0 & \sqrt{3/2} & -\sqrt{3/2}
\end{pmatrix}.
$$

With variables

$$
\widetilde w_1^k=\sqrt2\,w_1^k,
\qquad
\widetilde w_2^k=\sqrt{2/3}\,w_2^k,
$$

Algorithm 1 becomes

$$
\begin{cases}
x_1^{k+1}
=
J_{\frac{\sigma}{2}A_1}
\left(
\dfrac12\widetilde w_1^k
\right),
\\[1.5ex]
x_2^{k+1}
=
J_{\frac{\sigma}{2}A_2}
\left(
x_1^{k+1}
+
\dfrac34\widetilde w_2^k
-
\dfrac14\widetilde w_1^k
\right),
\\[1.5ex]
x_3^{k+1}
=
J_{\frac{\sigma}{2}A_3}
\left(
x_1^{k+1}
+
x_2^{k+1}
-
\dfrac34\widetilde w_2^k
-
\dfrac14\widetilde w_1^k
\right),
\\[1.5ex]
\widetilde w_1^{k+1}
=
\widetilde w_1^k
-
\theta_k(2x_1^{k+1}-x_2^{k+1}-x_3^{k+1}),
\\[1.5ex]
\widetilde w_2^{k+1}
=
\widetilde w_2^k
-
\theta_k(x_2^{k+1}-x_3^{k+1}).
\end{cases}
$$

---

## 8. Residuals and diagnostics

Define the fixed-point map

$$
\widetilde T=(I+C^*\triangleright\mathcal A)^{-1}.
$$

In Algorithm 1,

$$
\widetilde T w^k = w^k-\mathbf Z^*x^{k+1}.
$$

Thus the fixed-point residual is

$$
\|\widetilde T w^k-w^k\|
=
\|\mathbf Z^*x^{k+1}\|.
$$

If $0<\inf_k\theta_k\leq\sup_k\theta_k<2$, then

$$
\|\widetilde T w^k-w^k\|^2=o(k^{-1}),
\qquad
\|w^{k+1}-w^k\|^2=o(k^{-1}).
$$

### State variance

Define

$$
\bar x^k=\frac1N\sum_{i=1}^N x_i^k,
\qquad
\operatorname{Var}(x^k)
=
\frac1N\sum_{i=1}^N\|x_i^k-\bar x^k\|^2.
$$

Let $\lambda_1$ be the algebraic connectivity of the base graph, i.e. the first nonzero eigenvalue of $L(G')$. Then

$$
\operatorname{Var}(x^{k+1})
\leq
\frac{1}{\lambda_1 N}
\|\widetilde T w^k-w^k\|^2.
$$

Hence higher algebraic connectivity of the base graph can improve the consensus bound.

### Operator residual

Algorithm 1 implicitly defines $a_i^{k+1}\in A_ix_i^{k+1}$ by

$$
\mathbf Lx^{k+1}
+
(\mathbf\Sigma+\mathbf P)x^{k+1}
+
\sigma a^{k+1}
=
\mathbf Zw^k,
\qquad
a^{k+1}=(a_1^{k+1},\dots,a_N^{k+1}).
$$

The sum residual is bounded by state variance:

$$
\left\|\sum_{i=1}^N a_i^k\right\|^2
\leq
\frac{U_G^2N^2}{\sigma^2}
\operatorname{Var}(x^k).
$$

In particular, if $x_1^k=\cdots=x_N^k=:x^*$ at some iteration, then $x^*$ solves the original inclusion.

Combined estimate:

$$
\frac{\sigma^2}{N^2}
\left\|\sum_{i=1}^N a_i^{k+1}\right\|^2
\leq
U_G^2\operatorname{Var}(x^{k+1})
\leq
\frac{U_G^2}{\lambda_1N}
\|\widetilde T w^k-w^k\|^2
=
o(k^{-1}).
$$

---

## 9. Objective-function case

Assume

$$
A_i=\partial f_i,
$$

with $f_i$ proper, convex, lower semicontinuous, and usual regularity so that

$$
0\in\sum_i \partial f_i(x)
$$

corresponds to minimizing

$$
f(x)=\sum_i f_i(x).
$$

If relaxation satisfies $\theta_k\in[\varepsilon,2-\varepsilon]$, and all $f_i$ except possibly one $f_j$ are locally Lipschitz, then

$$
f(x_j^k)-\inf f=o(k^{-1/2}).
$$

If all $f_i$ are locally Lipschitz, then also

$$
f(\bar x^k)-\inf f=o(k^{-1/2}).
$$

If at least one operator $A_j$ is uniformly monotone on bounded sets of its domain, then all $x_i^k$ converge strongly to the solution.

---

## 10. Distributed protocols

The graph-based method can be interpreted as a distributed protocol where agent $i$ owns $A_i$ and computes $x_i$.

### 10.1 Tree base graph distributed protocol

Use Algorithm 2 with base-edge variables $w_{(h,i)}$.

Storage convention:

- For each base edge $(h,i)\in\mathcal E'$, agent $i$ stores and updates $w_{(h,i)}$.

Initialization:

- Agent $i$ chooses $w_{(h,i)}^0$ for all incoming base edges $(h,i)\in\mathcal E'$.
- Agent $i$ sends those initial values to the corresponding agents $h$.

At each iteration, agent $i$:

1. Receives:
   - $w_{(i,j)}^k$ from each $j$ with $(i,j)\in\mathcal E'$;
   - $x_h^{k+1}$ from each $h$ with $(h,i)\in\mathcal E$.
2. Computes $x_i^{k+1}$ using the tree-base update.
3. Updates $w_{(h,i)}^{k+1}$ for all incoming base edges $(h,i)\in\mathcal E'$.
4. Sends:
   - $x_i^{k+1}$ to each $j$ with $(i,j)\in\mathcal E$;
   - $w_{(h,i)}^{k+1}$ to each $h$ with $(h,i)\in\mathcal E'$.

### 10.2 General base graph distributed protocol

For a general base graph, avoid sparse factorization of $L$ by changing variables

$$
\widetilde w^k=\mathbf Z w^k\in\mathcal H^N.
$$

Then

$$
\begin{cases}
x^{k+1}=(\mathbf L+A_L)^{-1}\widetilde w^k,\\
\widetilde w^{k+1}
=
\widetilde w^k
-
\theta_k\mathbf Lx^{k+1}.
\end{cases}
$$

The initialization must satisfy

$$
\widetilde w^0\in\operatorname{Im}Z
\quad\Longleftrightarrow\quad
\sum_{i=1}^N \widetilde w_i^0=0.
$$

A simple choice is

$$
\widetilde w_i^0=0
\qquad\forall i.
$$

Agent $i$ stores $\widetilde w_i$.

At each iteration:

1. Agent $i$ receives $x_h^{k+1}$ from each $h$ with $(h,i)\in\mathcal E$.
2. Agent $i$ computes

$$
x_i^{k+1}
=
J_{\frac{\sigma}{d_i}A_i}
\left(
\frac{2}{d_i}\sum_{(h,i)\in\mathcal E}x_h^{k+1}
+
\frac{1}{d_i}\widetilde w_i^k
\right).
$$

3. Agent $i$ sends $x_i^{k+1}$:
   - to each $j$ with $(i,j)\in\mathcal E$;
   - to each $h$ with $(h,i)\in\mathcal E'$.
4. Agent $i$ receives $x_j^{k+1}$ from all $j\in\operatorname{adj}(i;G')$ not already received.
5. Agent $i$ updates

$$
\widetilde w_i^{k+1}
=
\widetilde w_i^k
-
\theta_k
\left(
d_i'x_i^{k+1}
-
\sum_{j\in\operatorname{adj}(i;G')}x_j^{k+1}
\right).
$$

Notes:

- This version uses $N$ lifted variables instead of $N-1$.
- It avoids explicitly choosing a sparse $Z$.
- It generally needs two communication phases.
- Only the estimates $x_i^k$ are shared, not the operator data.

---

## 11. Implementation checklist

Given a Python implementation, check these objects:

1. **Graph data**
   - `N`
   - state edges `E`
   - base edges `E_prime`
   - topological order: every `(h, i) in E` should satisfy `h < i` if using 1-based paper indexing.
   - degrees `d_i` from the state graph, treating adjacency undirected.

2. **Base matrix**
   - for general base graph: compute Laplacian `L_base`;
   - compute `Z` with shape `(N, N-1)` and `Z @ Z.T == L_base`;
   - for tree base graph: build oriented incidence matrix.

3. **Main variables**
   - `x`: shape `(N, ...)`;
   - `w`: shape `(N-1, ...)` for Algorithm 1/2;
   - or `w_tilde`: shape `(N, ...)` for the general-base distributed form.

4. **Resolvent calls**
   - update `x_i` sequentially in state-graph topological order;
   - call exactly one resolvent/prox per operator per iteration;
   - parameter for operator `i` is `sigma / d_i`.

5. **State update**
   - Algorithm 1:

     $$
     w^{k+1}=w^k-\theta_k Z^*x^{k+1}.
     $$

   - Tree-base edge form:

     $$
     w_{(h,i)}^{k+1}=w_{(h,i)}^k+\theta_k(x_i^{k+1}-x_h^{k+1}).
     $$

   - General-base distributed form:

     $$
     \widetilde w^{k+1}=\widetilde w^k-\theta_k Lx^{k+1}.
     $$

6. **Diagnostics**
   - consensus variance:

     $$
     \operatorname{Var}(x^k)=\frac1N\sum_i\|x_i^k-\bar x^k\|^2;
     $$

   - fixed-point residual:

     $$
     \|Z^*x^{k+1}\|;
     $$

   - for objective case, monitor $f(x_i^k)$ or $f(\bar x^k)$ if computable.

---

## 12. Minimal pseudocode

### General Algorithm 1

```python
# E uses paper convention: edge (h, i) means x_h is needed before x_i.
# Z has shape (N, N-1).
# w has shape (N-1, *space_shape).
# prox[i](y, gamma) returns J_{gamma A_i}(y).

for k in range(max_iter):
    x_new = empty_like_x()

    for i in range(N):  # topological order
        incoming_sum = sum(x_new[h] for (h, ii) in E if ii == i)
        zw_i = sum(Z[i, j] * w[j] for j in range(N - 1))

        arg = (2.0 / d[i]) * incoming_sum + (1.0 / d[i]) * zw_i
        x_new[i] = prox[i](arg, sigma / d[i])

    ztx = sum(Z[i, j] * x_new[i] for i in range(N) for j in range(N - 1))
    # More explicitly:
    for j in range(N - 1):
        w[j] = w[j] - theta[k] * sum(Z[i, j] * x_new[i] for i in range(N))

    x = x_new
```

### Tree-base Algorithm 2

```python
# w_edge[(h, i)] is stored for each edge in E_prime.
# d[i] is the degree in the state graph.

for k in range(max_iter):
    x_new = empty_like_x()

    for i in range(N):
        incoming_state = sum(x_new[h] for (h, ii) in E if ii == i)

        outgoing_base_w = sum(w_edge[(i, j)] for (ii, j) in E_prime if ii == i)
        incoming_base_w = sum(w_edge[(h, i)] for (h, ii) in E_prime if ii == i)

        arg = (
            (2.0 / d[i]) * incoming_state
            + (1.0 / d[i]) * (outgoing_base_w - incoming_base_w)
        )

        x_new[i] = prox[i](arg, sigma / d[i])

    for (h, i) in E_prime:
        w_edge[(h, i)] = w_edge[(h, i)] + theta[k] * (x_new[i] - x_new[h])

    x = x_new
```

Potential implementation caveat: if Python uses 0-based indexing but the paper uses 1-based indexing, translate all edge tuples consistently.
