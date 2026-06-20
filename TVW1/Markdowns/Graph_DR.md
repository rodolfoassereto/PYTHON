# Graph and distributed extensions of the Douglas–Rachford method

**Authors:** Kristian Bredies, Enis Chenchene, Emanuele Naldi
**Date:** November 10, 2022
**Reference:** arXiv:2211.04782v1 [math.OC], 9 Nov 2022

**Affiliations:**
- Kristian Bredies, Enis Chenchene — Institute of Mathematics and Scientific Computing, University of Graz, Graz, Austria. (`kristian.bredies@uni-graz.at`, `enis.chenchene@uni-graz.at`)
- Emanuele Naldi — Institute of Analysis and Algebra, TU Braunschweig. (`e.naldi@tu-braunschweig.de`)

> **Note on this document.** This is a Markdown transcription of the uploaded PDF, prepared for easy reading. The mathematical notation has been reconstructed in LaTeX from the (lossy) PDF text extraction. A few notational conventions used throughout:
> - $\mathcal{H}$ — the underlying (component) Hilbert space; each operator $A_i$ acts on $\mathcal{H}$.
> - $\mathbf{H} := \mathcal{H}^{2N-1}$ — the lifted product space used in the construction of Section 3.
> - $\mathcal{D} := \mathcal{H}^{N-1}$ — the reduced space.
> - $\mathbf{A}$ (bold) — the diagonal operator $(x_1,\dots,x_N)\mapsto(A_1x_1,\dots,A_Nx_N)$; $\mathcal{A}$ (calligraphic) — the assembled lifted operator of equation (12).
> - $G' = (\mathcal{N},\mathcal{E}')$ — the **base graph** (the prime is rendered as a superscript "0" in some places of the source).
> - $C^{*} \triangleright A := \left(C^{*}A^{-1}C\right)^{-1}$ — the **parallel composition**.
> - Boldface linear maps (e.g. $\mathbf{L}, \mathbf{Z}, \mathbf{\Sigma}, \mathbf{P}$) denote the Kronecker lift $\,\cdot\otimes I\,$ of the corresponding real matrix acting on $\mathcal{H}^N$.

---

## Abstract

In this paper, we propose several graph-based extensions of the Douglas–Rachford splitting (DRS) method to solve monotone inclusion problems involving the sum of $N$ maximal monotone operators. Our construction is based on a two-layer architecture that we refer to as **bilevel graphs**, to which we associate a generalization of the DRS algorithm that presents the prescribed structure. The resulting schemes can be understood as unconditionally stable frugal resolvent splitting methods with a minimal lifting in the sense of Ryu [Math Program 182(1):233–273, 2020], as well as instances of the (degenerate) Preconditioned Proximal Point method, which provides robust convergence guarantees. We further describe how the graph-based extensions of the DRS method can be leveraged to design new fully distributed protocols. Applications to a congested optimal transport problem and to distributed Support Vector Machines show interesting connections with the underlying graph topology and highly competitive performances with state-of-the-art distributed optimization approaches.

---

## 1. Introduction

The proximal point algorithm is a widely used tool for solving a variety of problems such as finding zeros of maximal monotone operators, fixed-points of nonexpansive mappings, as well as minimizing convex functions. Given a Hilbert space $\mathcal{H}$, the **Preconditioned Proximal Point (PPP)** method can be understood as a proximal point method with respect to a new metric induced by a self-adjoint (uniformly) positive definite linear map $M : \mathcal{H} \to \mathcal{H}$. For a maximal monotone operator $A : \mathcal{H} \to 2^{\mathcal{H}}$, the general iteration of a PPP method reads

$$
u^0 \in \mathcal{H}, \qquad u^{k+1} = u^k + \theta_k\left(T u^k - u^k\right) \quad \text{for all } k \in \mathbb{N}, \tag{1}
$$

where $T := (M + A)^{-1} M$, and $\theta_k \in (0, 2]$ are relaxation parameters that satisfy $\sum_k \theta_k(2-\theta_k) = +\infty$.

In our recent work [1], we focused on the degenerate case, i.e., assuming that $M$ is only positive semidefinite, allowing in this way $M$ to have a possibly large kernel. In that case, for the iterations in (1) to make sense, we restricted the analysis to the class of **admissible preconditioners**, i.e., such that $T$ is everywhere defined and single-valued. The degenerate PPP framework [1–3] allows us to study in a unifying theory a large class of known (and new) splitting methods such as Chambolle–Pock [4] (also in the degenerate case, i.e., in the notation of [4], where $\tau\sigma L^2 = 1$), Peaceman–Rachford [5], Davis–Yin [6] and Douglas–Rachford [7], and to easily derive new extensions to the $N$-operator problem:

$$
\text{find } x \in \mathcal{H} \text{ such that: } \quad 0 \in (A_1 + \cdots + A_N)\,x, \tag{2}
$$

where $A_i$ are maximal monotone operators on the Hilbert space $\mathcal{H}$. To solve problem (2), which we will always assume possible, we consider the class of so-called **frugal resolvent splitting (FRS)** methods introduced by Ryu in [8]. These are iterative methods, which at every iteration only require a single evaluation of the resolvents, i.e., $J_{\sigma_i A_i} := (I + \sigma_i A_i)^{-1}$ for some $\sigma_i > 0$, and simple algebraic operations, such as vector additions and scalar multiplications. It has also been proven in [8] and later extended in [9] that if $N > 2$, unconditionally stable FRS methods, i.e., which produce (weakly) convergent sequences to a solution to (2) for every tuple $(A_1, \dots, A_N)$ of maximal monotone operators, can only be designed on a $d$-fold product space with $d \geq N - 1$, thus requiring several additional variables. FRS methods with $d = N - 1$ are said to have a **minimal lifting** or **minimal variables**. In this paper, we focus on FRS methods with minimal variables.

While for the two-operator case, i.e., with $N = 2$, the class of unconditionally stable FRS methods with minimal variables reduces to the celebrated DRS method [8, Corollary 1], for larger problems, the resulting schemes present many different structures. Unconditionally stable FRS schemes with minimal variables and parallel structures can be derived with the so-called product-space trick, see, e.g., [10, Section 9.1]. Schemes with different structures have been discovered more recently. The Sequential DRS, introduced in [1], presents a purely sequential nature, which is very close to the method introduced by Malitsky and Tam in [9], where the pure sequentiality is in some sense broken with an additional communication between the first and the last operator. The Malitsky–Tam splitting, when $N = 3$, is in turn highly related, yet not equivalent, to the method introduced by Ryu in [8]. This systematic unfolding of structurally different FRS methods with minimal variables leads us to the natural question: **can all the structures be achieved?** The main novelty that this paper provides is a positive answer to this question, in a sense that we will make precise in the course of the paper, cf., Corollary 3.6.

The rest of this paper is organized as follows. In **Section 2** we introduce some preliminary notions, all the terminology and results related to the degenerate PPP framework and to the theory of FRS methods, along with the notion of bilevel graph. **Section 3** presents the proposed graph-based extensions of the DRS method along with some properties. In **Section 4** we show how the graph-based DRS can be leveraged to design new fully distributed schemes for (2) assuming tree or more general base graphs. In **Section 5** we show an application to a congested optimal transport problem emphasizing, in particular, an interesting influence of the algebraic connectivity of the graph topology on the convergence speed of the method. We conclude with an application to distributed Support Vector Machines, showing that the devised distributed schemes reach highly competitive performances compared to state-of-the-art methods such as P-EXTRA [11] and a distributed variant of the PDHG method [4].

---

## 2. Background and preliminary results

Let $\mathcal{H}$ be a real Hilbert space, $A : \mathcal{H} \to 2^{\mathcal{H}}$ be a maximal monotone operator and let $M : \mathcal{H} \to \mathcal{H}$ be a self-adjoint linear bounded operator. Finding a zero of $A$, i.e., a point $u \in \mathcal{H}$ such that $0 \in Au$, could be formulated as a fixed-point inclusion problem $u \in Tu$, with $T := (M + A)^{-1} M$. Even if $M$ is not invertible in the classical sense, we shall still consider $M^{-1}A$ as a composition of multivalued operators and it holds that $T = \left(I + M^{-1}A\right)^{-1}$. Note that $M$ defines a seminorm on $\mathcal{H}$, that is $\|u\|_M^2 = \langle Mu, u\rangle$ for all $u \in \mathcal{H}$. The following decomposition of $M$ will be useful, see [1, Proposition 2.3] for a proof.

> **Proposition 2.1.** Let $M : \mathcal{H} \to \mathcal{H}$ be a self-adjoint, linear, bounded, positive semidefinite operator. Then, there exists an injective operator $C : \mathcal{D} \to \mathcal{H}$, for some real Hilbert space $\mathcal{D}$, such that $M = CC^{*}$. Moreover, if $M$ has closed range, then $C^{*}$ is onto.

When $M$ has closed range, we call any factorization $M = CC^{*}$, with $C : \mathcal{D} \to \mathcal{H}$ injective and $\mathcal{D}$ a Hilbert space, an **onto decomposition** of $M$. In the following result, we prove that, once $\mathcal{D}$ is fixed, onto decompositions are unique modulo orthogonal transformations.

> **Proposition 2.2.** Let $M : \mathcal{H} \to \mathcal{H}$ be a self-adjoint, linear, bounded, positive semidefinite operator with closed range. Then, $M = CC^{*}$, with $C : \mathcal{D} \to \mathcal{H}$, is an onto decomposition of $M$ if and only if for every onto decomposition $M = \widetilde{C}\widetilde{C}^{*}$, with $\widetilde{C} : \widetilde{\mathcal{D}} \to \mathcal{H}$, there exists a linear isomorphism $O : \mathcal{D} \to \widetilde{\mathcal{D}}$ with $O^{-1} = O^{*}$, such that $C = \widetilde{C}O$.

**Proof.** First, note that, since $M$ has closed range, $\operatorname{Im}M$ equipped with the $M$-seminorm defines a Hilbert space. Further, for every onto decomposition $M = CC^{*}$, with $C : \mathcal{D} \to \mathcal{H}$, the operator $C^{*}$ is onto, and thus, $\operatorname{Im}C = \operatorname{Im}M$. It is easy to observe that $C^{*}|_{\operatorname{Im}M} : \operatorname{Im}M \to \mathcal{D}$, where $C^{*}|_{\operatorname{Im}M}$ is the restriction of $C^{*}$ to $\operatorname{Im}M = (\ker C^{*})^{\perp}$, and $C : \mathcal{D} \to \operatorname{Im}M$ define linear isomorphisms.

Now, given two onto decompositions $M = CC^{*} = \widetilde{C}\widetilde{C}^{*}$ with $C : \mathcal{D} \to \mathcal{H}$ and $\widetilde{C} : \widetilde{\mathcal{D}} \to \mathcal{H}$, we have $C^{*}CC^{*} = C^{*}\widetilde{C}\widetilde{C}^{*}$ and, since $C^{*}C$ is a linear isomorphism, we can write

$$
C^{*} = (C^{*}C)^{-1}C^{*}\widetilde{C}\widetilde{C}^{*}. \tag{3}
$$

From (3), it follows that $C^{*}C = (C^{*}C)^{-1}C^{*}\widetilde{C}\widetilde{C}^{*}C$ and again, since $C^{*}C$ is invertible, we get

$$
I = \left[(C^{*}C)^{-1}C^{*}\widetilde{C}\right]\left[\widetilde{C}^{*}C(C^{*}C)^{-1}\right]. \tag{4}
$$

Therefore, letting $O = \widetilde{C}^{*}C(C^{*}C)^{-1}$, we get from (4) that $O^{*}O = I$, and from (3), taking adjoints, that $C = \widetilde{C}O$. Being a composition of two linear isomorphisms, namely $\widetilde{C}^{*}C$ and $(C^{*}C)^{-1}$, the operator $O$ is a linear isomorphism between $\mathcal{D}$ and $\widetilde{\mathcal{D}}$, and $O^{-1} = O^{*}$. The converse statement is immediately clear. $\quad\blacksquare$

In [1] we show that if the preconditioner $M$ has closed range and $M = CC^{*}$ is an onto decomposition with $C : \mathcal{D} \to \mathcal{H}$, proximal point iterations with respect to $M^{-1}A$ are in some sense equivalent to proximal point iterations with respect to the so-called **parallel composition** $C^{*} \triangleright A := \left(C^{*}A^{-1}C\right)^{-1}$, which is defined on $\mathcal{D}$. The reason lies in the following result, proven in [1] and in [12] simultaneously.

> **Lemma 2.3.** Let $A : \mathcal{H} \to 2^{\mathcal{H}}$ be an operator, $M : \mathcal{H} \to \mathcal{H}$ be an admissible preconditioner with closed range and $M = CC^{*}$ be an onto decomposition with $C : \mathcal{D} \to \mathcal{H}$. Then, the operator $C^{*} \triangleright A$ is maximal monotone in $\mathcal{D}$ and
> $$
> (I + C^{*} \triangleright A)^{-1} = C^{*}(M + A)^{-1}C. \tag{5}
> $$

**The reduced scheme.** An onto decomposition of $M$ allows us to derive what we called in [1] the **reduced PPP method**. Indeed, since $M = CC^{*}$ for $C : \mathcal{D} \to \mathcal{H}$, the general PPP iteration writes

$$
u^{k+1} = u^k + \theta_k\left((M + A)^{-1}CC^{*}u^k - u^k\right). \tag{6}
$$

Simply applying $C^{*}$ to (6) and considering $w^k = C^{*}u^k$ for all $k \in \mathbb{N}$, we get, using (5), that

$$
w^{k+1} = w^k + \theta_k\left(C^{*}(M + A)^{-1}Cw^k - w^k\right) = w^k + \theta_k\left((I + C^{*} \triangleright A)^{-1}w^k - w^k\right). \tag{7}
$$

The method (7) is then a classical proximal point scheme with respect to the operator $(I + C^{*} \triangleright A)^{-1}$ that we denote by $\widetilde{T}$. We called such a method the reduced PPP method since, as we also saw in the proof of Proposition 2.2, the space $\mathcal{D}$ is (isometrically) isomorphic to the Hilbert space $\operatorname{Im}M$ endowed with the $M$-seminorm, which, if $M$ is degenerate, is strictly contained in $\mathcal{H}$.

> **Remark 2.4.** The reduced scheme does not depend on the onto decomposition of $M$. Indeed, if $CC^{*} = \widetilde{C}\widetilde{C}^{*} = M$ are two onto decompositions with $C : \mathcal{D} \to \mathcal{H}$ and $\widetilde{C} : \widetilde{\mathcal{D}} \to \mathcal{H}$, then by Proposition 2.2 we have $C = \widetilde{C}O$ for some linear isomorphism $O$ with $O^{-1} = O^{*}$. Thus, since the corresponding reduced sequences, $\{w^k\}_k$ and $\{\widetilde{w}^k\}_k$, satisfy $w^k = C^{*}u^k$ and $\widetilde{w}^k = \widetilde{C}^{*}u^k$, we have, for all $k \in \mathbb{N}$,
> $$
> w^k = C^{*}u^k = O^{*}\widetilde{C}^{*}u^k = O^{*}\widetilde{w}^k. \tag{8}
> $$
> Therefore, $Ow^k = \widetilde{w}^k$ for all $k \in \mathbb{N}$, and, hence, the two algorithms are equivalent.

**Convergence result.** The convergence analysis for degenerate PPP methods has been investigated in [1]. We summarize the main convergence result.

> **Theorem 2.5.** Let $A : \mathcal{H} \to 2^{\mathcal{H}}$ with $\operatorname{zer}A \neq \emptyset$ be a maximal monotone operator and $M$ be an admissible preconditioner with closed range. Let $M = CC^{*}$ be an onto decomposition of $M$ with $C : \mathcal{D} \to \mathcal{H}$, and let $T = (I + M^{-1}A)^{-1}$, $\widetilde{T} = (I + C^{*} \triangleright A)^{-1}$. Denote by $\{u^k\}_k$ the PPP sequence according to (1) and $\{w^k\}_k$ the corresponding reduced sequence according to (7). Then, we have
> 1. $\{w^k\}_k$ weakly converges in $\mathcal{D}$ to a point $w^{*} \in \mathcal{D}$ such that $u^{*} = (M + A)^{-1}Cw^{*} \in \operatorname{zer}A$.
> 2. If $(M + A)^{-1}$ is Lipschitz, then $\{(M + A)^{-1}Cw^k\}_k$ weakly converges to $u^{*}$.

We refer to [1, Theorem 2.14] and [1, Corollary 2.15] for a proof and further comments.

### 2.1 Frugal resolvent splitting methods

To tackle the $N$-operator problem (2) we consider the class of FRS methods introduced by Ryu in [8] and, later, further investigated in [9]. Here, for the reader's convenience, we outline the main properties and results on this class of methods, referring for further details to [9]. Let $\mathcal{H}$ be a Hilbert space and for $N \geq 1$ let $\mathcal{M}_N$ be the set of all $N$-tuples of maximal monotone operators on $\mathcal{H}$.

> **Definition 2.6 (Fixed-point encoding).** Let $\mathcal{D}$ and $\mathcal{H}$ be Hilbert spaces. A pair of operators $(T, S)$, with $T : \mathcal{M}_N \times \mathcal{D} \to \mathcal{D}$ and $S : \mathcal{M}_N \times \mathcal{D} \to \mathcal{H}$, is a **fixed-point encoding** for $\mathcal{M}_N$ if, for all $A = (A_1, \dots, A_N) \in \mathcal{M}_N$, the following hold:
> 1. $\operatorname{Fix}T(A, \cdot) \neq \emptyset$ if and only if $\operatorname{zer}(A_1 + \cdots + A_N) \neq \emptyset$,
> 2. If $w = T(A, w)$, then $S(A, w) \in \operatorname{zer}(A_1 + \cdots + A_N)$.
>
> The map $T$ is called **fixed-point operator**, and $S$ is called **solution operator**.

Fixed-point encodings, and in particular fixed-point operators, naturally define an associated fixed-point algorithm, namely for any $A \in \mathcal{M}_N$,

$$
w^{k+1} = T(A, w^k), \qquad \text{for } w^0 \in \mathcal{D}. \tag{9}
$$

> **Definition 2.7 (Unconditional stability).** The fixed-point encoding $(T, S)$ is **unconditionally stable** if for any starting point $w^0 \in \mathcal{D}$ and any $A = (A_1, \dots, A_N) \in \mathcal{M}_N$ with $\operatorname{zer}(A_1 + \cdots + A_N) \neq \emptyset$, the corresponding fixed-point algorithm (9) weakly converges to a fixed-point of $T(A, \cdot)$.

For the case $N = 1$ we have for instance: $A = (A_1)$, $T(A, \cdot) = J_{A_1}$ and $S(A, \cdot) = I$, where $I$ is the identity operator on $\mathcal{H}$, which corresponds to the proximal point algorithm. For $N = 2$, we can choose: $A = (A_1, A_2)$, $T(A, \cdot) = I + J_{A_2}(2J_{A_1} - I) - J_{A_1}$ and $S(A, \cdot) = J_{A_1}$, which yields the Douglas–Rachford algorithm. We also notice that in the two examples above the operators $T$ and $S$ can be evaluated efficiently applying successively (and only once) $J_{A_1}$ and $J_{A_2}$, which are assumed to be simple enough. This idea can be fixed by a definition.

> **Definition 2.8 (Frugal resolvent splitting).** We say that a fixed-point encoding $(T, S)$ is a **resolvent splitting** if, for all $A = (A_1, \dots, A_N) \in \mathcal{M}_N$, there is a finite procedure that evaluates $T(A, \cdot)$ and $S(A, \cdot)$ at a given point that uses only vector addition, scalar multiplication, and the resolvents of $A_1, \dots, A_N$. A resolvent splitting is **frugal** if, in addition, each of the resolvents of $A_1, \dots, A_N$ is evaluated exactly once.

Given $A = (A_1, \dots, A_N) \in \mathcal{M}_N$ with $\operatorname{zer}(A_1 + \cdots + A_N) \neq \emptyset$, a **FRS method** is the fixed-point algorithm associated with a frugal resolvent splitting $(T, S)$, with $T(A, \cdot) : \mathcal{D} \to \mathcal{D}$. Note that, roughly speaking, if the space $\mathcal{D}$ is large, implementing a FRS method may lead to huge memory requirements. For this reason, one should put adequate care on the definition of $\mathcal{D}$.

> **Definition 2.9 (Lifting).** Let $d \in \mathbb{N}$. A fixed-point encoding $(T, S)$ has a **$d$-fold lifting** for $\mathcal{M}_N$ if $\mathcal{D} = \mathcal{H}^d$.

A ground-breaking series of results initiated by Ryu in [8] for the three-operator problem and later extended by Malitsky and Tam in [9] for the general problem states that there is an inherent lower bound on the number of variables, i.e. $d$, for an unconditionally stable FRS method.

> **Theorem 2.10 (Minimal lifting [8, 9]).** Let $(T, S)$ be an unconditionally stable FRS for $\mathcal{M}_N$ with a $d$-fold lifting. If $N \geq 2$, then $d \geq N - 1$.

The authors proceed to show that the bound $N - 1$ is tight, and it is for this reason that we say that an unconditionally stable FRS method for $\mathcal{M}_N$ has minimal variables or a minimal lifting when $\mathcal{D} = \mathcal{H}^{N-1}$, i.e., algorithm (9) requires storing $N - 1$ variables living in $\mathcal{H}$. Interestingly, in [9] the authors also provided an explicit characterization of the general structure of a FRS, which will be helpful for our subsequent discussion.

> **Lemma 2.11 (Lemma 3.1 in [9]).** Let $(T, S)$ be a FRS for $\mathcal{M}_N$ with a $d$-fold lifting. Let $I$ be the identity on $\mathcal{H}$ and fix $A = (A_1, \dots, A_N) \in \mathcal{M}_N$. Then, for all $w = (w_1, \dots, w_d) \in \mathcal{H}^d$:
> $$
> T(A, w) = (T_w \otimes I)\,w + (T_x \otimes I)\,x,
> $$
> where $T_w \in \mathbb{R}^{d \times d}$, $T_x \in \mathbb{R}^{d \times N}$, and $x = (x_1, \dots, x_N) \in \mathcal{H}^N$ is given by
> $$
> x_i = J_{\sigma_i A_i}\!\left(\sum_{h=1}^{i-1} l_{ih}\,x_h + \sum_{j=1}^{N-1} b_{ij}\,w_j\right),
> $$
> where $\sigma_i > 0$ for all $i \in \{1, \dots, N\}$, $(l_{ih})$ are the components of a (strictly) lower triangular matrix $L \in \mathbb{R}^{N \times N}$ and $(b_{ij}) = B \in \mathbb{R}^{N \times d}$.

### 2.2 State and bilevel graphs

A **directed graph** is a pair $G = (\mathcal{N}, \mathcal{E})$, where $\mathcal{N}$ is a finite set and $\mathcal{E}$ a subset of $\mathcal{N} \times \mathcal{N}$. The elements of $\mathcal{N}$ are the **nodes** of the graph, the elements of $\mathcal{E}$ its **edges**. Two nodes $i$ and $j$ are **adjacent** if $(i, j) \in \mathcal{E}$ or $(j, i) \in \mathcal{E}$. We denote the set of adjacent nodes to $i$ in $G$ by $\operatorname{adj}(i; G)$. The **degree** of a node $i$ is the cardinality of $\operatorname{adj}(i; G)$ and we often denote it by $d_i$. A **path** between $i_0$ and $i_n$ is a sequence of distinct nodes $(i_0, i_1, \dots, i_n)$ with $i_k$ and $i_{k+1}$ adjacent for $k = 0, \dots, n - 1$. Two nodes of $\mathcal{N}$ are **connected** if there exists at least one path that has the two nodes as its end points. A graph is **connected** if every pair of nodes is connected.

For a directed graph $G = (\mathcal{N}, \mathcal{E})$ an **ordering** of $\mathcal{N}$ is a bijection $\alpha : \{1, \dots, N\} \to \mathcal{N}$. The triple $(\mathcal{N}, \mathcal{E}, \alpha)$ is sometimes referred to as **ordered graph**. In the remainder of this paper, we will refer to the couple $G = (\mathcal{N}, \mathcal{E})$ as a directed ordered graph via the identification $\mathcal{N} = \{1, \dots, N\}$. A **topological ordering** of $G$ is an ordering such that if $(i, j) \in \mathcal{E}$, then $i < j$. For an ordered directed graph, the **in-degree** (resp. the **out-degree**) of $i$ is the cardinality of $\operatorname{adj}(i, G) \cap \{h \mid h < i\}$ (resp. the cardinality of $\operatorname{adj}(i, G) \cap \{h \mid i < h\}$) and is denoted by $d_i^{+}$ (resp. $d_i^{-}$).

Recall from Lemma 2.11 that each FRS can be characterized by means of four matrices $T_w$, $T_x$, $L$, $B$ and a vector $\sigma = (\sigma_1, \dots, \sigma_N)$, where $L$ is strictly lower triangular, i.e., with zero diagonal and upper triangular part. The structure of $L$ imposes a topological order on the evaluations of the resolvents of $A_1, \dots, A_N$. We can therefore associate to each FRS a directed graph with a topological ordering.

> **Definition 2.12 (State graph of a FRS).** Let $(T, S)$ be a FRS for $\mathcal{M}_N$ and let $L$ be the triangular matrix given by Lemma 2.11. The **state graph** associated with $(T, S)$ is an ordered directed graph $G = (\mathcal{N}, \mathcal{E})$ with $\mathcal{N} = \{1, \dots, N\}$ and $(i, j) \in \mathcal{E}$ if and only if $l_{ij} \neq 0$. A FRS method has state graph $G$ if the corresponding FRS $(T, S)$ has state graph $G$.

Our construction of the graph extension of the DRS method requires endowing the notion of state graph with an additional layer.

> **Definition 2.13 (Bilevel graphs).** Let $N \in \mathbb{N}$. A **bilevel graph** is a triple $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ where $G = (\mathcal{N}, \mathcal{E})$ is a connected directed graph with a topological ordering $\mathcal{N} = \{1, \dots, N\}$ and $G' = (\mathcal{N}, \mathcal{E}')$ is a directed connected subgraph. We call $G$ the **state graph** and $G'$ the **base graph**.

In Section 3, we show that for any bilevel graph $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ there exists an unconditionally stable FRS method for $\mathcal{M}_N$ with a minimal lifting and state graph $G = (\mathcal{N}, \mathcal{E})$. The choice of the base graph further discriminates the resulting schemes and is related to the operators $B$ and $T_x$ from Lemma 2.11.

---

## 3. Graph-based Douglas–Rachford

In our construction, the Laplacian of the base graph plays a key role.

> **Definition 3.1 (Graph Laplacian).** Given a directed graph $G = (\mathcal{N}, \mathcal{E})$, with $\mathcal{N} = \{1, \dots, N\}$, the **graph Laplacian** of $G$ is the matrix $L = (L_{ij})_{ij} \in \mathbb{R}^{N \times N}$ defined by
> $$
> L_{ij} := \begin{cases} d_i & \text{if } i = j, \\ -1 & \text{if } i \text{ and } j \text{ are adjacent}, \\ 0 & \text{else}, \end{cases}
> $$
> where $d_1, \dots, d_N \in \mathbb{R}$ are the degrees of the nodes $1, \dots, N$, respectively.

> **Lemma 3.2.** Let $N > 1$. For each connected graph $G = (\mathcal{N}, \mathcal{E})$ with $\mathcal{N} = \{1, \dots, N\}$ there exists $z_1, \dots, z_N \in \mathbb{R}^{N-1}$ such that
> - **(a)** $z_i \cdot z_j \neq 0$ if and only if $i$ and $j$ are adjacent;
> - **(b)** It holds $z_1 + \cdots + z_N = 0$;
> - **(c)** $\operatorname{span}\{z_1, \dots, z_N\} = \mathbb{R}^{N-1}$.

**Proof.** Let $L = (L_{ij})_{ij} \in \mathbb{R}^{N \times N}$ be the Laplacian of the graph $G$. The matrix $L$ has the following properties: $L$ is symmetric and positive semidefinite; $L$, since $G$ is connected, has rank $N - 1$; for $1 \leq i \leq j \leq N$, $(i, j) \in \mathcal{E}$ if and only if $L_{ij} \neq 0$, and $\mathbf{1} \in \ker L$, where $\mathbf{1} = (1, \dots, 1)^{*}$, see, e.g., [13]. Let $L = ZZ^{*}$ be an onto decomposition of $L$, where $Z \in \mathbb{R}^{N \times (N-1)}$ and set $z_1, \dots, z_N \in \mathbb{R}^{N-1}$ to be the rows of $Z$. Using that $\ker Z^{*} = \ker L$ and the fact that $z_i \cdot z_j = L_{ij}$ it is clear that $z_1, \dots, z_N$ satisfy (a), (b) and (c). $\quad\blacksquare$

A collection $z_1, \dots, z_N \in \mathbb{R}^{N-1}$ that satisfies (a) and (c) in Lemma 3.2 is often called a **faithful orthogonal representation** of $G$, see [14]. Here, we seek a faithful orthogonal representation that also sums up to zero.

> **Remark 3.3 (Gossip matrices).** The matrix $Z \in \mathbb{R}^{N \times (N-1)}$ introduced in the proof of Lemma 3.2, with rows $z_i$, has full rank and $\ker Z^{*} = \operatorname{span}\{\mathbf{1}\}$. In practice, $Z$ can be obtained deriving an onto decomposition of the Laplacian matrix of $G$, which can be done via spectral decomposition. If $G$ is a tree, one can take the incidence matrix (cf., Section 4.1). Note that, in general, one could replace the Laplacian with any symmetric positive semidefinite matrix $W = (W_{ij})_{ij}$ of rank $N - 1$ such that for $1 \leq i \leq j \leq N$, $(i, j) \in \mathcal{E}$ if and only if $W_{ij} \neq 0$, and $\mathbf{1} \in \ker W$, where $\mathbf{1} = (1, \dots, 1)^{*}$. This class of operators is often called **Gossip matrices** [15]. In the remainder of this paper, we stick to the Laplacian for simplicity.

Let $G$ be a state graph for the $N$-operator problem (2). In order to find an unconditionally stable FRS method with minimal (i.e., $N - 1$) variables and associated state graph $G$, we stick to the following methodology. We find a maximal monotone operator $\mathcal{A} : \mathbf{H} \to 2^{\mathbf{H}}$ on the Hilbert space $\mathbf{H} := \mathcal{H}^{2N-1}$ such that if $0 \in \mathcal{A}u$ then the first $N$ components of $u$ are equal and solve (2) and, conversely, if $x \in \mathcal{H}$ solves (2) there exists $u \in \mathbf{H}$ such that $0 \in \mathcal{A}u$ and the first $N$ components of $u$ all coincide with $x$. Then, we design an admissible preconditioner $M : \mathbf{H} \to \mathbf{H}$ for $\mathcal{A}$ that admits an onto decomposition with $\mathcal{D} := \mathcal{H}^{N-1}$. In this way, the corresponding reduced PPP method according to (7) would need to store exactly $N - 1$ variables. Imposing certain structural properties on $\mathcal{A}$ and $M$, the method will also meet the desired structure.

**Building $M = CC^{*}$.** Consider the base graph $G'$. Let $Z = (Z_{ij})_{ij} \in \mathbb{R}^{N \times (N-1)}$ be a matrix whose rows are the vectors given by Lemma 3.2 applied to $G'$. Recall that, in particular, we can choose $Z$ such that $L = ZZ^{*}$ where $L$ is the graph Laplacian of $G'$. Let $C$ be the following operator

$$
C^{*} = \begin{pmatrix} \mathbf{Z}^{*} & \mathbf{I} \end{pmatrix}, \tag{10}
$$

where $\mathbf{Z}^{*} = Z^{*} \otimes I$, $\mathbf{I}$ is the identity map on $\mathcal{D} = \mathcal{H}^{N-1}$, and $I$ is the identity map on $\mathcal{H}$. Once the operator $C : \mathcal{D} \to \mathbf{H}$ is fixed, the preconditioner can be obtained as $M = CC^{*}$, which yields

$$
M = \begin{pmatrix} \mathbf{L} & \mathbf{Z} \\ \mathbf{Z}^{*} & \mathbf{I} \end{pmatrix}, \tag{11}
$$

where $\mathbf{L} = \mathbf{Z}\mathbf{Z}^{*} = L \otimes I$. Note that, since $C^{*}$ is onto, the factorization $M = CC^{*}$ is an onto decomposition of $M$ with $C : \mathcal{D} \to \mathbf{H}$.

**Building $\mathcal{A}$.** Let $A \in \mathcal{M}_N$. We need to find a suitable maximal monotone operator $\mathcal{A}$ on $\mathbf{H}$ such that the reduced PPP method with respect to $\mathcal{A}$ and $M = CC^{*}$ defined in (11) meets the desired structure. We first define

$$
\Sigma = \begin{pmatrix} 0 & -L_{12} & \cdots & -L_{1N} \\ L_{21} & 0 & & \vdots \\ \vdots & & \ddots & -L_{N-1,N} \\ L_{N1} & \cdots & L_{N,N-1} & 0 \end{pmatrix}
$$

where $L_{ij}$ are the components of the graph Laplacian of $G'$. We denote by $\mathbf{A}$ the diagonal operator $\mathbf{A} : (x_1, \dots, x_N) \mapsto (A_1 x_1, \dots, A_N x_N)$. Then, we set $B_L := \mathbf{A} + \mathbf{\Sigma}$, where $\mathbf{\Sigma} = \Sigma \otimes I$. Now, consider the difference $\mathcal{E} \setminus \mathcal{E}' := \{(i, j) \in \mathcal{E} \mid (i, j) \notin \mathcal{E}'\}$ and let $\mathbf{P} : \mathcal{H}^N \to \mathcal{H}^N$ be the operator defined as $\mathbf{P} = \sum_{(i,j) \in \mathcal{E} \setminus \mathcal{E}'} \mathbf{P}^{ij}$ where, for each $(i, j) \in \mathcal{E} \setminus \mathcal{E}'$, the operator $\mathbf{P}^{ij}$ is given by $\mathbf{P}^{ij} := P^{ij} \otimes I$ with $P^{ij} \in \mathbb{R}^{N \times N}$ defined by:

$$
(P^{ij})_{hk} := \begin{cases} 1 & \text{if } h = i \text{ and } k = i, \\ 1 & \text{if } h = j \text{ and } k = j, \\ -2 & \text{if } h = j \text{ and } k = i, \\ 0 & \text{else}. \end{cases}
$$

Eventually, we define $A_L := B_L + \mathbf{P}$ and build the operator $\mathcal{A}$ on $\mathbf{H}$ assembling $A_L$ and $\mathbf{Z}$ in the following block structure

$$
\mathcal{A} := \begin{pmatrix} A_L & -\mathbf{Z} \\ \mathbf{Z}^{*} & 0 \end{pmatrix}, \tag{12}
$$

where $0$ is the zero operator on $\mathcal{D}$.

> **Theorem 3.4.** Let $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ be a bilevel graph for the $N$-operator problem (2) with respect to $A \in \mathcal{M}_N$ with $\operatorname{zer}(A_1 + \cdots + A_N) \neq \emptyset$. Let $\mathcal{A} : \mathbf{H} \to 2^{\mathbf{H}}$ be the operator defined in (12). Then, $\mathcal{A}$ is maximal monotone, and for $(x_1, \dots, x_N) \in \mathcal{H}^N$, there exists $(v_1, \dots, v_{N-1}) \in \mathcal{D}$ such that $u = (x_1, \dots, x_N, v_1, \dots, v_{N-1}) \in \mathbf{H}$ satisfies $0 \in \mathcal{A}u$ if and only if $x = x_1 = \cdots = x_N \in \mathcal{H}$ solve (2).

**Proof.** First, we suppose that $u = (x_1, \dots, x_N, v_1, \dots, v_{N-1})$ is such that $0 \in \mathcal{A}u$. Let us denote $x = (x_1, \dots, x_N)$ and $v = (v_1, \dots, v_{N-1})$. By construction, we have that $\mathbf{Z}^{*}x = 0$, which implies $x_1 = \cdots = x_N = x$ (by definition and Lemma 3.2(b)). Now, the first block-row of (12) yields

$$
0 \in A_L x - \mathbf{Z}v = B_L x + \mathbf{P}x - \mathbf{Z}v = \mathbf{A}x + \mathbf{\Sigma}x + \mathbf{P}x - \mathbf{Z}v.
$$

Thus, there exists $a = (a_1, \dots, a_N)$ with $a_i \in A_i x$ for all $i \in \{1, \dots, N\}$ such that

$$
0 = a + \mathbf{\Sigma}x + \mathbf{P}x - \mathbf{Z}v. \tag{13}
$$

Note that $x = (\mathbf{1} \otimes I)x$ and, thus, applying $(\mathbf{1} \otimes I)^{*} = (\mathbf{1}^{*} \otimes I)$ to $\mathbf{\Sigma}x$, as $\Sigma$ is skew-symmetric, yields $0 \in \mathcal{H}$. Recall that $\mathbf{P} = P \otimes I$, with $P = \sum_{(i,j) \in \mathcal{E} \setminus \mathcal{E}'} P^{ij}$. It is clear that $\mathbf{1}^{*}P^{ij}\mathbf{1} = 0$ for each $(i, j) \in \mathcal{E} \setminus \mathcal{E}'$, hence $(\mathbf{1} \otimes I)^{*}\mathbf{P}x = 0 \in \mathcal{H}$. Eventually, also $(\mathbf{1} \otimes I)^{*}\mathbf{Z}v = (\mathbf{1}^{*}Z \otimes I)v = 0$, as $\ker Z^{*} = \operatorname{span}\{\mathbf{1}\}$. In summary, applying $(\mathbf{1} \otimes I)^{*}$ to (13) we get

$$
0 = (\mathbf{1} \otimes I)^{*}a = \sum_{i=1}^{N} a_i \in \sum_{i=1}^{N} A_i x.
$$

On the other hand, if we have a solution $x$ of (2), i.e., there exist $a_i \in A_i x$ for all $i \in \{1, \dots, N\}$ such that $\sum_{i=1}^{N} a_i = 0$, we define $x = (x_1, \dots, x_N)$ with $x_1 = \cdots = x_N = x$ and $a = (a_1, \dots, a_N)$. In this way, $\mathbf{Z}^{*}x = 0$. To conclude, we only need to find $v = (v_1, \dots, v_{N-1})$ such that (13) holds. Such an element can be found as a solution of the linear system $\mathbf{Z}v = a + \mathbf{\Sigma}x + \mathbf{P}x$, which always exists. Indeed, since $x = (\mathbf{1} \otimes I)x$, $\mathbf{\Sigma} = \Sigma \otimes I$ and $\mathbf{P} = P \otimes I$ with $\mathbf{1}^{*}\Sigma\mathbf{1} = \mathbf{1}^{*}P\mathbf{1} = 0$ (because $\Sigma$ is skew-symmetric and $P = \sum_{(i,j) \in \mathcal{E} \setminus \mathcal{E}'} P^{ij}$ with $\mathbf{1}^{*}P^{ij}\mathbf{1} = 0$ for all $(i, j) \in \mathcal{E} \setminus \mathcal{E}'$), we have $(\mathbf{1} \otimes I)^{*}(a + (\mathbf{\Sigma} + \mathbf{P})x) = \sum_{i=1}^{N} a_i = 0$, i.e., the right-hand side obeys $a + \mathbf{\Sigma}x + \mathbf{P}x \in (\ker Z^{*})^{\perp} = \operatorname{Im}Z$.

For the maximal monotonicity of $\mathcal{A}$, let us first note that since $\operatorname{zer}(A_1 + \cdots + A_N) \neq \emptyset$, $\operatorname{dom}\mathcal{A} \neq \emptyset$. Recall from (12) that we have

$$
\mathcal{A} := \begin{pmatrix} B_L & -\mathbf{Z} \\ \mathbf{Z}^{*} & 0 \end{pmatrix} + \begin{pmatrix} \mathbf{P} & 0 \\ 0 & 0 \end{pmatrix}, \tag{14}
$$

where the zeros may be different but are denoted the same. In (14), $B_L$ is maximal monotone being the sum of a maximal monotone operator $\mathbf{A}$ and a skew-symmetric linear map $\mathbf{\Sigma}$, see [16, Corollary 24.4]. The same reasoning applies to the first term in (14). Regarding the second term in (14), we only need to show that $\mathbf{P}$ is monotone. Indeed, monotone linear maps are also maximal [16, Example 20.15]. Recall that $\mathbf{P} = \sum_{(i,j) \in \mathcal{E} \setminus \mathcal{E}'} \mathbf{P}^{ij}$ with $\mathbf{P}^{ij} = P^{ij} \otimes I$. The claim follows from the fact that the operator $\mathbf{P}^{ij}$ is monotone for all $(i, j) \in \mathcal{E} \setminus \mathcal{E}'$, indeed, we can easily see that

$$
\langle \mathbf{P}^{ij}\xi, \xi\rangle = |\xi_i - \xi_j|^2 \quad \text{for all } \xi = (\xi_1, \dots, \xi_N) \in \mathcal{H}^N.
$$

The maximality of $\mathcal{A}$ follows for instance from [16, Corollary 24.4]. $\quad\blacksquare$

**General iterations.** We can derive a closed-form expression for the reduced PPP method derived with respect to the maximal monotone operator $\mathcal{A}$ and the preconditioner $M = CC^{*}$ given by (12) and (11), respectively. Indeed, $M + \mathcal{A}$ has a lower triangular structure and, thus, one can easily derive the PPP iteration according to (1) with $\theta_k = 1$ for all $k \in \mathbb{N}$, which reads

$$
\begin{cases}
x^{k+1} = (\mathbf{L} + A_L)^{-1}\,\mathbf{Z}(\mathbf{Z}^{*}x^k + v^k), \\
v^{k+1} = \mathbf{Z}^{*}x^k + v^k - 2\mathbf{Z}^{*}x^{k+1},
\end{cases}
$$

with $u^k = (x^k, v^k) \in \mathbf{H}$, $x^k \in \mathcal{H}^N$ and $v^k \in \mathcal{D}$. The onto decomposition $M = CC^{*}$ yields a reduced algorithm according to (7) with the substitution $w^k = C^{*}u^k = \mathbf{Z}^{*}x^k + v^k$, resulting in

$$
\begin{cases}
x^{k+1} = (\mathbf{L} + A_L)^{-1}\,\mathbf{Z}w^k, \\
w^{k+1} = w^k - \mathbf{Z}^{*}x^{k+1}.
\end{cases} \tag{15}
$$

Recall that, by construction, $w^{k+1} = \widetilde{T}w^k$, with $\widetilde{T} := (I + C^{*} \triangleright \mathcal{A})^{-1}$. Thus, for general relaxation parameters $\theta_k \in (0, 2]$ such that $\sum_k \theta_k(2 - \theta_k) = +\infty$, we would simply have

$$
w^{k+1} = w^k + \theta_k\left(\widetilde{T}w^k - w^k\right) = w^k + \theta_k\left(w^k - \mathbf{Z}^{*}x^{k+1} - w^k\right) = w^k - \theta_k\mathbf{Z}^{*}x^{k+1}, \tag{16}
$$

where, still, $x^{k+1} = (\mathbf{L} + A_L)^{-1}\mathbf{Z}w^k$. Note that (16) consists only in a simple modification to (15), and that, whenever $\theta_k \neq 1$, one should not confuse $x^{k+1}$ with the first $N$ components of $u^{k+1}$ according to (1).

The operator $(\mathbf{L} + A_L)$ has a lower triangular structure and thus, it is easy to invert explicitly. Indeed, for $i \in \{1, \dots, N\}$, we have

$$
\left(\sum_{h=1}^{i-1} 2L_{ih}x_h^{k+1}\right) - \left(\sum_{(h,i) \in \mathcal{E} \setminus \mathcal{E}'} 2x_h^{k+1}\right) + (d_i' + \bar{d}_i)x_i^{k+1} + A_i x_i^{k+1} \ni \sum_{h=1}^{N-1} Z_{ih}w_h^k,
$$

where $d_i'$ is the degree of $i$ in the base graph $G' = (\mathcal{N}, \mathcal{E}')$ and $\bar{d}_i$ the degree of $i$ in the graph $(\mathcal{N}, \mathcal{E} \setminus \mathcal{E}')$. Thus, $\bar{d}_i + d_i' = d_i$, i.e., the degree of $i$ in the state graph $G$. Therefore, using that $L_{ih} = -1$ if and only if $(h, i) \in \mathcal{E}'$, we get

$$
-\left(\sum_{(h,i) \in \mathcal{E}'} 2x_h^{k+1}\right) - \left(\sum_{(h,i) \in \mathcal{E} \setminus \mathcal{E}'} 2x_h^{k+1}\right) + d_i x_i^{k+1} + A_i x_i^{k+1} \ni \sum_{h=1}^{N-1} Z_{ih}w_h^k,
$$

such that we can invert explicitly provided that $A_i$ is maximal monotone for every $i \in \{1, \dots, N\}$. Further, we can insert positive step-sizes $\sigma > 0$ considering for every $i \in \{1, \dots, N\}$ the operator $\sigma A_i$ instead of $A_i$. Eventually, consider a bilevel graph $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$, an onto decomposition $L = ZZ^{*}$ of the graph Laplacian of the base graph $G' = (\mathcal{N}, \mathcal{E}')$, a step size $\sigma > 0$ and relaxation parameters $\theta_k \in (0, 2]$ such that $\sum_k \theta_k(2 - \theta_k) = +\infty$. Denoting by $d_1, \dots, d_N$ the degrees of the nodes in the state graph, we have the following FRS method with minimal variables.

> **Algorithm 1: The graph-based Douglas–Rachford method associated to the bilevel graph $\mathrm{biG}$.**
>
> **Initialize:** $w_1^0, \dots, w_{N-1}^0 \in \mathcal{H}$
> **for** $k = 0, 1, \dots$ **do**
> &nbsp;&nbsp;**for** $i = 1, \dots, N$ **do**
> $$
> x_i^{k+1} = J_{\frac{\sigma}{d_i}A_i}\!\left(\frac{2}{d_i}\sum_{(h,i) \in \mathcal{E}} x_h^{k+1} + \frac{1}{d_i}\sum_{j=1}^{N-1} Z_{ij}w_j^k\right)
> $$
> &nbsp;&nbsp;**for** $j = 1, \dots, N-1$ **do**
> $$
> w_j^{k+1} = w_j^k - \theta_k \sum_{i=1}^{N} Z_{ij}x_i^{k+1}
> $$

Interestingly, the choice of different base graphs leads to different methods, where the difference can be clearly seen in the update formula for the $w$ variables. The base graph will play a crucial role in the application to distributed optimization in Section 4.

Following from the general framework on degenerate PPP algorithms, we can easily establish convergence.

> **Theorem 3.5.** Let $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ be a bilevel graph for (2). Let $w_1^k, \dots, w_{N-1}^k$ and $x_1^k, \dots, x_N^k$ be given by Algorithm 1 with respect to $(A_1, \dots, A_N) \in \mathcal{M}_N$ with $\operatorname{zer}(A_1 + \cdots + A_N) \neq \emptyset$. Then, for all $i \in \{1, \dots, N-1\}$ each variable $w_i^k$ converges weakly to some $w_i^{*}$ such that $x_1^{*}, \dots, x_N^{*} \in \mathcal{H}$ defined by
> $$
> x_i^{*} = J_{\frac{\sigma}{d_i}A_i}\!\left(\frac{2}{d_i}\sum_{(h,i) \in \mathcal{E}} x_h^{*} + \frac{1}{d_i}\sum_{j=1}^{N-1} Z_{ij}w_j^{*}\right)
> $$
> coincide for all $i \in \{1, \dots, N\}$ and solve (2). Moreover, all the sequences $\{x_i^k\}_k$ for $i \in \{1, \dots, N\}$ converge weakly to that solution.

**Proof.** The proof is an application of Theorem 2.5. Indeed, Algorithm 1 is a reduced PPP method with respect to the operators $\mathcal{A}$ and $M$ defined in (12) and (11) respectively and the onto decomposition $M = CC^{*}$ with $C : \mathcal{D} \to \mathbf{H}$ defined in (10). Furthermore, $\mathcal{A}$ is maximal monotone with $\operatorname{zer}\mathcal{A} \neq \emptyset$ from Theorem 3.4 and $\operatorname{Im}M$ is closed by construction. The operator $(M + \mathcal{A})^{-1}$ is a combination of resolvents and simple algebraic operations and is thus Lipschitz. Recall from (15) that, by construction, $x^{k+1} = (\mathbf{L} + A_L)^{-1}\mathbf{Z}w^k$ contains the first $N$ block-components of $Tu^k$, where $\{u^k\}_k$ is the corresponding PPP sequence, and that, since $C^{*}u^k = w^k$ for every $k \in \mathbb{N}$ (cf., (7)), $Tu^k = (M + \mathcal{A})^{-1}Cw^k$ for all $k \in \mathbb{N}$. Thus, from part 2. of Theorem 2.5, we have $x^{k+1} \rightharpoonup x^{*}$, with $(x^{*}, v^{*}) \in \operatorname{zer}\mathcal{A}$ for some $v^{*} \in \mathcal{D}$. Note as well that part 1. of Theorem 2.5 yields that all the sequences $\{w_i^k\}_k$ converge weakly to some elements $w_i^{*}$ for all $i \in \{1, \dots, N-1\}$, and, denoting by $w^{*} = (w_1^{*}, \dots, w_{N-1}^{*})$, we also have $x^{*} = (\mathbf{L} + A_L)^{-1}\mathbf{Z}w^{*}$. The claim follows applying again Theorem 3.4. $\quad\blacksquare$

> **Corollary 3.6.** Let $G = (\mathcal{N}, \mathcal{E})$ be a directed connected graph with a topological ordering on $N$ nodes. Then, there exists an unconditionally stable FRS method with minimal variables and state graph $G$.

**Proof.** Let $\mathcal{E}' \subset \mathcal{E}$ and consider the bilevel graph $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$. The graph-based DRS with bilevel graph $\mathrm{biG}$ and relaxation parameters $\theta_k = \theta \in (0, 2]$ for all $k \in \mathbb{N}$ is a FRS with respect to the fixed-point encoding $(T, S)$, with $T(A, \cdot) = I + \theta(\widetilde{T} - I)$ and $S(A, \cdot) = \pi_i(\mathbf{L} + A_L)^{-1}\mathbf{Z}$, where $\widetilde{T} = (I + C^{*} \triangleright \mathcal{A})^{-1}$ and $\pi_i : \mathcal{H}^N \to \mathcal{H}$ is the projection onto the $i$-th component, for some $i \in \{1, \dots, N\}$. The fixed-point encoding $(T, S)$ is frugal and has state graph $G$ from Algorithm 1, is unconditionally stable from Theorem 3.5 and has minimal variables as $\mathcal{D} = \mathcal{H}^{N-1}$. $\quad\blacksquare$

> **Remark 3.7.** Algorithm 1 does not depend on the onto decomposition of the Laplacian $L$ of the base graph. Given two different onto decompositions of $L$, say $L = ZZ^{*} = \widetilde{Z}\widetilde{Z}^{*}$ with $Z, \widetilde{Z} \in \mathbb{R}^{N \times (N-1)}$, from Proposition 2.2, there exists an orthogonal matrix $O \in \mathbb{R}^{(N-1) \times (N-1)}$ such that $\widetilde{Z} = ZO$. As before, let $\mathbf{Z}, \widetilde{\mathbf{Z}}, \mathbf{O}$ be the corresponding block operators on $\mathcal{H}^N$ and $\{\widetilde{w}^k\}_k, \{\widetilde{x}^k\}_k$ and $\{w^k\}_k, \{x^k\}_k$ be the two sequences given by Algorithm 1 with respect to $\widetilde{\mathbf{Z}}$ and $\mathbf{Z}$, respectively. Note from (15) that
> $$
> \widetilde{x}^{k+1} = (\mathbf{L} + A_L)^{-1}\widetilde{\mathbf{Z}}\widetilde{w}^k = (\mathbf{L} + A_L)^{-1}\mathbf{Z}\mathbf{O}\widetilde{w}^k, \tag{17}
> $$
> and that $\widetilde{w}^{k+1} = \widetilde{w}^k - \theta_k\mathbf{O}^{*}\mathbf{Z}^{*}\widetilde{x}^{k+1}$, thus for every $k \in \mathbb{N}$,
> $$
> \mathbf{O}\widetilde{w}^{k+1} = \mathbf{O}\widetilde{w}^k - \theta_k\mathbf{Z}^{*}\widetilde{x}^{k+1}. \tag{18}
> $$
> From (17) and (18) we get that if $w^0 = \mathbf{O}\widetilde{w}^0$, then for every $k \in \mathbb{N}$, $w^k = \mathbf{O}\widetilde{w}^k$ and $\widetilde{x}^k = x^k$. Thus, the two choices generate the same sequences modulo an orthogonal transformation of the space.

### 3.1 Examples

**Tree base graphs.** Let $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ be a bilevel graph and assume that $G' = (\mathcal{N}, \mathcal{E}')$ defines a **tree**, i.e., $G'$ has no cycles. This case is particularly relevant for our work, because, for trees, an onto decomposition of the Laplacian of the base graph is simply given by the incidence matrix, so that there is no need to factorize the Laplacian numerically.

To define the incidence matrix of $G'$ we first need to order the edges from $1$ to $|\mathcal{E}'|$. Then, we let $Z \in \mathbb{R}^{N \times |\mathcal{E}'|}$ be the matrix such that $Z_{ij} = 1$ if the $j$-th edge leaves $i$, $Z_{ij} = -1$ if the $j$-th edge enters $i$, and $0$ otherwise. It is well known that if $G'$ is a connected tree, then $|\mathcal{E}'| = N - 1$ and thus the incidence matrix is full-rank. Furthermore, $Z$ is such that $ZZ^{*}$ is equal to the graph Laplacian of $G'$, thus, in this way, $L = ZZ^{*}$ is indeed an onto decomposition with $Z \in \mathbb{R}^{N \times (N-1)}$.

Choosing a tree as a base graph, for the sake of notation, we can associate the variables with the edges in the base graph and rename $w_j$ as $w_{(h,i)}$, where $(h, i)$ is the $j$-th edge in $\mathcal{E}'$.

Given a bilevel graph $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$, with a tree base graph $G' = (\mathcal{N}, \mathcal{E}')$, a step-size $\sigma > 0$ and relaxation parameters $\theta_k \in (0, 2]$ such that $\sum_k \theta_k(2 - \theta_k) = +\infty$, denoting by $d_1, \dots, d_N$ the degrees of the nodes in the state graph, Algorithm 1 turns into the FRS method with minimal variables as presented in Algorithm 2.

> **Algorithm 2: Graph-based Douglas–Rachford with a tree base graph.**
>
> **Initialize:** $w_{(h,i)}^0 \in \mathcal{H}$ for all $(h, i) \in \mathcal{E}'$
> **for** $k = 0, 1, \dots$ **do**
> &nbsp;&nbsp;**for** $i = 1, \dots, N$ **do**
> $$
> x_i^{k+1} = J_{\frac{\sigma}{d_i}A_i}\!\left(\frac{2}{d_i}\sum_{(h,i) \in \mathcal{E}} x_h^{k+1} + \frac{1}{d_i}\Big(\sum_{(i,j) \in \mathcal{E}'} w_{(i,j)}^k - \sum_{(h,i) \in \mathcal{E}'} w_{(h,i)}^k\Big)\right) \tag{19}
> $$
> &nbsp;&nbsp;**for** $(h, i) \in \mathcal{E}'$ **do**
> $$
> w_{(h,i)}^{k+1} = w_{(h,i)}^k + \theta_k\left(x_i^{k+1} - x_h^{k+1}\right) \tag{20}
> $$

**Ryu's splitting for $N$ operators.** To generalize the method introduced by Ryu in [8] for the 3-operator problem to the $N$-operator problem, we consider a complete state graph with a star-shaped tree base graph having node $N$ as the root. In this way, the base graph edge set consists of $\mathcal{E}' = \{(i, N) \mid i = 1, \dots, N-1\}$. With this choice and turning back again to the standard indexing of the $w$ variables, we get

$$
\begin{cases}
x_i^{k+1} = J_{\frac{\sigma}{N-1}A_i}\!\left(\dfrac{2}{N-1}\displaystyle\sum_{h=1}^{i-1} x_h^{k+1} + \dfrac{1}{N-1}w_i^k\right) & \text{for } i \in \{1, \dots, N-1\}, \\[2ex]
x_N^{k+1} = J_{\frac{\sigma}{N-1}A_N}\!\left(\dfrac{2}{N-1}\displaystyle\sum_{h=1}^{N-1} x_h^{k+1} - \dfrac{1}{N-1}\displaystyle\sum_{j=1}^{N-1} w_j^k\right), \\[2ex]
w_j^{k+1} = w_j^k + \theta_k\left(x_N^{k+1} - x_j^{k+1}\right) & \text{for } j \in \{1, \dots, N-1\}.
\end{cases}
$$

Note, in fact, that if $N = 3$, after a suitable rescaling of the $w$ variables, one gets the method in [8, Section 4.1]. This is one possible way to correct the attempt in [9, Remark 4.7].

**Malitsky–Tam splitting as a proximal point method.** If we consider a sequential tree base graph, i.e., $\mathcal{E}' := \{(i, i+1) \mid i = 1, \dots, N-1\}$, and the state graph $\mathcal{E} = \mathcal{E}' \cup \{(1, N)\}$, Algorithm 2 turns into the Malitsky–Tam splitting introduced in [9]. We can therefore conclude that both the Ryu and the Malitsky–Tam splitting can be understood as proximal point methods, which answers an open question in the conclusions of [9].

**Three operator splitting with complete base graph.** The proposed graph-based DRS method when applied to the three operator problem encompasses several already known extensions of the DRS method, namely: the sequential extension introduced in [1] (choosing $\mathcal{E} = \mathcal{E}' = \{(1, 2), (2, 3)\}$), two well-known parallel extensions [10, Section 9.1] (choosing $\mathcal{E} = \mathcal{E}' = \{(1, 3), (2, 3)\}$ or $\mathcal{E} = \mathcal{E}' = \{(1, 2), (1, 3)\}$), and the Ryu and the Malitsky–Tam methods (choosing, respectively, $\mathcal{E}' = \{(1, 3), (2, 3)\}$, $\mathcal{E} = \{(1, 3), (2, 3), (1, 2)\}$ and $\mathcal{E}' = \{(1, 2), (2, 3)\}$, $\mathcal{E} = \{(1, 3), (2, 3), (1, 3)\}$). Note in particular that all the aforementioned choices feature tree base graphs. Yet, for the 3-operator problem the proposed framework provides another interesting architecture: the case of a **complete base graph**. Here, as an onto decomposition of the Laplacian of the complete graph we can pick

$$
Z^{*} = \begin{pmatrix} \sqrt{2} & -\sqrt{1/2} & -\sqrt{1/2} \\ 0 & \sqrt{3/2} & -\sqrt{3/2} \end{pmatrix}.
$$

With this choice, considering $\widetilde{w}_1^k = \sqrt{2}\,w_1^k$ and $\widetilde{w}_2^k = \sqrt{2/3}\,w_2^k$ for every $k \in \mathbb{N}$, Algorithm 1 writes

$$
\begin{cases}
x_1^{k+1} = J_{\frac{\sigma}{2}A_1}\!\left(\dfrac{1}{2}\widetilde{w}_1^k\right), \\[1.5ex]
x_2^{k+1} = J_{\frac{\sigma}{2}A_2}\!\left(x_1^{k+1} + \dfrac{3}{4}\widetilde{w}_2^k - \dfrac{1}{4}\widetilde{w}_1^k\right), \\[1.5ex]
x_3^{k+1} = J_{\frac{\sigma}{2}A_3}\!\left(x_1^{k+1} + x_2^{k+1} - \dfrac{3}{4}\widetilde{w}_2^k - \dfrac{1}{4}\widetilde{w}_1^k\right), \\[1.5ex]
\widetilde{w}_1^{k+1} = \widetilde{w}_1^k - \theta_k\left(2x_1^{k+1} - x_2^{k+1} - x_3^{k+1}\right), \qquad \widetilde{w}_2^{k+1} = \widetilde{w}_2^k - \theta_k\left(x_2^{k+1} - x_3^{k+1}\right),
\end{cases}
$$

where $\sigma > 0$ is a positive step-size and $\{\theta_k\}_k$ in $(0, 2]$ are positive relaxation parameters. Note that in Section 5, we will see that considering fully connected base graphs can lead to faster methods.

### 3.2 Further properties

The reduced PPP method, and in particular Algorithm 1, inherits the well known general convergence guarantees of the proximal point algorithm. Indeed, if we assume that the relaxation parameters satisfy $0 < \inf_k \theta_k \leq \sup_k \theta_k < 2$, then a standard result is the asymptotic rate

$$
\|\widetilde{T}w^k - w^k\|^2 = o(k^{-1}) \quad \text{for } k \to +\infty, \tag{21}
$$

see e.g., [6, Theorem 1]. Note that, as we set $\theta_k \in [\varepsilon, 2 - \varepsilon]$ for some $0 < \varepsilon < 1$, then also $\|w^{k+1} - w^k\|^2 = o(k^{-1})$. In our framework, the residual $\|\widetilde{T}w^k - w^k\|^2$ has an elegant connection with an interesting quantity that measures how far the solution estimates $x_i^{k+1}$ are from consensus, namely, the **state variance**, which we define for all $k \in \mathbb{N}$ as

$$
\operatorname{Var}(x^k) := \frac{1}{N}\sum_{i=1}^{N}\|x_i^k - \bar{x}^k\|^2, \qquad \text{for } \bar{x}^k := \frac{1}{N}\sum_{i=1}^{N} x_i^k. \tag{22}
$$

Indeed, we have the following result.

> **Proposition 3.8.** Let $N \geq 2$, let $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ be a bilevel graph for the $N$-operator problem (2) for $(A_1, \dots, A_N) \in \mathcal{M}_N$ with $\operatorname{zer}(A_1 + \cdots + A_N) \neq \emptyset$. Let $x^k = (x_1^k, \dots, x_N^k)$, $w^k = (w_1^k, \dots, w_{N-1}^k)$ be the sequences generated by Algorithm 1 with step-size $\sigma > 0$ and relaxation parameters $\{\theta_k\}_k$ in $(0, 2]$ such that $\sum_{k=0}^{\infty} \theta_k(2 - \theta_k) = +\infty$. Then
> $$
> \operatorname{Var}(x^{k+1}) \leq \frac{1}{\lambda_1 N}\|\widetilde{T}w^k - w^k\|^2 \quad \text{for all } k \in \mathbb{N}, \tag{23}
> $$
> where $\lambda_1$ is the **algebraic connectivity** of the base graph, i.e., the first nonzero eigenvalue of the graph Laplacian.

**Proof.** From $\widetilde{T}w^k = w^k - \mathbf{Z}^{*}x^{k+1}$ for all $k \in \mathbb{N}$, we deduce

$$
\|\mathbf{Z}^{*}x^{k+1}\|^2 = \|\widetilde{T}w^k - w^k\|^2. \tag{24}
$$

Since $\ker \mathbf{Z}^{*} = \{(x, \dots, x) \mid x \in \mathcal{H}\}$, the projection onto the orthogonal complement of $\ker \mathbf{Z}^{*}$ of $x^{k+1}$ is simply $x^{k+1} - \bar{x}^{k+1}$ where $\bar{x}^{k+1} = (\bar{x}^{k+1}, \dots, \bar{x}^{k+1}) \in \mathcal{H}^N$, which, since $\mathbf{L} = \mathbf{Z}\mathbf{Z}^{*}$, gives

$$
\lambda_1\|x^{k+1} - \bar{x}^{k+1}\|^2 \leq \langle \mathbf{L}(x^{k+1} - \bar{x}^{k+1}), x^{k+1} - \bar{x}^{k+1}\rangle = \|\mathbf{Z}^{*}(x^{k+1} - \bar{x}^{k+1})\|^2 = \|\mathbf{Z}^{*}x^{k+1}\|^2. \tag{25}
$$

The identity (24) together with (25) yields (23). $\quad\blacksquare$

Note that Proposition 3.8 shows an interesting dependence on the algebraic connectivity of the base graph, which we shall investigate better in the experiments (cf., Section 5.1). Additionally, from (25) we can already conclude that the state variance converges to zero with a $o(k^{-1})$ worst-case rate.

We now address the converse question, that is, whether the state variance can be understood as a measure of convergence for Algorithm 1. Here, another feature of the underlying bilevel graph turns out to be particularly relevant, namely, the **unbalance** of the state graph:

$$
U_G := \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(d_i^{-} - d_i^{+}\right)^2}, \tag{26}
$$

where, for all $i \in \{1, \dots, N\}$, $d_i^{+}$ and $d_i^{-}$ denote the out-degree and the in-degree of node $i$ in $G$ respectively. We observe, first, that from the maximality of $A_1, \dots, A_N$ in (2), Algorithm 1 uniquely defines $N$ sequences $\{a_i^{k+1}\}_k$, with $a_i^{k+1} \in A_i x_i^{k+1}$ for $i \in \{1, \dots, N\}$ and all $k \in \mathbb{N}$. By construction, $a^{k+1} := (a_1^{k+1}, \dots, a_N^{k+1}) \in \mathcal{H}^N$ is characterized as the unique element in $\mathbf{A}x^{k+1}$ that solves

$$
\mathbf{L}x^{k+1} + (\mathbf{\Sigma} + \mathbf{P})x^{k+1} + \sigma a^{k+1} = \mathbf{Z}w^k \quad \text{for all } k \in \mathbb{N}. \tag{27}
$$

In the following result, we show that the state variance provides an upper bound for the norm of $\sum_{i=1}^{N} a_i^{k+1}$.

> **Proposition 3.9.** Let $N \geq 2$, let $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ be a bilevel graph for the $N$-operator problem (2) for $A = (A_1, \dots, A_N) \in \mathcal{M}_N$ with $\operatorname{zer}(A_1 + \cdots + A_N) \neq \emptyset$. Let $w^k = (w_1^k, \dots, w_{N-1}^k)$, $x^{k+1} = (x_1^{k+1}, \dots, x_N^{k+1})$ and $a^k = (a_1^k, \dots, a_N^k) \in \mathbf{A}x^k$ be the sequences generated by Algorithm 1 with step-size $\sigma > 0$ and relaxation parameters $\{\theta_k\}_k$ in $(0, 2]$ such that $\sum_{k=0}^{\infty} \theta_k(2 - \theta_k) = +\infty$. Then,
> $$
> \left\|\sum_{i=1}^{N} a_i^k\right\|^2 \leq \frac{U_G^2 N^2}{\sigma^2}\operatorname{Var}(x^k) \quad \text{for all } k \in \mathbb{N}. \tag{28}
> $$
> In particular, if $x_1^k = \cdots = x_N^k =: x^{*}$ for some $k \in \mathbb{N}$, then $x^{*}$ is a solution to (2).

**Proof.** First, notice that by definition of $\Sigma$ and $P$, for all $j \in \{1, \dots, N\}$, we have

$$
\sum_{i=1}^{N}(\Sigma + P)_{ij} = \sum_{i > j} L_{ij} - \sum_{i < j} L_{ij} + \sum_{i \geq j}\sum_{(h,k) \in \mathcal{E} \setminus \mathcal{E}'} P_{ij}^{hk}
$$
$$
= \Big(\sum_{(j,i) \in \mathcal{E}'}(-1)\Big) + \Big(\sum_{(i,j) \in \mathcal{E}'} 1\Big) + \bar{d}_j + \sum_{(j,h) \in \mathcal{E} \setminus \mathcal{E}'}(-2) = (d_j')^{+} - (d_j')^{-} + (\bar{d}_j)^{+} - (\bar{d}_j)^{-} = d_j^{+} - d_j^{-},
$$

where $(d_j')^{+}, (d_j')^{-}, (\bar{d}_j)^{+}, (\bar{d}_j)^{-}, \bar{d}_j, d_j^{+}, d_j^{-}$ are respectively: the in- and out-degrees of node $j$ in $G'$, the in- and out-degrees of node $j$ in $(\mathcal{N}, \mathcal{E} \setminus \mathcal{E}')$, the degree of node $j$ in $(\mathcal{N}, \mathcal{E} \setminus \mathcal{E}')$, and the in- and out-degrees of node $j$ in $G$. Therefore, we have

$$
\|(\mathbf{1} \otimes I)^{*}(\mathbf{\Sigma} + \mathbf{P})\| = \sqrt{N}\,U_G. \tag{29}
$$

Since $\{(x, \dots, x) \mid x \in \mathcal{H}\} \subset \ker(\mathbf{1} \otimes I)^{*}(\mathbf{\Sigma} + \mathbf{P})$, using (29), we get

$$
\|(\mathbf{1} \otimes I)^{*}(\mathbf{\Sigma} + \mathbf{P})x^k\| = \|(\mathbf{1} \otimes I)^{*}(\mathbf{\Sigma} + \mathbf{P})(x^k - \bar{x}^k)\| \leq \|(\mathbf{1} \otimes I)^{*}(\mathbf{\Sigma} + \mathbf{P})\|\,\|x^k - \bar{x}^k\| = U_G\sqrt{N}\,\|x^k - \bar{x}^k\|.
$$

From (27), since $\mathbf{1}^{*}Z = 0$ by construction, $(\mathbf{1} \otimes I)^{*}\mathbf{Z}w = ((\mathbf{1}^{*}Z) \otimes I)w = 0$ for every $w \in \mathcal{D}$, hence it easily follows that $\sigma\sum_{i=1}^{N} a_i^k = -(\mathbf{1} \otimes I)^{*}(\mathbf{\Sigma} + \mathbf{P})x^k$, and thus (28). The rest of the proof is straightforward. $\quad\blacksquare$

Proposition 3.9, apart from answering the natural question on whether consensus is reached only at a solution point, which is trivial to prove but perhaps not clear from Algorithm 1, also yields that if the state variance is small, i.e., all the solution estimates $x_1^k, \dots, x_N^k$ are close to consensus, then $\sum_{i=1}^{N} a_i^k$ is close to zero as well. Thus, without further structure, the state variance can actually be considered as a residual for Algorithm 1.

Gathering Propositions 3.8 and 3.9 with (21), we can easily conclude that

$$
\frac{\sigma^2}{N^2}\left\|\sum_{i=1}^{N} a_i^{k+1}\right\|^2 \leq U_G^2 \operatorname{Var}(x^{k+1}) \leq \frac{U_G^2}{\lambda_1 N}\|\widetilde{T}w^k - w^k\|^2 = o(k^{-1}) \quad \text{for } k \to +\infty. \tag{30}
$$

Thus, also $\sum_{i=1}^{N} a_i^{k+1}$ converges to zero strongly as $k \to +\infty$ with a $o(k^{-1/2})$ rate.

We conclude this section with a discussion on further results assuming additional hypotheses on the operators $A_1, \dots, A_N$. First, we assume that $A_i = \partial f_i$ for some convex, proper and lower semicontinuous functions $f_1, \dots, f_N : \mathcal{H} \to \mathbb{R} \cup \{+\infty\}$, so that (2) turns into

$$
\text{find } x \in \mathcal{H} \text{ such that: } \quad 0 \in (\partial f_1 + \cdots + \partial f_N)\,x. \tag{31}
$$

Note that (31) is equivalent to minimizing $f := f_1 + \cdots + f_N$ under mild regularity conditions, see, e.g., [16, Corollary 16.38], which we shall implicitly assume, and, in that case, the condition $\operatorname{zer}(A_1 + \cdots + A_N) = \operatorname{zer}(\partial f_1 + \cdots + \partial f_N) \neq \emptyset$ is equivalent to the existence of minimizers of $f$.

> **Proposition 3.10 (Rate on the objective function).** Let $N \geq 2$, let $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ be a bilevel graph for (31) and assume that $\operatorname{zer}(\partial f_1 + \cdots + \partial f_N) \neq \emptyset$. Let $x^k = (x_1^k, \dots, x_N^k)$, $w^k = (w_1^k, \dots, w_N^k)$ be the sequences generated by Algorithm 1 with step-size $\sigma > 0$ and relaxation parameters $\{\theta_k\}_k$ in $[\varepsilon, 2 - \varepsilon]$ for some $0 < \varepsilon < 1$. Let $j \in \{1, \dots, N\}$ and suppose $f_i$ is locally Lipschitz continuous for all $i \neq j$, then we have
> $$
> f(x_j^k) - \inf f = o(k^{-1/2}). \tag{32}
> $$
> Furthermore, if also $f_j$ is locally Lipschitz then $f(\bar{x}^k) - \inf f = o(k^{-1/2})$.

**Proof.** Let $x^{*}$ be a solution to (2). From convexity of $f_i$ we have for all $i \in \{1, \dots, N\}$ that

$$
f_i(x_i^k) - f_i(x^{*}) \leq \langle a_i^k, x_i^k - x^{*}\rangle, \tag{33}
$$

where $a_i^k \in \partial f_i(x_i^k)$ are defined by (27). Summing up for all $i \in \{1, \dots, N\}$ and using the local Lipschitz property of $f_i$ for all $i \neq j$, we get

$$
f(x_j^k) - f(x^{*}) = \sum_{i=1}^{N}\left(f_i(x_i^k) - f_i(x^{*}) + f_i(x_j^k) - f_i(x_i^k)\right) \leq \sum_{i=1}^{N}\langle a_i^k, x_i^k - x^{*}\rangle + \sum_{i \neq j} L_i\|x_j^k - x_i^k\|, \tag{34}
$$

where $L_i$ are the Lipschitz constants of $f_i$ for all $i \neq j$ on some ball containing $x_1^k, \dots, x_N^k$, for all $k \in \mathbb{N}$, which exists because $\{x^k\}_k$ is a bounded sequence. Using (30) three times together with the fact that $\{x_j^k - x^{*}\}_k$ and $\{a_i^k\}_k$ for all $i \neq j$ are bounded sequences (cf., (27)), we have

$$
\sum_{i=1}^{N}\langle a_i^k, x_i^k - x^{*}\rangle = \left\langle \sum_{i=1}^{N} a_i^k, x_j^k - x^{*}\right\rangle + \sum_{i \neq j}\langle a_i^k, x_i^k - x_j^k\rangle
$$
$$
\leq \left\|\sum_{i=1}^{N} a_i^k\right\|\,\|x_j^k - x^{*}\| + \sum_{i \neq j}\|a_i^k\|\,\|x_i^k - x_j^k\| = o(k^{-1/2}). \tag{35}
$$

Combining (35) with (34) and again (30) we get (32), since a solution of (31) is also a zero for $\partial f$ and thus, a minimizer for $f$. The variance estimate in (30) provides also the rate for $f(\bar{x}^k) - \inf f$ when $f_j$ is locally Lipschitz as well. $\quad\blacksquare$

Proposition 3.10 is in line with similar results on DRS and the more general forward–DRS scheme (see, e.g., [17, Theorem 3.4, Corollary 3.5]). The next result shows that the convergence of the solution estimates is strong if at least one operator is uniformly monotone (cf., [16, Definition 22.1(iii)]).

> **Proposition 3.11 (Strong convergence).** Let $N \geq 2$, let $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ be a bilevel graph for the $N$-operator problem (2) for $A = (A_1, \dots, A_N) \in \mathcal{M}_N$ with $\operatorname{zer}(A_1 + \cdots + A_N) \neq \emptyset$. Assume that there exists $j \in \{1, \dots, N\}$ such that $A_j$ is uniformly monotone on every bounded set of $\operatorname{dom}A_j$. Then, for all $i \in \{1, \dots, N\}$ the sequences $\{x_i^k\}_k$ generated by Algorithm 1, with step-size $\sigma > 0$ and relaxation parameters $\{\theta_k\}_k$ in $[\varepsilon, 2 - \varepsilon]$ for some $0 < \varepsilon < 1$, converge strongly to a solution to (2).

**Proof.** Let $\{u^k\}_k$ be the corresponding PPP sequence according to (1) and $u^{*} = (x^{*}, v^{*}) \in \operatorname{zer}\mathcal{A}$ be the weak limit of $\{Tu^k\}_k$ (cf., proof of Theorem 3.5), then, by Theorem 3.4, $x^{*} = (\mathbf{1} \otimes I)x^{*}$ for some $x^{*}$ that solves (2). Consider the bounded set $S = \{x^{*}\} \cup \{x_j^k\}_k \subset \operatorname{dom}A_j$. By definition of uniform monotonicity, there exists an increasing function $\phi : \mathbb{R}_{+} \to [0, +\infty]$ that vanishes only at $0$ such that

$$
\langle a_j - a_j', x - x'\rangle \geq \phi(\|x - x'\|) \quad \text{for all } x, x', a_j, a_j' \text{ such that: } a_j \in A_j x,\ a_j' \in A_j x'.
$$

By definition of $T$ we have $M(u^k - Tu^k) \in \mathcal{A}Tu^k$, and consequently, by construction,

$$
M(u^k - Tu^k) = \begin{pmatrix} (\mathbf{\Sigma} + \mathbf{P})x^{k+1} + \sigma a^{k+1} - \mathbf{Z}v^{k+1} \\ \mathbf{Z}^{*}x^{k+1} \end{pmatrix},
$$

where $a^{k+1} \in \mathbf{A}x^{k+1}$ and $(x^{k+1}, v^{k+1}) = Tu^k$. Thus, since $u^{*} \in \operatorname{zer}\mathcal{A}$, there exists $a^{*} \in \mathbf{A}x^{*}$ such that $(\mathbf{\Sigma} + \mathbf{P})x^{*} + \sigma a^{*} - \mathbf{Z}v^{*} = 0$ and $\mathbf{Z}^{*}x^{*} = 0$. Therefore, we have for all $k \in \mathbb{N}$ that

$$
\langle M(u^k - Tu^k), Tu^k - u^{*}\rangle = \langle (\mathbf{\Sigma} + \mathbf{P})(x^{k+1} - x^{*}) + \sigma(a^{k+1} - a^{*}) - \mathbf{Z}(v^{k+1} - v^{*}), x^{k+1} - x^{*}\rangle + \langle \mathbf{Z}^{*}x^{k+1}, v^{k+1} - v^{*}\rangle
$$
$$
= \langle (\mathbf{\Sigma} + \mathbf{P})(x^{k+1} - x^{*}), x^{k+1} - x^{*}\rangle + \langle \sigma(a^{k+1} - a^{*}), x^{k+1} - x^{*}\rangle - \langle v^{k+1} - v^{*}, \mathbf{Z}^{*}x^{k+1}\rangle + \langle \mathbf{Z}^{*}x^{k+1}, v^{k+1} - v^{*}\rangle.
$$

Now, using that $\mathbf{P}$ is monotone by construction (cf., proof of Theorem 3.4) and $\mathbf{\Sigma}$ is skew-symmetric yields $\langle (\mathbf{\Sigma} + \mathbf{P})(x^{k+1} - x^{*}), x^{k+1} - x^{*}\rangle \geq 0$. Therefore, by the uniform monotonicity of $A_j$,

$$
\langle M(u^k - Tu^k), Tu^k - u^{*}\rangle \geq \langle \sigma(a^{k+1} - a^{*}), x^{k+1} - x^{*}\rangle = \sum_{i=1}^{N}\sigma\langle a_i^{k+1} - a_i^{*}, x_i^{k+1} - x_i^{*}\rangle \geq \sigma\phi(\|x_j^{k+1} - x^{*}\|).
$$

Using (21), recalling that $M(u^k - Tu^k) = C(w^k - \widetilde{T}w^k)$ and the fact that $Tu^k \rightharpoonup u^{*}$ weakly in $\mathbf{H}$, the left hand-side vanishes as $k \to +\infty$, thus $x_j^k \to x^{*}$ strongly in $\mathcal{H}$. For all other sequences $\{x_i^k\}_k$, (30) yields in particular $x_i^k - x_j^k \to 0$ for all $i \in \{1, \dots, N\}$. The thesis follows. $\quad\blacksquare$

> **Remark 3.12.** Under the assumptions of Proposition 3.11, assuming further structure on the function $\phi$ given by the definition of uniform monotonicity, we shall also provide a rate for $\|x_j^k - x^{*}\|$. In particular, if there is some $\mu > 0$ such that we can choose $\phi(h) = \mu h^2$ for all $h \in \mathbb{R}_{+}$, then we get $\|x_j^k - x^{*}\|^2 = o(k^{-1/2})$ for $k \to +\infty$, which is in line with previous results on DRS and forward–DRS (see, [17, Theorem 4.1.3]).

---

## 4. Application to distributed optimization

Let $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ be a bilevel graph for the $N$-operator problem (2) and suppose that each node in $\mathcal{N}$ represents an **agent**. Each agent is independent and can do parallel, asynchronous computations. Further, every agent can send and receive data to adjacent nodes, and has to wait for the transfer to finish in case of reception. The agents are supposed to collaborate to solve (2). For each $i \in \mathcal{N}$, we assume that only agent $i$ knows the operator $A_i$, and can access it only through evaluation of its resolvent. Additionally, we assume that for each variable $w_j$, there is only one agent in charge of storing and updating it.

### 4.1 Tree base graphs

The inherent sparsity of the incidence matrix allows us to easily turn Algorithm 2 into a simple fully distributed scheme. Indeed, from Algorithm 2 it is easy to notice that agent $i$ at iteration $k \in \mathbb{N}$, in order to compute (19), would need to receive all the $x_h^{k+1}$ for every $(h, i) \in \mathcal{E}$, which can be transmitted from $h$ to $i$, and all the variables $w_{(i,j)}^k$ and $w_{(h,i)}^k$ for all $(i, j) \in \mathcal{E}'$ and $(h, i) \in \mathcal{E}'$. Here, we can assume that each variable $w_{(h,i)}^k$ is stored and updated by agent $i$ for all $(h, i) \in \mathcal{E}'$. In this way, agent $i$ would only need to receive, in addition to all $x_h^{k+1}$ from each $h$ with $(h, i) \in \mathcal{E}$, all the $w_{(i,j)}^k$ from each $j$ with $(i, j) \in \mathcal{E}'$. Eventually, once agent $i$ computed $x_i^{k+1}$, agent $i$ can also update all the variables $w_{(h,i)}^k$ for all $h$ such that $(h, i) \in \mathcal{E}'$, as this computation only involves $w_{(h,i)}^k$ (already stored), $x_h^{k+1}$ (received from $h$) and $x_i^{k+1}$ (just computed). In summary, given a bilevel graph $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ with a tree base graph $G' = (\mathcal{N}, \mathcal{E}')$, in the notation of Algorithm 2, we get the fully distributed protocol described in Algorithm 3.

> **Algorithm 3: Distributed protocol for the graph-based Douglas–Rachford method with a tree base graph.**
>
> **Initialize:** For each $(h, i) \in \mathcal{E}'$, agent $i$ chooses $w_{(h,i)}^0 \in \mathcal{H}$ and sends $w_{(h,i)}^0$ to each agent $h$ such that $(h, i) \in \mathcal{E}'$
> **for** $k = 0, 1, \dots$ **do**
> &nbsp;&nbsp;**for** $i \in \mathcal{N}$ **do**
> &nbsp;&nbsp;&nbsp;&nbsp;Agent $i$ receives $w_{(i,j)}^k$ from each agent $j$ such that $(i, j) \in \mathcal{E}'$, and $x_h^{k+1}$ from each agent $h$ such that $(h, i) \in \mathcal{E}$
> &nbsp;&nbsp;&nbsp;&nbsp;Agent $i$ computes $x_i^{k+1}$ as in (19), and updates $w_{(h,i)}^{k+1}$ for all $(h, i) \in \mathcal{E}'$ as in (20)
> &nbsp;&nbsp;&nbsp;&nbsp;Eventually, agent $i$ sends $x_i^{k+1}$ to each agent $j$ such that $(i, j) \in \mathcal{E}$, and $w_{(h,i)}^{k+1}$ to each agent $h$ such that $(h, i) \in \mathcal{E}'$

### 4.2 General base graphs

As we will show in the experiments, considering base graphs with high algebraic connectivity could be beneficial in terms of convergence speed. Here, though, the design of a distributed protocol with in some sense minimal information flow through the network can lead to severe graph-theoretical challenges, as one would need to find a sparse onto decomposition of the Laplacian of the base graph, which for large scale instances can be costly or even intractable [18, 19].

For general base graphs, we can avoid such sparse factorization issues at the cost of introducing one additional variable, ending up with $N$ variables instead of $N - 1$. Specifically, considering the change of variables $\widetilde{w}^k = \mathbf{Z}w^k$ for all $k \in \mathbb{N}$, the graph-based Douglas–Rachford iteration in (15) reads for all $k \in \mathbb{N}$ as

$$
\begin{cases}
x^{k+1} = (\mathbf{L} + A_L)^{-1}\widetilde{w}^k, \\
\widetilde{w}^{k+1} = \widetilde{w}^k - \theta_k\mathbf{L}x^{k+1},
\end{cases}
\qquad \widetilde{w}^0 = (\widetilde{w}_1^0, \dots, \widetilde{w}_N^0) \in \operatorname{Im}Z. \tag{36}
$$

In this setting, we assume that the variable $\widetilde{w}_i$ is stored and updated by the agent $i$, for every $i \in \{1, \dots, N\}$. The only additional restriction, here, is that the starting point $\widetilde{w}^0$ should be chosen in the range of $Z$, namely, such that $\widetilde{w}_1^0 + \cdots + \widetilde{w}_N^0 = 0$. To avoid further communications between the agents, we could simply let the agents initialize $\widetilde{w}_i^0 = 0$ for every $i \in \{1, \dots, N\}$. Setting, for all $i \in \mathcal{N}$, $d_i'$ to be the degree of $i$ in $G'$, we get the distributed protocol described in Algorithm 4.

> **Algorithm 4: Distributed protocol for the graph-based Douglas–Rachford method with a general base graph.**
>
> **Initialize:** For each $i \in \mathcal{N}$, agent $i$ chooses $\widetilde{w}_i^0 = 0 \in \mathcal{H}$
> **for** $k = 0, 1, \dots$ **do**
> &nbsp;&nbsp;**for** $i \in \mathcal{N}$ **do**
> &nbsp;&nbsp;&nbsp;&nbsp;Agent $i$ receives $x_h^{k+1}$ from each agent $h$ such that $(h, i) \in \mathcal{E}$
> &nbsp;&nbsp;&nbsp;&nbsp;Agent $i$ computes $x_i^{k+1}$ as
> $$
> x_i^{k+1} = J_{\frac{\sigma}{d_i}A_i}\!\left(\frac{2}{d_i}\sum_{(h,i) \in \mathcal{E}} x_h^{k+1} + \frac{1}{d_i}\widetilde{w}_i\right)
> $$
> &nbsp;&nbsp;&nbsp;&nbsp;Agent $i$ sends $x_i^{k+1}$ to each agent $j$ such that $(i, j) \in \mathcal{E}$ and to each agent $h$ such that $(h, i) \in \mathcal{E}'$
> &nbsp;&nbsp;&nbsp;&nbsp;Agent $i$ receives $x_j^{k+1}$ from all the agents $j$ such that $(i, j) \in \mathcal{E}'$ and, eventually, updates $\widetilde{w}_i^{k+1}$ according to
> $$
> \widetilde{w}_i^{k+1} = \widetilde{w}_i^k - \theta_k\left(d_i'\,x_i^{k+1} - \sum_{j \in \operatorname{adj}(i;G')} x_j^{k+1}\right)
> $$

Note that Algorithm 4, contrarily to Algorithm 2, features two communication phases. The first communication phase is similar to the communication phase in Algorithm 2, but only involves a transmission of the solution estimates. The second communication phase is fundamental, because to update $\widetilde{w}_i^{k+1}$, node $i$ needs: $\widetilde{w}_i^k$ (already stored), $x_i^{k+1}$ (computed before) and all the $x_j^{k+1}$ for all $j \in \operatorname{adj}(i; G')$. The latter can be divided into two classes: $x_j^{k+1}$ with $j \leq i$, which node $i$ received in the first communication phase, and $x_j^{k+1}$ with $j \geq i$, received in the second communication phase. Note, in particular, that in Algorithm 4 only the solution estimates $x_1^k, \dots, x_N^k$ are shared.

---

## 5. Numerical experiments

In this section, we present our numerical implementation of the graph-based DRS and its distributed variant applied to a congested optimal transport problem and to a distributed Support Vector Machine problem. All the experiments are performed in Python on an Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz and 8 Gb of RAM and are available for reproducibility at <https://github.com/TraDE-OPT/graph-DRS>.

### 5.1 Congested transport

The congested transport problem has a rich history that dates back to the works of Beckmann in the 50's [20]. The problem gathered a renewed interest more recently thanks to its connections to the optimal transport theory, mainly due to Santambrogio, Carlier et al. [21]. The problem is of the form

$$
\min_{\sigma \in L^{3/2}(\Omega, \mathbb{R}^2)} \int_{\Omega} \|\sigma(x)\|^{3/2}\,dx + \int_{\Omega} \|\sigma(x)\|\,dx \quad \text{subject to: } \begin{cases} \operatorname{div}\sigma = \nu - \mu & \text{in } \Omega, \\ \sigma \cdot n = 0 & \text{on } \partial\Omega, \end{cases} \tag{37}
$$

where $\Omega$ is a smooth compact domain in $\mathbb{R}^2$, and $\mu, \nu$ are two (sufficiently regular) probability densities. The divergence constraint in (37) has to be understood weakly with Neumann boundary conditions ($n$ is the outward unit vector), see [22, Section 4.4.1] for further details. A feasible $\sigma$ is referred to as **transport flow** and its total variation $|\sigma| : \Omega \to [0, \infty)$, defined for all $x \in \Omega$ as $|\sigma|(x) := \|\sigma(x)\|$, is referred to as **transport density**. To give a rough idea, the integral of the transport density on some region $A \subset \Omega$ can be understood as the total amount of mass that is moving from $\mu$ to $\nu$ passing through $A$, while the flow contains the information on the direction that the mass is taking.

Here, we suppose that the two probability densities $\mu$ and $\nu$ are separated by a region that does not allow any transportation, e.g., a lake, on top of which there is a bridge with limited capacity. Specifically, we introduce in (37) an additional constraint of the form $\big\||\sigma|_{\mathrm{Brg}}\big\|_{\infty} \leq C$ and $\sigma|_{\mathrm{Wtr}} = 0$ for some $C > 0$, where $|\sigma|_{\mathrm{Brg}}$ and $|\sigma|_{\mathrm{Wtr}}$ are the restrictions of $|\sigma|$ on $\mathrm{Brg} \subset \Omega$ (the bridge), and $\mathrm{Wtr} \subset \Omega$ (the lake), respectively ($\mathrm{Wtr} \cap \mathrm{Brg} = \emptyset$). Confer to Figure 2 for an illustration.

We discretize the problem on a square grid of size $n = p \times p$ (that we shall still denote by $\Omega$) using forward finite differences, getting to a problem of the form

$$
\min_{\sigma \in \mathbb{R}^{n \times 2}} \sum_{i=1}^{n} \|\sigma_i\|^{3/2} + \sum_{i=1}^{n} \|\sigma_i\| + I\{\Lambda\sigma = \nu - \mu\} + I\left\{\sigma|_{\mathrm{Wtr}} = 0,\ \|\sigma_i\| \leq C\ \ \forall i \in \mathrm{Brg}\right\} \tag{38}
$$

where $\mathrm{Brg}$ and $\mathrm{Wtr}$ denote the set of indices corresponding to the bridge and the lake, respectively, and $\Lambda$ is the discrete divergence operator with no-flux constraints on the boundary, and $I\{x \in C\}$ denotes, with a slight abuse of notation, the indicator function of the convex set $C$, i.e., $I\{x \in C\} = +\infty$ if $x \notin C$ and $0$ else. The optimization problem (38) has four non-smooth simple terms, i.e., for which one can easily compute the proximity operator and, thus, it is for us a good benchmark problem on which we can investigate all the different graph-based extensions of the DRS method.

**The influence of the connectivity.** In this experiment, we investigate the influence of the algebraic connectivity of the base graph on the convergence performance of Algorithm 1. As $N = 4$, there are exactly $38$ possible state graphs. By enumeration, one can further see that the algebraic connectivity takes exactly four possible values, namely: $\lambda_1 = 2 - \sqrt{2}\ (0.5857\dots)$, $1$, $2$ and $4$.

In our first numerical experiment, we set $p = 35$, and consider bilevel graphs with equal state and base graphs. We pick a step-size $\tau = 2$ and set $C = 5 \cdot 10^{-2}$. For each choice of the base graph $G' = (\mathcal{N}, \mathcal{E}')$, we run Algorithm 1 with step-size $\tau$, bilevel graph $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E}')$ with $\mathcal{E} = \mathcal{E}'$, and starting point $w^0 = 0$, and plot the state variance (22) with respect to the iteration number with a specific color representing the algebraic connectivity of $G'$. The results are shown in Figure 1(a). Then, we repeat the procedure fixing a complete state graph and letting the base graph vary among all possible sub-graphs. The results are shown in Figure 1(b). Recall from Remark 3.7 that Algorithm 1 is in some sense independent of the onto decomposition of the Laplacian of the base graph, thus, we do not compare different factorization choices.

In both cases, we can clearly see an effect of the algebraic connectivity of the base graph on the decrease of the state variance. Such a phenomenon is pretty common in the distributed optimization literature, see for instance [23]. We plan to investigate it better in future works.

> **Figure 1.** *Influence of the algebraic connectivity $\lambda_1$ of the base graph on the decrease of the state variance (22) as a function of the iteration number.* Two semi-log plots of state variance vs. iterations ($0$–$1000$), with curves grouped by the four connectivity values $\lambda_1 \in \{4,\ 2,\ 1,\ 0.5857\dots\}$.
> - **(a)** Setting the state graph $G$ equal to the base graph $G'$ and letting $G$ vary.
> - **(b)** Setting the state graph $G$ to be the complete graph and letting $G'$ vary.
>
> In both panels, a larger algebraic connectivity $\lambda_1$ corresponds to a faster decrease of the state variance.

**The choice of the output.** In this experiment, we consider a much more refined grid ($p = 720$), we set $G'$ (and thus $G$) to be the complete graph, pick the step-size $\tau = 10^{-1}$, and compare the four different estimates of the optimal solution to (38) yielded by Algorithm 1 before reaching consensus, which we denote by $\sigma_1^k, \dots, \sigma_4^k$ for every $k \in \mathbb{N}$. Specifically, $\sigma_1^k$ is associated to the divergence constraint, $\sigma_2^k$ to the superlinear term, $\sigma_3^k$ to the $\ell^1$ functional, and $\sigma_4^k$ to the bridge and water constraint (the last functional in (38)). As expected, in the red squares in Figure 2(a) we can see that the solution estimate corresponding to the divergence constraint always satisfies the conservation law (i.e., the divergence constraint), while not respecting exactly the bridge constraint. The solution estimate corresponding to the bridge constraint presents the opposite behavior, and the one corresponding to the $\ell^1$ functional does provide a very sparse estimate. These differences and an (approximately) optimal solution are shown in Figure 2.

> **Figure 2.** *Comparison of different output choices.* To better visualize the vector fields $x \mapsto \sigma(x)$, the intensity of the corresponding transport densities $|\sigma| : x \mapsto \|\sigma(x)\|$ is shown in gray-scale; the blue and red regions correspond to the measures $\mu$ and $\nu$ respectively.
> - **(a)** The four solution estimates $|\sigma_1^k|, |\sigma_2^k|, |\sigma_3^k|, |\sigma_4^k|$ at the early iteration $k = 40$ (red squares mark the zoomed-in regions shown in the middle).
> - **(b)** Solution estimate $\sigma_3^k$ after $300$ iterations and a state variance of $\operatorname{Var}(\sigma^k) = 10^{-4}$, where $\sigma^k = (\sigma_1^k, \dots, \sigma_4^k)$ is the vector of solution estimates. The lake ($\mathrm{Wtr}$) and bridge ($\mathrm{Brg}$) regions are highlighted.

### 5.2 Distributed SVM

In this experiment, we show an application of the proposed graph-based Douglas–Rachford method in a fully distributed optimization framework. We consider the classical *Support Vector Machine* (SVM) problem formulated in primal form [24]:

$$
\min_{f \in \mathcal{H}_K} \sum_{i=1}^{n} \max\{1 - y_i f(x_i), 0\} + \gamma\|f\|_K^2, \tag{39}
$$

where $\{(x_1, y_1), \dots, (x_n, y_n)\}$ are labeled points in some domain $\Omega \subset \mathbb{R}^d$, $\gamma$ is a positive parameter, and $\mathcal{H}_K$ is a Reproducing Kernel Hilbert Space [25] endowed with the scalar product induced by the Gaussian kernel, namely $k(x, y) := \exp\{-\|x - y\|^2 / (2\sigma^2)\}$ for some fixed $\sigma > 0$ and all $x, y \in \Omega$. Despite having an infinite-dimensional formulation, by the Representer Theorem [26], an optimal solution of (39) can be found as a linear combination of the kernel function evaluated at the training points, namely $f^{*}(x) := \alpha_1^{*}k(x_1, x) + \cdots + \alpha_n^{*}k(x_n, x)$ for all $x \in \Omega$. Hence, problem (39) admits a finite-dimensional reformulation, which reads as

$$
\min_{\alpha \in \mathbb{R}^n} \sum_{i=1}^{n} \max\{1 - y_i(k_i \cdot \alpha), 0\} + \gamma\,\alpha^{*}K\alpha, \tag{40}
$$

where $K \in \mathbb{R}^{n \times n}$ is the matrix defined as $K_{ij} := k(x_i, x_j)$ for all $x_i, x_j$ in the training set, and $k_i \in \mathbb{R}^n$ is the $i$-th row of $K$ for all $i \in \{1, \dots, n\}$.

**Privacy and communication constraints.** Suppose that each training point $x_i \in \Omega$ represents an agent equipped with a personal information, i.e., the label $y_i \in \{\pm 1\}$. Due to some security policy, each agent is able to communicate only with a *local superior*, that we will refer to as *official*. An official is an agent with a higher relevance in the network, who is in charge of treating the information of a subset of agents in its neighborhood. We assume that each official knows the full matrix $K$, whereas agent $i$ only knows its label $y_i$ and the $i$-th row of $K$. We suppose there are $C$ officials spread around $\Omega$, which can communicate between each other. Of course, such a structure serves as an example and it has to be clear that, in general, any communication graph can be considered.

Taking into account our communication constraints, we set: $g_c(\alpha) := d_c / \big(\textstyle\sum_{c=1}^{C} d_c\big)\,\alpha^{*}K\alpha$ for each $c \in \{1, \dots, C\}$, where $d_c$ are the degrees of the officials. Then, we partition the set of indices $i \in \{1, \dots, n\}$ into $C$ sets $\mathcal{I}_1, \dots, \mathcal{I}_C$, each containing exactly $p$ indices corresponding to the agents that communicate with the official $c$, and denote by $h_{c,i}(\alpha) := \max\{1 - y_\xi(k_\xi \cdot \alpha), 0\}$, where $\xi$ is the index of the $i$-th agent under the official $c$. Therefore, the objective function (40) can be split into

$$
\min_{\alpha \in \mathbb{R}^n} \gamma\sum_{c=1}^{C} g_c(\alpha) + \sum_{c=1}^{C}\sum_{i=1}^{p} h_{c,i}(\alpha). \tag{41}
$$

In our model problem, we suppose that the officials are located in a circle in such a way that the first can communicate with the second and the last, while the second can only communicate with the third and the first, the third with the second and the fourth and so on. The resulting communication structure is depicted in Figure 3. Specifically, we consider the ordered directed graph $G = (\mathcal{N}, \mathcal{E})$ with nodes $\mathcal{N} = \{1, \dots, C\} \cup (\{1, \dots, C\} \times \{1, \dots, p\})$, and edges $\mathcal{E} := \{(c, (c, i)) \mid c = 1, \dots, C,\ i = 1, \dots, p\} \cup \{(c, c+1) \mid c = 1, \dots, C-1\} \cup \{(1, C)\}$. Note that $\mathcal{N}$ can be ordered enumerating the nodes as follows: $1, (1, 1), (1, 2), \dots, (1, p), 2, (2, 1), (2, 2), \dots, (2, p), \dots, C, (C, 1), \dots, (C, p)$. The terms in the objective function (40) are associated with the nodes in $\mathcal{N}$ in the obvious way (i.e., each function $g_c$ is associated to the node $c$, and each function $h_{c,i}$ is associated to the node $(c, i)$ for all $c = 1, \dots, C$ and $i = 1, \dots, p$), and, thus, can be ordered analogously.

> **Figure 3.** *Bilevel graph considered in the distributed SVM experiment (cf., Section 5.2).* The officials $1, 2, \dots, C$ form a ring (edges $(c, c+1)$ for $c = 1, \dots, C-1$ together with the closing edge $(1, C)$), and each official $c$ is connected to its $p$ subordinate agents $(c, 1), \dots, (c, p)$. Thick edges are associated to the base graph $\mathcal{E}' = \mathcal{E} \setminus \{(1, C)\}$.

**Methods and comparisons.** We compare the performance of the following methods.

1. A distributed Douglas–Rachford method according to Algorithm 3, with step-size $\sigma > 0$ and relaxation parameters $\theta_k = 1$ for all $k \in \mathbb{N}$, associated to the bilevel graph $\mathrm{biG} = (\mathcal{N}, \mathcal{E}, \mathcal{E} \setminus \{(1, C)\})$, where $G = (\mathcal{N}, \mathcal{E})$ is the ordered directed graph depicted in Figure 3. Note, in particular, that we are considering a tree base graph.
2. P-EXTRA [11, Algorithm 2] with step-size $\sigma > 0$ and *mixing matrices* $W = I - \frac{1}{n+C}L$ and $\widetilde{W} = \frac{1}{2}(W + I)$, where $L \in \mathbb{R}^{(n+C)^2}$ is the graph Laplacian of $G$ and $I \in \mathbb{R}^{(n+C)^2}$ is the identity.
3. A distributed PDHG method [4] obtained with the following procedure. Let $L$ be the Laplacian of the state graph in Figure 3, and consider $\mathbf{L} := L \otimes I_n$ where $I_n$ is the identity matrix in $\mathbb{R}^n$. The PDHG method can be used to solve in a distributed fashion the following product-space reformulation of (40):

$$
\min_{\boldsymbol{\alpha} \in \mathbb{R}^{n \times (n+C)}} \gamma\sum_{c=1}^{C} g_c(\alpha_c) + \sum_{c=1}^{C}\sum_{i=1}^{p} h_{c,i}(\alpha_{c,i}) + I\{\mathbf{L}\boldsymbol{\alpha} = 0\}, \tag{42}
$$

where $\alpha_c$ and $\alpha_{c,i}$ for all $c \in \{1, \dots, C\}$ and $i \in \{1, \dots, p\}$ are all $n$-dimensional vectors and together form the $n \times (n+C)$ vector $\boldsymbol{\alpha}$. Note that $\mathbf{L}\boldsymbol{\alpha} = 0$ if and only if all these $\alpha$ coincide.

To test and compare the three methods we use the following procedure. We create an artificial dataset in $\mathbb{R}^2$ with $n = 50$, $C = 5$, and $p = 10$. We pick $10$ values for the step-size $\sigma > 0$ in a logarithmic scale between $10^{-2}$ and $10^1$. For each step-size $\sigma$ we run the distributed DRS and P-EXTRA with step-size $\sigma$. For the PDHG method we only vary the primal step-size, choosing the dual step-size $\gamma = (\sigma\|L\|^2)^{-1}$ to guarantee convergence. We tested PDHG also for other choices of dual step-size $\gamma \leq (\sigma\|L\|^2)^{-1}$, but the performance was always worse, and hence, these choices were discarded. At every iteration, every method provides an estimate of the optimal solution to (40), e.g., the mean of all estimated solutions of every single node. Such points are used to evaluate the objective function (40) in Figure 4(a). Further, at every iteration we compute the three state variances, and, eventually, compare them in Figure 4(b).

**Comments.** Figure 4 shows the mean and the region between the best and the worst performance from $10$ independent runs with different step-size choices, for the three compared methods. For the sake of fairness, we compared the objective function and the state variance for the three methods as a function of the iteration number. Here, we can clearly see that the proposed distributed DRS outperforms P-EXTRA and the distributed PDHG, reaching a state variance of the order $10^{-2}$ within about a thousand iterations, see Figure 4(b).

> **Figure 4.** *Comparison between the proposed method DRS, P-EXTRA and PDHG.* For every method, each line is the mean of $10$ independent runs with different step-sizes; the corresponding shaded regions denote the best and the worst cases. Iterations range over $0$–$10000$.
> - **(a)** Mean objective function as a function of the iteration number for DRS, P-EXTRA and PDHG.
> - **(b)** Mean state variance (semi-log) as a function of the iteration number for DRS, P-EXTRA and PDHG.
>
> The proposed DRS attains the lowest objective and the lowest state variance throughout.

---

## 6. Conclusions

In this work, we proposed graph-based extensions of the DRS method based on the notion of bilevel graph, which encompasses several known and new generalizations of the DRS method to (2). This work shows that for the $N$-operator problem there are at least as many unconditionally stable FRS methods with a minimal lifting as the number of possible bilevel graphs for (2). In fact, we believe that these are infinitely many (modulo equivalence). A deeper question is whether the graph-based DRS encompasses *all* possible methods for (2). This will be the topic of future work. In the future, we also plan to embed $N - 1$ forward terms into Algorithm 1 leading to a graph-based extension of the Davis–Yin method.

---

## Acknowledgments

This work has received funding from the European Union's Framework Programme for Research and Innovation Horizon 2020 (2014–2020) under the Marie Skłodowska-Curie Grant Agreement No. 861137. The Institute of Mathematics and Scientific Computing at the University of Graz, with which K.B. and E.C. are affiliated, is a member of NAWI Graz (<https://nawigraz.at/en>).

---

## References

[1] K. Bredies, E. Chenchene, D. A. Lorenz, and E. Naldi, "Degenerate preconditioned proximal point algorithms," *SIAM Journal on Optimization*, vol. 32, no. 3, pp. 2376–2401, 2022.

[2] K. Bredies and H. Sun, "Preconditioned Douglas–Rachford splitting methods for convex-concave saddle-point problems," *SIAM Journal on Numerical Analysis*, vol. 53, no. 1, pp. 421–444, 2015.

[3] K. Bredies and H. Sun, "A proximal point analysis of the preconditioned alternating direction method of multipliers," *Journal of Optimization Theory and Applications*, vol. 173, no. 3, pp. 878–907, 2017.

[4] A. Chambolle and T. Pock, "A first-order primal-dual algorithm for convex problems with applications to imaging," *Journal of Mathematical Imaging and Vision*, vol. 40, no. 1, pp. 120–145, 2011.

[5] D. W. Peaceman and H. H. Rachford, Jr., "The numerical solution of parabolic and elliptic differential equations," *Journal of the Society for Industrial and Applied Mathematics*, vol. 3, no. 1, pp. 28–41, 1955.

[6] D. Davis and W. Yin, "Convergence rate analysis of several splitting schemes," in *Splitting Methods in Communication, Imaging, Science, and Engineering*, pp. 115–163, Springer International Publishing, 2016.

[7] P. L. Lions and B. Mercier, "Splitting algorithms for the sum of two nonlinear operators," *SIAM Journal on Numerical Analysis*, vol. 16, no. 6, pp. 964–979, 1979.

[8] E. K. Ryu, "Uniqueness of DRS as the 2 operator resolvent-splitting and impossibility of 3 operator resolvent-splitting," *Mathematical Programming*, vol. 182, no. 1, pp. 233–273, 2020.

[9] Y. Malitsky and M. K. Tam, "Resolvent splitting for sums of monotone operators with minimal lifting." arXiv:2108.02897v1, 2021.

[10] L. Condat, D. Kitahara, A. Contreras, and A. Hirabayashi, "Proximal splitting algorithms for convex optimization: A tour of recent advances, with new twists." arXiv:1912.00137, 2019.

[11] W. Shi, Q. Ling, G. Wu, and W. Yin, "A proximal gradient algorithm for decentralized composite optimization," *IEEE Transactions on Signal Processing*, vol. 63, no. 22, pp. 6013–6023, 2015.

[12] L. M. Briceño-Arias and F. Roldán, "Resolvent of the parallel composition and the proximity operator of the infimal postcomposition," *Optimization Letters*, 2022.

[13] N. M. M. de Abreu, "Old and new results on algebraic connectivity of graphs," *Linear Algebra and its Applications*, vol. 423, no. 1, pp. 53–73, 2007.

[14] L. Lovász, M. Saks, and A. Schrijver, "Orthogonal representations and connectivity of graphs," *Linear Algebra and its Applications*, vol. 114–115, pp. 439–454, 1989.

[15] W. Shi, Q. Ling, G. Wu, and W. Yin, "EXTRA: an exact first-order algorithm for decentralized consensus optimization," *SIAM Journal on Optimization*, vol. 25, no. 2, pp. 944–966, 2015.

[16] H. H. Bauschke and P. L. Combettes, *Convex analysis and monotone operator theory in Hilbert spaces.* CMS Books in Mathematics/Ouvrages de Mathématiques de la SMC, Springer, Cham, second ed., 2017.

[17] D. Davis, "Convergence rate analysis of the forward-Douglas–Rachford splitting scheme," *SIAM Journal on Optimization*, vol. 25, no. 3, pp. 1760–1786, 2015.

[18] D. J. Rose, "A graph-theoretic study of the numerical solution of sparse positive definite systems of linear equations," in *Graph Theory and Computing*, pp. 183–217, Academic Press, 1972.

[19] L. Vandenberghe and M. S. Andersen, "Chordal graphs and semidefinite optimization," *Foundations and Trends in Optimization*, vol. 1, no. 4, pp. 241–433, 2015.

[20] M. Beckmann, "A continuous model of transportation," *Econometrica*, vol. 20, no. 4, pp. 643–660, 1952.

[21] G. Carlier, C. Jimenez, and F. Santambrogio, "Optimal transportation with traffic congestion and wardrop equilibria," *SIAM Journal on Control and Optimization*, vol. 47, no. 3, pp. 1330–1350, 2008.

[22] F. Santambrogio, *Optimal Transport for Applied Mathematicians: Calculus of Variations, PDEs, and Modeling.* Progress in Nonlinear Differential Equations and Their Applications, Springer International Publishing, 2015.

[23] S. Holly and A. Nieße, "On the effects of communication topologies on the performance of distributed optimization heuristics in smart grids," in *INFORMATIK 2020*, pp. 783–794, Gesellschaft für Informatik, Bonn, 2021.

[24] O. Chapelle, "Training a support vector machine in the primal," *Neural Computation*, vol. 19, no. 5, pp. 1155–1178, 2007.

[25] N. Aronszajn, "Theory of reproducing kernels," *Transactions of the American Mathematical Society*, vol. 68, pp. 337–404, 1950.

[26] G. S. Kimeldorf and G. Wahba, "A correspondence between Bayesian estimation on stochastic processes and smoothing by splines," *The Annals of Mathematical Statistics*, vol. 41, no. 2, pp. 495–502, 1970.
