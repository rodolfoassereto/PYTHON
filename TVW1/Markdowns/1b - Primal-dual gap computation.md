# Notes on the primal-dual gap computation

$K=UF$ is the undersampled Fourier operator, $\mathcal{K}$ is the stacked
saddle-point operator, the proximable part is $F$ and the composite part is $G^{*}$.

---

## 1. Primal–dual formulation

Let $P:X\times Y\to\mathbb{R}$, with $|X|=n_x$, $|Y|=n_y$, and $N=n_x\cdot n_y$.

$$
\boxed{\ \min_{P}\ \tfrac12\|K(P)-E\|^2+\mathrm{TV}^{\alpha_1,\alpha_2}_{\widetilde W_1}(P),
\qquad K=UF\ }
$$

$$
\mathrm{TV}^{\alpha_1,\alpha_2}_{\widetilde W_1}(P)=\alpha_1\,\mathrm{TV}_{W_1}(\mathbb{J}P)+\alpha_2\,\mathrm{TV}(\mathbb{I}P).
$$

The two linear operators:

$$
\mathbb{I}P:X\to\mathbb{R},\qquad (\mathbb{I}P)_x=\sum_y P_{x,y},
$$

$$
\mathbb{J}P:X\times Y\to\mathbb{R},\qquad
\mathbb{J}=\operatorname{id}-\tfrac1{n_y}\mathbb{I}^{*}\mathbb{I}
\quad\Bigg(\text{s.t. }\ \sum_y(\mathbb{J}P)_{x,y}=0\ \ \forall x\Bigg).
$$

### Dual formulation of the transport TV

By using the Rubinstein-Kantorovich dual formulation of the $W_1$ (for the first term) and the dual definition of $\|\cdot\|_1$ (for the second), we get:

$$
\mathrm{TV}^{\alpha_1}_{W_1}(P)=
\max_{\substack{\|\nabla_y f\|_{2,\infty}\le\alpha_1\\ \delta f=0}}\langle\nabla_x\mathbb{J}P,\,f\rangle
\;+\;\max_{\|g\|_\infty\le\alpha_2}\langle\nabla\mathbb{I}P,\,g\rangle,
$$

where $\delta$ **selects the $y$-centers**. Introducing the Beckmann flux $Q$ to encode the constraint
$\|\nabla_y f\|_{2,\infty}\le\alpha_1$:

$$
=\max_{f}\min_{Q}\ \alpha_1\|Q\|_{2,1}+\langle\nabla\mathbb{I}P,g\rangle
-\mathcal{I}_0(\delta f)-\mathcal{I}_{B^\infty}\!\Big(\tfrac{g}{\alpha_2}\Big)
+\langle\nabla_x\mathbb{J}P,f\rangle-\langle\nabla_y f,Q\rangle.
$$

### Saddle-point form of the problem

$$
\min_{P,Q}\max_{f,g}\
\underbrace{\tfrac12\|K(P)-E\|^2}_{F_1(P)}
+\underbrace{\alpha_1\|Q\|_{2,1}}_{F_2(Q)}
-\underbrace{\mathcal{I}_0(\delta f)}_{G_1(f)}
-\underbrace{\mathcal{I}_{\alpha_2 B^\infty}(g)}_{G_2(g)}
$$

$$
+\ \langle\nabla_x\mathbb{J}P,f\rangle+\langle\nabla\mathbb{I}P,g\rangle+\langle\operatorname{div}_y Q,f\rangle.
$$

Stacked operator and its adjoint (columns $P,Q$ / rows $f,g$):

$$
\mathcal{K}=\begin{bmatrix}\nabla_x\mathbb{J} & \operatorname{div}_y\\[2pt]\nabla\mathbb{I} & 0\end{bmatrix}
\begin{matrix}f\\ g\end{matrix}
\qquad
\mathcal{K}^{*}=\begin{bmatrix}-\mathbb{J}\operatorname{div}_x & -\mathbb{I}^{*}\operatorname{div}\\[2pt]-\nabla_y & 0\end{bmatrix}
\begin{matrix}P\\ Q\end{matrix}
$$

### 2. Problems

### The primal-dual gap is infinite

**The saddle point takes form**

$$
\min_{P,Q}\max_{f,g}\ F(P,Q)+\langle\mathcal{K}(P,Q),(f,g)\rangle-G(f,g),
\qquad
G(f,g)=0(f)+\mathcal{I}_{B^\infty}(g/\alpha_2),
$$

The primal-dual gap in throughout the (CVhambolle-Pock or Graph-DR) iterations is:

$\mathcal{G}(P^n,Q^n,f^n,g^n)=\mathcal{P}(P^n,Q^n)+\mathcal{D}(f^n,g^n).$

with

$ \mathcal P ( P,Q ) = \mathcal F (P,Q) + \mathcal G^*( \mathcal K (P,Q) ) $

$\mathcal D (f,g) = \mathcal F^*(-\mathcal K^* (f,g)) + \mathcal G(f,g)$

Crucially $0(f)$ is the zero function and its conjugate is $\mathcal I_0$, which is a constrant that is not enforced because only $\mathcal F$ and $\mathcal G$ are directly proxed, not $\mathcal F^*$ and $\mathcal G^*$. This generates the term $\ \mathcal{I}_0(\nabla_\Omega\mathbb{J}P+\operatorname{div}_Y Q)$

Moreover, another issue arises: being $F$ separable, $F^{*}(\tilde P,\tilde Q)=F_1^{*}(\tilde P)+F_2^{*}(\tilde Q)$, where
$F_1=\tfrac12\|K\cdot-E\|^2$, $F_2=\alpha_1\|\cdot\|_{2,1}$. One gets $F_2^{*}(\tilde Q)=\mathcal{I}_{B^{2\infty}}(\tilde Q/\alpha_1)$ and

$$
\mathcal{D}(f,g)=
\underbrace{\tfrac12\big\|K(K^{*}K)^{-1}(\mathbb{J}\operatorname{div}_\Omega f+\mathbb{I}^{*}\operatorname{div} g+K^{*}E)\big\|^2-\tfrac12\|E\|^2
+\mathcal{I}_{B^{2\infty}}\!\big(\tfrac1{\alpha_1}\nabla_Y f\big)}_{F^{*}(-\mathcal{K}^{*}(f,g))}
+\underbrace{O(f)+\mathcal{I}_{B^\infty}(g/\alpha_2)}_{G(f,g)} .
$$

And $K^*K=F^*U^*UF$ is in general not invertible (as soon as we do not sample the whole Fourier space). Requesting that the function $\mathcal D$ is defines is equivalent to imposing the constraint $\ \mathcal{I}_{\operatorname{range}(K^{*})}$.

**Esplicit primal and dual**

$$
\min_{P,Q}\ \max_{f,g}\quad
\underbrace{\tfrac12\|K P-E\|^2+\alpha_1\|Q\|_{2,1}}_{F(P,Q)}
\;-\;\underbrace{\mathcal I_{B_\infty}\!\big(\tfrac1{\alpha_2}g\big)}_{G(f,g)}
\;+\;\langle\nabla_{X}\mathbb J P,\,f\rangle+\langle\operatorname{div}_Y Q,\,f\rangle+\langle\nabla\mathbb I P,\,g\rangle .
$$

## # WORK IN PROGRESS

## 3. In short: the straightforward gap is unusable: three indicator blow-ups

At the iterates, three indicator terms are $+\infty$:

1. **Primal coupling indicator** $\ \mathcal{I}_0(\nabla_\Omega\mathbb{J}P+\operatorname{div}_Y Q)$ — infinite when the iterate violates the continuity constraint.
2. **Dual transport indicator** $\ \mathcal{I}_{B^{2\infty}}\!\big(\tfrac1{\alpha_1}\nabla_Y f\big)$ — the conjugate of $\alpha_1\|Q\|_{2,1}$; infinite when $\|\nabla_Y f\|_{2,\infty}>\alpha_1$.
3. **Hidden fidelity range indicator** $\ \mathcal{I}_{\operatorname{range}(K^{*})}(\tilde P)$ — concealed inside $G_1^{*}$ because $\ker K\neq\{0\}$ (§7); infinite when $\tilde P\notin\operatorname{range}(K^{*})$.

Note that the *proxed* term, $\mathcal{I}_{B^\infty}(g/\alpha_2)$, stays at $0$ (it is enforced by projection
on $g$), so it is harmless.

## 4. How to compute a primal-dual gap? The surrogate problem

**Consider this example.** For
$\min_x f_1(x_1)+f_2(x_2)+G(\mathcal{K}x)$ with $f_1(x_1)=0$ and $\mathcal{K}=[A,B]$, the dual term
$f_1^{*}(-A^{*}y)$ becomes an indicator $\mathcal{I}_{\{0\}}(-A^{*}y)$ the iterates violate. Fix:
bound $\|x_1^{*}\|\le C_f$ and replace $f_1$ by the indicator function $ \mathcal{I}_{C_f B^(f)} $ (or possibly something else, such as $f_1+\tfrac12(\|x_1\|-C_f)_+^2$), where $B$ is a ball in the norm with which we bounded the optimum $x_1^*$. In this way, the dual $f_1^*$ becomes the dual norm $\| \cdot \|_*$, which is always finite, and we have eliminated the problems of infinity of the primal-dual gaop for what concerns $f_1$. Note that the new "surrogate" problem is equivalent to the previous one thanks to the bound $\|x_1^{*}\|\le C_f$.
