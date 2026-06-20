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

### Dual of the transport TV

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

### Saddle-point form

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

### Proximal step of the fidelity

$$
\operatorname{prox}_{\sigma F_1}(\bar P)
=\arg\min_P\ \tfrac12\|K(P)-E\|^2+\tfrac1{2\sigma}\|P-\bar P\|^2 .
$$

Stationarity:
$
\sigma K^{*}(K(P)-E)+P-\bar P=0
\;\Longrightarrow\;
\sigma K^{*}K\,P-\sigma K^{*}E+P=\bar P,
$($\mathrm{id}+\sigma K^{*}K)^{-1}(\bar P+\sigma K^* *E).
$

So
$
\operatorname{prox}_{\sigma F_1}(\bar P)=(\operatorname{id}+\sigma K^{*}K)^{-1}(\bar P+\sigma K^{*}E),
$and, using $K=UF$ with $F$ unitary,
$\operatorname{id}+\sigma K^{*}K=F^{*}(\operatorname{id}+\sigma U^{*}U)F,
\qquad
(\operatorname{id}+\sigma K^{*}K)^{-1}=F^{*}(\operatorname{id}+\sigma U^{*}U)^{-1}F.
$

Note that this expression is not everywhere defined: only if the FFT-transformed input lives in the sampled frequencies.

---

## 2. Another primal-dual formulation, by dualizing the fidelity

Using $\ \tfrac12\|K(P)-E\|^2=\max_h\langle K(P)-E,h\rangle-\tfrac12\|h\|^2
=\max_h\langle K(P),h\rangle-\big(\tfrac12\|h\|^2+\langle E,h\rangle\big)$:

$$
\min_{\substack{P,Q\\ P\ge0}}\max_{f,g,h}\
\underbrace{\alpha_1\|Q\|_{2,1}}_{F(P,Q)}
-\underbrace{\Big(\mathcal{I}_0(\delta f)+\tfrac12\|h\|^2+\langle E,h\rangle+\mathcal{I}_{\alpha_2 B^\infty}(g)\Big)}_{G(f,g,h)}
$$

$$
+\ \langle\nabla_x\mathbb{J}P,f\rangle+\langle\operatorname{div}_y Q,f\rangle+\langle\nabla\mathbb{I}P,g\rangle+\langle K(P),h\rangle .
$$

Stacked operator with the fidelity row added (columns $P,Q$ / rows $f,g,h$):

$$
\mathcal{K}=\begin{bmatrix}\nabla_x\mathbb{J} & \operatorname{div}_y\\[2pt]\nabla\mathbb{I} & 0\\[2pt]K & 0\end{bmatrix}
\begin{matrix}f\\ g\\ h\end{matrix}
\qquad
\mathcal{K}^{*}=\begin{bmatrix}-\mathbb{J}^{*}\operatorname{div}_x & -\mathbb{I}^{*}\operatorname{div} & K^{*}\\[2pt]-\nabla_y & 0 & 0\end{bmatrix}
\begin{matrix}P\\ Q\end{matrix}
$$

### Primal

$$
\min_{P,Q}\max_{f,g,h}\
\mathcal{I}_{\ge0}(P)+\alpha_1\|Q\|_{2,1}
-\Big(\mathcal{I}_0(\delta f)+\tfrac12\|h\|^2+\langle E,h\rangle+\mathcal{I}_{\alpha_2 B^\infty}(g)\Big)
$$

$$
+\ \langle\nabla_x\mathbb{J}P+\operatorname{div}_y Q,f\rangle+\langle\nabla\mathbb{I}P,g\rangle+\langle K(P),h\rangle .
$$

Collapsing the inner maxima term by term:

$$
\min_{P,Q}\ \mathcal{I}_{\ge0}(P)+\alpha_1\|Q\|_{2,1}
$$

$$
+\underbrace{\max_h\ \langle K(P),h\rangle-\tfrac12\|h\|^2-\langle E,h\rangle}_{=\ \frac12\|K(P)-E\|^2}
$$

$$
+\underbrace{\max_g\ \langle\nabla\mathbb{I}P,g\rangle-\mathcal{I}_{\alpha_2 B^\infty}(g)}_{=\ \alpha_2\|\nabla\mathbb{I}P\|_1}
$$

$$
+\underbrace{\max_{\delta f=0}\ \langle\nabla_x\mathbb{J}P+\operatorname{div}_y Q,f\rangle}_{=\ \mathcal{I}_0\big(\bar\delta(\nabla_x\mathbb{J}P+\operatorname{div}_y Q)\big)} ,
$$

where $\bar\delta$ **selects everything *but* the $y$-centers**.

### Dual

$$
\max_{f,g,h}\min_{P,Q}\
\mathcal{I}_{\ge0}(P)+\alpha_1\|Q\|_{2,1}-\langle\nabla_y f,Q\rangle
+\langle P,\ K^{*}h-\mathbb{J}\operatorname{div}_x f-\mathbb{I}^{*}\operatorname{div} g\rangle-G(f,g,h)
$$

$$
-\ \mathcal{I}_{\alpha_1 B^{2\infty}}(\nabla_y f)
-\ \mathcal{I}_{\le0}\big(K^{*}h-\mathbb{J}\operatorname{div}_x f-\mathbb{I}^{*}\operatorname{div} g\big).
$$

So the dual problem is

$$
\min_{f,g,h}\
\underbrace{\mathcal{I}_0(\delta f)+\mathcal{I}_{\alpha_2 B^\infty}(g)+\tfrac12\|h\|^2+\langle E,h\rangle}_{G(f,g,h)}
+\mathcal{I}_{\alpha_1 B^{2\infty}}(\nabla_y f)
$$

$$
+\underbrace{\mathcal{I}_{\le0}\big(K^{*}h-\mathbb{J}\operatorname{div}_x f-\mathbb{I}^{*}\operatorname{div} g\big)}_{\subset\ F^{*}(-\mathcal{K}^{*}(f,g,h))} .
$$

### Grouped form

**Primal:**
$
\min_{P,Q}\
\underbrace{\mathcal{I}_{\ge0}(P)}_{F_1(P)}
+\underbrace{\alpha_1\|Q\|_{2,1}}_{F_2(Q)}
+\underbrace{\mathcal{I}_0\big(\bar\delta(\nabla_x\mathbb{J}P+\operatorname{div}_y Q)\big)+\alpha_2\|\nabla\mathbb{I}P\|_1+\tfrac12\|K(P)-E\|^2}_{G^{*}(\mathcal{K}(P,Q))} .
$

**Dual:**
$
\min_{f,g,h}\
\underbrace{\mathcal{I}_0(\delta f)}_{G_1(f)}
+\underbrace{\mathcal{I}_{\alpha_2 B^\infty}(g)}_{G_2(g)}
+\underbrace{\tfrac12\|h\|^2+\langle E,h\rangle}_{G_3(h)}
+\underbrace{\mathcal{I}_{\le0}\big(K^{*}h-\mathbb{J}\operatorname{div}_x f-\mathbb{I}^{*}\operatorname{div} g\big) +\mathcal{I}_{\alpha_1 B^{2\infty}}(\nabla_y f)}_{F^{*}(-\mathcal{K}^{*}(f,g,h))} .
$

---

## 3. Surrogate problem

Reverse-Huber surrogate (steepness $M$):
$
\phi_M(t)=
\begin{cases}
|t| & \text{if }|t|\le M,\\[3pt]
\dfrac M2+\dfrac1{2M}\,t^2 & \text{if }|t|>M.
\end{cases}
$

**Surrogate primal:**

$\min_{P,Q} \mathcal{I}_{\ge 0}(P) + \mathcal{I}_{C_P B^\infty}(P)
+\phi_{C_Q}\big(\alpha_1\|Q\|_{2,1}\big)
+\boxed{\,C_f\big\|\bar\delta(\nabla_x\mathbb{J}P+\operatorname{div}_y Q)\big\|_1\,}
+\alpha_2\|\nabla\mathbb{I}P\|_1
+\tfrac12\|K(P)-E\|^2 .
$

**Surrogate dual:**

$\min_{f,g,h}\
\mathcal{I}_0(\delta f)+\mathcal{I}_{C_f B^\infty}(f)+\mathcal{I}_{\alpha_2 B^\infty}(g)
+\tfrac12\|h\|^2+\langle E,h\rangle
+ \|\big(K^{*}h-\mathbb{J}\operatorname{div}_x f-\mathbb{I}^{*}\operatorname{div} g \big)_+\|_1
+\phi^{*}_{C_Q}\big(\alpha_1\|\nabla_y f\|_{2,\infty}\big).
$

Where we used:

- $
  \big(\mathcal{I}_0(\delta\,\cdot)+\mathcal{I}_{C_f B^\infty}(\cdot)\big)^{*}(\tilde f)
  =\sup_{\substack{\delta f=0\\ \|f\|_\infty\le C_f}}\langle\tilde f,f\rangle
  =C_f\,\big\|\bar\delta\,\tilde f\big\|_1 .
  $

- $ \big( \mathcal I_{\ge0}(\cdot) + \mathcal I _{C_f B^\infty}(\cdot)^*\big) (\tilde{P})
  = \sup_{ 0 \le P \le C_P} \langle \tilde{P}, P \rangle
  = \| (\tilde{P})_+\|_1$

---

## 4. Dual of the "surrogate" function $\phi_M$

$$
\boxed{\
\phi_M(t)=
\begin{cases}
|t| & \text{if }|t|\le M,\\[3pt]
\dfrac M2+\dfrac1{2M}t^2 & \text{if }|t|>M.
\end{cases}\ }
$$

**Remark.** $\phi$ is even $\Rightarrow$ $\phi^{*}$ is also even, so it suffices to take $s\ge0$.

$$
\phi_M^{*}(s)=\sup_t\ st-\phi(t)=
\begin{cases}
|t|\le M:\ \sup_t\ st-|t|,\\[3pt]
|t|>M:\ \sup_t\ -\tfrac1{2M}t^2+st-\tfrac M2 .
\end{cases}
$$

**Branch $|t|\le M$:**
$
\begin{cases}
s<-1:\ -Ms-M=-M(s+1) & (\text{discarded}),\\
-1\le s\le1:\ 0,\\
s>1:\ Ms-M & (\text{discarded}).
\end{cases}
$

**Branch $|t|\ge M$:** the unconstrained vertex is
$
t^{*}=
\begin{cases}
sM & \text{if }|s|\ge1,\\
M\,\dfrac{s}{|s|} & \text{if }|s|<1,
\end{cases}
\qquad\text{(at the boundary }M:\ -\tfrac12 M+sM-\tfrac M2=M(s-1)\text{)},
$giving the values
$\begin{cases}
-\tfrac12 s^2 M+s^2 M-\tfrac M2=\boxed{\tfrac12 M(s^2-1)} & |s|\ge1,\\[3pt]
-\tfrac12 M+sM-\tfrac M2=M(|s|-1) & |s|<1.
\end{cases}
$

**Comparison of the two branches.**

- If $s<-1$: $\ \tfrac12 M(s^2-1)>-M(s+1)$, i.e.
  $\tfrac12 M(s+1)(s-1)>-M(s+1)\iff\tfrac12(s-1)<-1\iff s<-1.$ ✓
- If $s>1$: $\ \tfrac12 M(s^2-1)>M(s-1)$, i.e.
  $\tfrac12 M(s+1)(s-1)>M(s-1)\iff\tfrac12(s+1)>1\iff s>1.$ ✓

Therefore
$
\boxed{\
\phi_M^{*}(s)=
\begin{cases}
0 & |s|\le1,\\[3pt]
\tfrac12 M(s^2-1) & |s|\ge1,
\end{cases}
\qquad\text{i.e.}\quad \phi_M^{*}(s)=\tfrac12 M\,(s^2-1)_+\ }
$

A finite, $C^1$ surrogate of $\mathcal{I}_{[-1,1]}$ with **steepness $=M$** (flat on $[-1,1]$, quadratic outside).

### Conjugate of a function of a norm

Let $\varphi(Q)=\phi(\|Q\|)$.

**(Caution: for a general norm one cannot assume the maximizer of $\langle\tilde Q,Q\rangle-\phi(\|Q\|)$ is simply $Q$ aligned with $\tilde Q$.)*

$$
\varphi^{*}(\tilde Q)=\sup_Q\ \langle\tilde Q,Q\rangle-\phi(\|Q\|)
=\sup_{t\ge0}\ \sup_{\|Q\|=t}\langle\tilde Q,Q\rangle-\phi(\|Q\|)
=\sup_{t\ge0}\ t\,\|\tilde Q\|_{*}-\phi(t)
=\phi^{*}\big(\|\tilde Q\|_{*}\big),
$$

using the dual norm $\|\tilde Q\|_{*}$.

---

## 5. Bounds

Since $(P^{*},Q^{*})=(0,0)$ is feasible,
$
\tfrac12\|K(P^{*})-E\|^2+\alpha_1\|Q^{*}\|_{2,1}+\dots\le\tfrac12\|E\|^2.$

### Bound on $P^{*}$

$$
\tfrac12\|K(P^{*})-E\|^2\le\tfrac12\|E\|^2
\;\Longrightarrow\;
\|K(P^{*})\|\le2\|E\|. \tag{$\star$}
$$

**One possibility: using $P\ge0$** (nonnegativity is enforced in the model; the DC/center frequency is sampled; $F$ is the orthonormal FFT):

$$
\|P^*\|_\infty\ (\text{or }\|P^*\|_2)\ \le\ \|P^*\|_1=\sum_{x,y}P^*_{x,y}
=(FP^*)_{\text{center}}
\le \|K(P^*)\|_2
{\le}2\|E\|_2.
$$

(Annotations: $\le\|P\|_1$" by nonnegativity; "$=(FP)_{\text{center}}$" because $U$ always samples the center and F ortho FFT).


**Another possibility (uncertain).** $(\star)$ bounds $\operatorname{proj}_{(\ker K)^\perp}(P^{*})$.
One could then try to use the regularizer $R(P)=\alpha_1\mathrm{TV}_{W_1}(\mathbb{J}P)+\alpha_2\mathrm{TV}(\mathbb{I}P)$
to bound $\operatorname{proj}_{\ker K}(P^{*})$.

### Bound on $Q^{*}$

From $(*)$:
$
\boxed{\ \|Q^{*}\|_{2,1}\le\frac{\|E\|^2}{2\alpha_1}=:C_Q\ }
$

### Bound on $f^{*}$

On the displacement grid
$
Y=\Big\{-\tfrac{N_1}2,\dots,0,\dots,\tfrac{N_1}2\Big\}\times\Big\{-\tfrac{N_2}2,\dots,0,\dots,\tfrac{N_2}2\Big\},
\qquad |\Omega|=(N_1+1)(N_2+1),
$(both factors of even length), gauge-fix $(x,0)=0\ \ \forall x\in\Omega,\ \forall f:Y\to\mathbb{R}^2 .
$ationarity gives $\|\nabla_y f^{*}\|_{2,\infty}\le\alpha_1$, hence $\forall x\in\Omega$,
$\|\nabla f^{*}_x\|_{2,\infty}\le\alpha_1$. Summing increments along a path from the center,
$\|f^{*}\|_\infty\le\max\{N_1,N_2\}\Big(1+\tfrac{\sqrt2}{2}\Big)=:C_f
$

(the $1+\tfrac{\sqrt2}{2}$ accounts for an axial step of length $1$ plus diagonal steps of length
$\tfrac{\sqrt2}{2}$ per unit).

> **WORK IN PROGRESS:** check and refine bound for $C_f$.
