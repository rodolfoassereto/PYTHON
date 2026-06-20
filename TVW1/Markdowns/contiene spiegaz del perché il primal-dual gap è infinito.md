# Surrogate Primal–Dual Gap for the TV–Wasserstein EAP Problem

*Full working record. The goal is a **finite, computable primal–dual gap** to use as a stopping
criterion for the $\beta=0$ TV–Wasserstein / EAP reconstruction, whose naive gap is $+\infty$ at
the iterates because of three indicator terms.*

---

## How to read this document

- **Part I (§1–§10) is the main path** — the actual construction, end to end.
- **Part II (Appendices A–C)** collects the corrections to the first draft, the design rationale
  behind the choices (the "why not X?" questions), and the open points still to confirm.

The one-line summary of the method:

> **Each of the three blow-ups is an indicator. Bound the variable whose dual partner blows up;
> that bound enters as a ball, and the ball's support function (or a smoothing of it) replaces the
> offending indicator.** The bound that fixes a blow-up on one side of the duality always lives on
> a variable from the *opposite* side (the **cross-pairing principle**).

---

## Notation and conventions

| Symbol                     | Meaning                                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| $\Omega,\ Y$               | spatial domain / displacement (q-space) domain                                                                    |
| $P$                        | EAP field (primal variable); physically $P\ge 0$                                                                  |
| $Q$                        | **Beckmann flux/momentum** field (not a Kantorovich coupling)                                                     |
| $K=UF$                     | undersampled Fourier operator; $F$ unitary DFT, $U$ sampling mask. Generally **not injective**: $\ker K\neq\{0\}$ |
| $\mathcal{K}$              | stacked saddle-point operator (kept notationally distinct from $K$)                                               |
| $E$                        | measured data                                                                                                     |
| $f,g$                      | dual variables                                                                                                    |
| $\alpha_1,\alpha_2>0$      | transport / mass-TV regularisation weights                                                                        |
| $\mathbb{J},\mathbb{I}$    | linear operators; $\mathbb{J}=\mathbb{J}^{*}$ (self-adjoint)                                                      |
| $R(P)$                     | regulariser $\ \alpha_1\mathrm{TV}_{W_1}(\mathbb{J}P)+\alpha_2\mathrm{TV}(\mathbb{I}P)$                           |
| $\mathcal{I}_S,\ \sigma_S$ | indicator of $S$ / support function $\sigma_S(\cdot)=\sup_{w\in S}\langle w,\cdot\rangle$                         |
| $\Pi_{\ker K}$             | orthogonal ($\ell^2$) projection onto $\ker K$                                                                    |
| $(K^{*})^{\dagger}$        | Moore–Penrose pseudoinverse of $K^{*}$                                                                            |
| $(a)_+$                    | $\max(a,0)$                                                                                                       |

**Convention (non-standard — important).** Following the working draft, the *proximable* part is
called $F$ and the *composite* part (fed through $\mathcal K$) is called $G^{*}$:

$$
F(P,Q)=\tfrac12\|KP-E\|^2+\alpha_1\|Q\|_{2,1},
\qquad
G^{*}(\mathcal{K}(P,Q))=\text{composite part}.
$$

This is the transpose of the usual Chambolle–Pock labelling, so keep it in mind throughout.

---

# PART I — MAIN PATH

## 1. The variational problem (primal)

With $\beta=0$,

$$
\mathcal{P}(P,Q)=
\underbrace{\tfrac12\|KP-E\|^2+\alpha_1\|Q\|_{2,1}}_{F(P,Q)}
+\underbrace{\mathcal{I}_0(\nabla_\Omega\mathbb{J}P+\operatorname{div}_Y Q)+\alpha_2\|\nabla\mathbb{I}P\|_1}_{G^{*}(\mathcal{K}(P,Q))} .
$$

- The coupling indicator $\mathcal{I}_0(\nabla_\Omega\mathbb{J}P+\operatorname{div}_Y Q)$ enforces the
  **Beckmann continuity (mass-balance) constraint** of the flux formulation of $W_1$: $Q$ is the
  displacement-domain flux balancing the zero-mean spatial variation $\nabla_\Omega\mathbb{J}P$, and
  $\alpha_1\|Q\|_{2,1}$ is its transport cost. (See Appendix B.4.)
- The mass term is $\alpha_2\|\nabla\mathbb{I}P\|_1=\alpha_2\,\mathrm{TV}(\mathbb{I}P)$ — note the
  coefficient is $\alpha_2$, **not** $1/\alpha_2$ (Appendix A).

The stacked operator and its adjoint (with $\mathbb{J}=\mathbb{J}^{*}$):

$$
\mathcal{K}=\begin{pmatrix}\nabla_\Omega\mathbb{J} & \operatorname{div}_Y\\[2pt]\nabla\mathbb{I} & 0\end{pmatrix},
\qquad
\mathcal{K}^{*}\begin{pmatrix}f\\g\end{pmatrix}
=-\begin{pmatrix}\mathbb{J}\operatorname{div}_\Omega f+\mathbb{I}^{*}\operatorname{div} g\\[2pt]\nabla_Y f\end{pmatrix}
=\begin{pmatrix}-\tilde P\\-\tilde Q\end{pmatrix},
$$

so the data-side dual variable is $\ \tilde P:=\mathbb{J}\operatorname{div}_\Omega f+\mathbb{I}^{*}\operatorname{div} g\ $ and $\ \tilde Q:=\nabla_Y f$.

## 2. Saddle-point form and dual

**Saddle point.**

$$
\min_{P,Q}\max_{f,g}\ F(P,Q)+\langle\mathcal{K}(P,Q),(f,g)\rangle-G(f,g),
\qquad
G(f,g)=O(f)+\mathcal{I}_{B^\infty}(g/\alpha_2),
$$

with the pairing

$$
\langle\mathcal{K}(P,Q),(f,g)\rangle
=\langle\nabla_\Omega\mathbb{J}P,f\rangle+\langle\nabla\mathbb{I}P,g\rangle+\langle\operatorname{div}_Y Q,f\rangle
=-\langle\mathbb{J}\operatorname{div}_\Omega f,P\rangle-\langle\mathbb{I}^{*}\operatorname{div} g,P\rangle-\langle\nabla_Y f,Q\rangle .
$$

Crucially $O(f)=\mathcal{I}_0^{*}(f)\equiv 0$: the multiplier $f$ of the equality constraint is
**unconstrained** in the saddle point (Appendix A.2).

**Dual.** With $F$ separable, $F^{*}(\tilde P,\tilde Q)=F_1^{*}(\tilde P)+F_2^{*}(\tilde Q)$, where
$F_1=\tfrac12\|K\cdot-E\|^2$, $F_2=\alpha_1\|\cdot\|_{2,1}$. One gets $F_2^{*}(\tilde Q)=\mathcal{I}_{B^{2\infty}}(\tilde Q/\alpha_1)$ and

$$
\mathcal{D}(f,g)=
\underbrace{\tfrac12\big\|K(K^{*}K)^{-1}(\mathbb{J}\operatorname{div}_\Omega f+\mathbb{I}^{*}\operatorname{div} g+K^{*}E)\big\|^2-\tfrac12\|E\|^2
+\mathcal{I}_{B^{2\infty}}\!\big(\tfrac1{\alpha_1}\nabla_Y f\big)}_{F^{*}(-\mathcal{K}^{*}(f,g))}
+\underbrace{O(f)+\mathcal{I}_{B^\infty}(g/\alpha_2)}_{G(f,g)} .
$$

The conjugate uses $K=UF$ (not $\mathcal K$) throughout (Appendix A.3). **The $(K^{*}K)^{-1}$ form is
valid only when $K$ is injective** — see §7.

**Gap.** $\ \mathcal{G}(P^n,Q^n,f^n,g^n)=\mathcal{P}(P^n,Q^n)+\mathcal{D}(f^n,g^n).$

## 3. Why the naive gap is unusable: three indicator blow-ups

At the iterates, three indicator terms are $+\infty$:

1. **Primal coupling indicator** $\ \mathcal{I}_0(\nabla_\Omega\mathbb{J}P+\operatorname{div}_Y Q)$ — infinite when the iterate violates the continuity constraint.
2. **Dual transport indicator** $\ \mathcal{I}_{B^{2\infty}}\!\big(\tfrac1{\alpha_1}\nabla_Y f\big)$ — the conjugate of $\alpha_1\|Q\|_{2,1}$; infinite when $\|\nabla_Y f\|_{2,\infty}>\alpha_1$.
3. **Hidden fidelity range indicator** $\ \mathcal{I}_{\operatorname{range}(K^{*})}(\tilde P)$ — concealed inside $G_1^{*}$ because $\ker K\neq\{0\}$ (§7); infinite when $\tilde P\notin\operatorname{range}(K^{*})$.

The only *proxed* term, $\mathcal{I}_{B^\infty}(g/\alpha_2)$, stays at $0$ (it is enforced by projection
on $g$), so it is harmless.

## 4. The surrogate-gap strategy: an example

**Consider this example.** For
$\min_x f_1(x_1)+f_2(x_2)+G(\mathcal{K}x)$ with $f_1(x_1)=0$ and $\mathcal{K}=[A,B]$, the dual term
$f_1^{*}(-A^{*}y)$ becomes an indicator $\mathcal{I}_{\{0\}}(-A^{*}y)$ the iterates violate. Fix:
bound $\|x_1^{*}\|\le C_f$ and replace $f_1$ by the indicator function $ \mathcal{I}_{C_f B^(f)} $ (or possibly something else, such as $f_1+\tfrac12(\|x_1\|-C_f)_+^2$), where $B$ is a ball in the norm with which we bounded the optimum $x_1^*$. In this way, the dual $f_1^*$ becomes the dual norm $\| \cdot \|_*$, which is always finite, and we have eliminated the problems of infinity of the primal-dual gaop for what concerns $f_1$. Note that the new "surrogate" problem is equivalent to the previous one thanks to the bound $\|x_1^{*}\|\le C_f$.

## 5. The three a priori bounds

All three use the **same energy comparison at $(0,0)$**: since $(0,0)$ is feasible with
$\mathcal{P}(0,0)=\tfrac12\|E\|^2$, optimality gives
$\tfrac12\|KP^{*}-E\|^2+\alpha_1\|Q^{*}\|_{2,1}+\alpha_2\|\nabla\mathbb{I}P^{*}\|_1\le\tfrac12\|E\|^2$, hence

$$
\|Q^{*}\|_{2,1}\le\frac{\|E\|^2}{2\alpha_1},
\qquad
\|KP^{*}-E\|\le\|E\|\ \Rightarrow\ \|KP^{*}\|\le 2\|E\|,
\qquad
R(P^{*})\le\tfrac12\|E\|^2 .
$$

**$C_Q$ (transport).**
$$
\boxed{\ C_Q=\frac{\|E\|^2}{2\alpha_1}\ }\ \ge\ \|Q^{*}\|_{2,1}.
$$

**$C_f$ (coupling).** Gauge-fix $f^{*}(x,0)=0\ \forall x$ on the displacement grid
$Y=\{-\tfrac{N_1}2,\dots,\tfrac{N_1}2\}\times\{-\tfrac{N_2}2,\dots,\tfrac{N_2}2\}$; stationarity in
$Q$ gives $\|\nabla_Y f^{*}\|_{2,\infty}\le\alpha_1$; summing increments along a path from the origin,

$$
\boxed{\ C_f=\max\{N_1,N_2\}\Big(1+\tfrac{\sqrt2}{2}\Big)\ }\ \ge\ \|f^{*}\|_\infty .
$$

> ⚠️ **Open:** the increment bound carries a factor $\alpha_1$, so dimensionally one expects
> $C_f=\alpha_1\max\{N_1,N_2\}(1+\tfrac{\sqrt2}{2})$. Confirm whether $\alpha_1$ was set to $1$ or
> absorbed (Appendix C).

**$C_P$ (fidelity kernel).** Two routes:

- **(Preferred — main path) Nonnegativity.** Impose $P\ge0$ (an EAP is a density), and assume the
  $0$-frequency is sampled (the $b=0$ acquisition). With $F$ the **unitary** DFT,
  $(FP)_0=\tfrac1{\sqrt N}\sum_i P_i=\tfrac1{\sqrt N}\|P\|_1$ (equality, by nonnegativity), and the DC
  coefficient is a coordinate of $KP$, so $|(FP)_0|\le\|KP\|_2$. Hence
  
  $$
  \|\Pi_{\ker K}P^{*}\|_2\le\|P^{*}\|_2\le\|P^{*}\|_1=\sqrt N\,(FP^{*})_0\le\sqrt N\,\|KP^{*}\|_2
\le \boxed{\ 2\sqrt N\,\|E\|\ }=:C_P .
  $$

- **(Fallback) Coercivity.** If nonnegativity is *not* imposed: under
  $\ker K\cap\ker R=\{0\}$, $N(P):=\|KP\|_2+R(P)$ is a norm, so $\|P\|_2\le c\,N(P)$ with
  $c=\big(\min_{\|P\|_2=1}(\|KP\|_2+R(P))\big)^{-1}>0$, giving
  $\|\Pi_{\ker K}P^{*}\|_2\le c\big(2\|E\|+\tfrac12\|E\|^2\big)=:C_P$. Here $c$ is a fixed problem
  constant (joint coercivity modulus of $(K,R)$), estimable offline; it need not be tight.

---

# PART II — SUPPORTING MATERIAL & DESIGN RATIONALE

## Appendix A. Corrections applied to the original draft

The first `temp.pdf` had the right overall conclusion (the gap is $+\infty$ because of the indicator
terms) but five issues, since fixed:

1. **Mass-term coefficient.** It read $\tfrac1{\alpha_2}\|\nabla\mathbb{I}P\|_1=\alpha_2\mathrm{TV}(\mathbb{I}P)$,
   which holds only if $\alpha_2=1$. Correct: $\alpha_2\|\nabla\mathbb{I}P\|_1$. Forced also by the
   dual constraint $\|g\|_\infty\le\alpha_2$, i.e. $(\alpha_2\|\cdot\|_1)^{*}=\mathcal{I}_{B^\infty}(g/\alpha_2)$.
2. **$F^{*}$ misattribution.** With $F(a,b)=\mathcal{I}_0(a)+\alpha_2\|b\|_1$,
   $F^{*}(f,g)=\mathcal{I}_{\{0\}}^{*}(f)+\mathcal{I}_{B^\infty}(g/\alpha_2)=0+\mathcal{I}_{B^\infty}(g/\alpha_2)$.
   So $f$ is the **free** multiplier of the equality constraint, *not* constrained by a "$O(f)$" term.
   The Kantorovich/Lipschitz constraint $\|\nabla_Y f\|_{2,\infty}\le\alpha_1$ arises on the **dual**
   side, from $G_2^{*}=(\alpha_1\|\cdot\|_{2,1})^{*}$. (And it is "$\le\alpha_1$", not "$\le1$".)
3. **$G_1^{*}$ operator.** The data conjugate must use $K=UF$ throughout:
   $\tfrac12\|K(K^{*}K)^{-1}(\tilde P+K^{*}E)\|^2-\tfrac12\|E\|^2$, **not** $\mathcal{K}^{*}\mathcal{K}$
   or $\mathcal{K}^{*}E$ (the latter does not even typecheck — $E$ lives in measurement space). Root
   cause: the symbol "$K$" was overloaded for both $UF$ and the stacked operator; keep $K$ vs
   $\mathcal{K}$ distinct.
4. **Beckmann vs Kantorovich (terminology).** $\mathcal{I}_0(\nabla_\Omega\mathbb{J}P+\operatorname{div}_Y Q)$
   enforces the **Beckmann continuity equation**, not a "Kantorovich optimality condition"; $Q$ is a
   flux/momentum field, not an "optimal transport plan" (the coupling $\pi$). The genuinely
   *Kantorovich* object is the potential $f$, so "Kantorovich potential" for $f$ is fine. Corrected
   sentence:
   
   > *The coupling constraint $\nabla_\Omega\mathbb J P+\operatorname{div}_Y Q=0$ encoded by $\mathcal I_0$
   > is the continuity (mass-balance) constraint of the Beckmann/flux formulation of $W_1$: it
   > requires the displacement-domain flux $Q$ to balance the zero-mean spatial variation
   > $\nabla_\Omega\mathbb J P$, so that minimising $\alpha_1\|Q\|_{2,1}$ subject to it reproduces the
   > $W_1$ cost on $\mathbb J P$. The multiplier $f$ is the Kantorovich potential, constrained on the
   > dual side by $\|\nabla_Y f\|_{2,\infty}\le\alpha_1$.*
5. **Sign of the coupling.** Appears as $+\nabla_\Omega\mathbb{J}P+\operatorname{div}_Y Q$ here vs
   $-\nabla_\Omega\mathbb{J}P+\operatorname{div}_Y Q$ elsewhere; convention-dependent and internally
   consistent with this $\mathcal{K}$, just keep aligned across notes.
