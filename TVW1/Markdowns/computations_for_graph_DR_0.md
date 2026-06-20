# Computations for graph-DR

Clean Markdown transcription of the handwritten notes **“Computations for graph-DR”**, written in the notation of `variational_problem.md`.

The notes concern the optimality system of the primal-dual variational problem, its splitting into operator blocks, and the computation of the corresponding resolvents/proximal steps.

---

## 0. Variables and basic notation

The unknown is

$$
z=(P,Q,f,g),
$$

where:

- $P$ is the EAP/primal density field;
- $Q$ is the Beckmann flux/momentum field;
- $f,g$ are dual variables;
- $K=UF$, with $F$ the unitary Fourier transform and $U$ the sampling mask;
- $\mathbb J$ and $\mathbb I$ are the lifted operators from the variational problem;
- $\nabla_\Omega$ is the spatial gradient;
- $\nabla_Y$ is the gradient in the displacement/$Y$ variable;
- $\operatorname{div}_\Omega$, $\operatorname{div}_Y$ are the corresponding discrete divergences.

The functional blocks used in the notes are

$$
F_1(P)=\frac12\|KP-E\|^2,
\qquad
F_2(Q)=\alpha_1\|Q\|_{2,1},
$$

and

$$
G_1(f),
\qquad
G_2(g)=\mathcal I_{\alpha_2 B^\infty}(g).
$$

In the handwritten notes, $G_1$ is used mainly to fix the gauge of $f$, for instance by imposing that the $Y$-centres of $f$ vanish. In a bounded/surrogate version one may instead use

$$
G_1(f)=\mathcal I_{\{0\}}(Sf)+\mathcal I_{C_fB^\infty}(f),
$$

so that

$$
\operatorname{prox}_{\lambda G_1}(f)
=\operatorname{proj}_{\ker S\cap C_fB^\infty}(f).
$$

Here $S$ is the operator extracting the gauge component of $f$.

---

## 1. Optimality system

The handwritten optimality system is encoded by the block inclusion

$$
0\in
\begin{pmatrix}
\partial F_1(P) & 0 & -\mathbb J^*\operatorname{div}_\Omega & -\mathbb I^*\operatorname{div} \\
0 & \partial F_2(Q) & -\nabla_Y & 0 \\
\nabla_\Omega\mathbb J & \operatorname{div}_Y & \partial G_1(f) & 0 \\
\nabla\mathbb I & 0 & 0 & \partial G_2(g)
\end{pmatrix}
\begin{pmatrix}
P\\Q\\f\\g
\end{pmatrix}.
$$

Equivalently, componentwise:

$$
\begin{aligned}
\partial_P: \quad&
\partial F_1(P)-\mathbb J^*\operatorname{div}_\Omega f
-\mathbb I^*\operatorname{div} g \ni 0,
\\
\partial_Q: \quad&
\partial F_2(Q)-\nabla_Y f \ni 0,
\\
\partial_f: \quad&
\partial G_1(f)-\nabla_\Omega\mathbb J P
+\operatorname{div}_Y Q \ni 0,
\\
\partial_g: \quad&
\partial G_2(g)+\nabla\mathbb I P \ni 0.
\end{aligned}
$$

> **Sign convention.** The handwritten pages contain the usual sign-sensitive pairings between gradients and divergences. The resolvent formulas below are written with the signs inferred from the actual resolvent computations in the notes. If the primal coupling is written with the opposite sign convention, the corresponding signs in the $A_1$ and/or $A_3$ blocks must be flipped consistently.

---

## 2. Splitting into four blocks

The operator is split as

$$
A=A_0+A_1+A_2+A_3,
$$

where

$$
A_0=
\begin{pmatrix}
\partial F_1 & 0 & 0 & 0\\
0 & \partial F_2 & 0 & 0\\
0 & 0 & \partial G_1 & 0\\
0 & 0 & 0 & \partial G_2
\end{pmatrix},
$$

$$
A_1=
\begin{pmatrix}
0 & 0 & -\mathbb J^*\operatorname{div}_\Omega & 0\\
0 & 0 & 0 & 0\\
-\nabla_\Omega\mathbb J & 0 & 0 & 0\\
0 & 0 & 0 & 0
\end{pmatrix},
$$

$$
A_2=
\begin{pmatrix}
0 & 0 & 0 & -\mathbb I^*\operatorname{div}\\
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0\\
-\nabla\mathbb I & 0 & 0 & 0
\end{pmatrix},
$$

and

$$
A_3=
\begin{pmatrix}
0 & 0 & 0 & 0\\
0 & 0 & -\nabla_Y & 0\\
0 & -\operatorname{div}_Y & 0 & 0\\
0 & 0 & 0 & 0
\end{pmatrix}.
$$

For each block one needs to compute the resolvent

$$
J_{\lambda A_i}:=(\operatorname{id}+\lambda A_i)^{-1}.
$$

Given an input

$$
\bar z=(\bar P,\bar Q,\bar f,\bar g),
$$

the resolvent output is denoted

$$
z=(P,Q,f,g).
$$

---

## 3. Generic two-variable resolvent identity

The notes use the following elementary identity repeatedly. Consider

$$
\begin{pmatrix}
\operatorname{id} & L^*\\
-L & \operatorname{id}
\end{pmatrix}
\begin{pmatrix}
x\\y
\end{pmatrix}
=
\begin{pmatrix}
\bar x\\\bar y
\end{pmatrix}.
$$

Then the system may be solved in either of the two equivalent ways:

$$
\begin{cases}
x=\bar x-L^*y,\\
y=(\operatorname{id}+LL^*)^{-1}(\bar y+L\bar x),
\end{cases}
$$

or

$$
\begin{cases}
x=(\operatorname{id}+L^*L)^{-1}(\bar x-L^*\bar y),\\
y=\bar y+Lx.
\end{cases}
$$

With a resolvent step size $\lambda$, the operator $L$ is effectively replaced by $\lambda L$, which is why the linear systems below contain terms of order $\lambda^2$.

---

## 4. Resolvent of $A_1$

The $A_1$ block couples $P$ and $f$ through $\mathbb J$ and the spatial gradient.

The resolvent equation is

$$
(\operatorname{id}+\lambda A_1)z=\bar z,
$$

i.e.

$$
\begin{pmatrix}
\operatorname{id} & 0 & -\lambda\mathbb J^*\operatorname{div}_\Omega & 0\\
0 & \operatorname{id} & 0 & 0\\
-\lambda\nabla_\Omega\mathbb J & 0 & \operatorname{id} & 0\\
0 & 0 & 0 & \operatorname{id}
\end{pmatrix}
\begin{pmatrix}
P\\Q\\f\\g
\end{pmatrix}
=
\begin{pmatrix}
\bar P\\\bar Q\\\bar f\\\bar g
\end{pmatrix}.
$$

Thus

$$
\begin{cases}
P-\lambda\mathbb J^*\operatorname{div}_\Omega f=\bar P,\\
f-\lambda\nabla_\Omega\mathbb J P=\bar f,\\
Q=\bar Q,\\
g=\bar g.
\end{cases}
$$

Substituting the second equation into the first gives

$$
\left(\operatorname{id}-\lambda^2\mathbb J^*\Delta_\Omega\mathbb J\right)P
=
\bar P+\lambda\mathbb J^*\operatorname{div}_\Omega\bar f.
$$

Therefore

$$
\boxed{
P=
\left(\operatorname{id}-\lambda^2\mathbb J^*\Delta_\Omega\mathbb J\right)^{-1}
\left(\bar P+\lambda\mathbb J^*\operatorname{div}_\Omega\bar f\right)
}
$$

and then

$$
\boxed{
f=\bar f+\lambda\nabla_\Omega\mathbb J P,
\qquad
Q=\bar Q,
\qquad
g=\bar g.
}
$$

---

## 5. Resolvent of $A_2$

The $A_2$ block couples $P$ and $g$ through $\mathbb I$.

The resolvent equations are

$$
\begin{cases}
P-\lambda\mathbb I^*\operatorname{div}g=\bar P,\\
g-\lambda\nabla\mathbb I P=\bar g,\\
Q=\bar Q,\\
f=\bar f.
\end{cases}
$$

Substitution gives

$$
\left(\operatorname{id}-\lambda^2\mathbb I^*\Delta\mathbb I\right)P
=
\bar P+\lambda\mathbb I^*\operatorname{div}\bar g.
$$

Therefore

$$
\boxed{
P=
\left(\operatorname{id}-\lambda^2\mathbb I^*\Delta\mathbb I\right)^{-1}
\left(\bar P+\lambda\mathbb I^*\operatorname{div}\bar g\right)
}
$$

and then

$$
\boxed{
g=\bar g+\lambda\nabla\mathbb I P,
\qquad
Q=\bar Q,
\qquad
f=\bar f.
}
$$

---

## 6. Resolvent of $A_3$

The $A_3$ block couples $Q$ and $f$ through the $Y$-gradient.

The resolvent equations are

$$
\begin{cases}
Q-\lambda\nabla_Y f=\bar Q,\\
f-\lambda\operatorname{div}_Y Q=\bar f,\\
P=\bar P,\\
g=\bar g.
\end{cases}
$$

Hence

$$
Q=\bar Q+\lambda\nabla_Y f.
$$

Substitution gives

$$
\left(\operatorname{id}-\lambda^2\Delta_Y\right)f
=
\bar f+\lambda\operatorname{div}_Y\bar Q.
$$

Therefore

$$
\boxed{
f=\left(\operatorname{id}-\lambda^2\Delta_Y\right)^{-1}
\left(\bar f+\lambda\operatorname{div}_Y\bar Q\right)
}
$$

and

$$
\boxed{
Q=\bar Q+\lambda\nabla_Y f,
\qquad
P=\bar P,
\qquad
g=\bar g.
}
$$

> **Transcription note.** On page 2 of the handwritten notes the $A_3$ formula appears to contain $\operatorname{id}-\lambda\Delta_Y$. By the same algebra used for $A_1$ and $A_2$, and from the displayed equations $Q=\bar Q+\lambda\nabla_Yf$, the coefficient should be $\lambda^2$. I have therefore recorded the consistent formula $\operatorname{id}-\lambda^2\Delta_Y$.

A crossed-out line in the notes also mentions the optional positivity projection

$$
P=\operatorname{proj}_{\ge 0}(\bar P),
$$

but it is not part of the displayed $A_3$ block computation.

---

## 7. Resolvent of $A_0$

The $A_0$ block is diagonal, so its resolvent is given componentwise by proximal maps.

### 7.1. $P$-component: fidelity term

For

$$
F_1(P)=\frac12\|KP-E\|^2,
$$

one has

$$
\operatorname{prox}_{\delta F_1}(P)
=
\left(\operatorname{id}+\delta K^*K\right)^{-1}
\left(P+
\delta K^*E\right).
$$

Since $K=UF$, this can be computed in the Fourier domain:

$$
\boxed{
\operatorname{prox}_{\delta F_1}(P)
=F^*\left(\operatorname{id}+\delta U^*U\right)^{-1}
F\left(P+
\delta K^*E\right).
}
$$

If $U$ is a sampling mask, then $U^*U$ is diagonal/a coordinate projection, so
$(\operatorname{id}+\delta U^*U)^{-1}$ is pointwise in the Fourier domain.

### 7.2. $Q$-component: $\ell^{2,1}$ transport cost

For

$$
F_2(Q)=\alpha_1\|Q\|_{2,1},
$$

one has

$$
\operatorname{prox}_{\lambda F_2}(Q)
=
\operatorname{prox}_{\lambda\alpha_1\|\cdot\|_{2,1}}(Q).
$$

Blockwise, for each vector block $Q_i$,

$$
\boxed{
\left[\operatorname{prox}_{\lambda\alpha_1\|\cdot\|_{2,1}}(Q)\right]_i
=
\left(1-\frac{\lambda\alpha_1}{\|Q_i\|_2}\right)_+Q_i.
}
$$

### 7.3. $g$-component: projection onto $\alpha_2B^\infty$

For

$$
G_2(g)=\mathcal I_{\alpha_2B^\infty}(g),
$$

one has

$$
\boxed{
\operatorname{prox}_{\lambda G_2}(g)
=\operatorname{proj}_{\alpha_2B^\infty}(g).
}
$$

This is the pointwise projection/clipping onto the $\ell^\infty$ ball of radius $\alpha_2$.

### 7.4. $f$-component: gauge fixing and bounded surrogate version

The notes first write

$$
G_1(f)=\mathcal I_{\{0\}}(Sf),
$$

hence

$$
\boxed{
\operatorname{prox}_{\lambda G_1}(f)=\operatorname{proj}_{\ker S}(f).
}
$$

In words: this sets the $Y$-centres of $f$ to zero.

The grey/red correction in the notes then considers the bounded version

$$
G_1(f)=\mathcal I_{\{0\}}(Sf)+\mathcal I_{C_fB^\infty}(f),
$$

which gives

$$
\boxed{
\operatorname{prox}_{\lambda G_1}(f)
=\operatorname{proj}_{\ker S\cap C_fB^\infty}(f).
}
$$

---

# 8. Linear systems needed in the resolvents

The last two pages of the handwritten notes derive efficient inverses for the linear systems appearing in the $A_1$, $A_2$, and $A_0$ resolvents.

---

## 8.1. Inverting $\operatorname{id}-\delta\mathbb J^*\Delta_\Omega\mathbb J$

The goal is to invert

$$
\operatorname{id}-\delta\mathbb J^*\Delta_\Omega\mathbb J.
$$

The notes use the tensor-product structure

$$
\mathbb J=\operatorname{id}_{n_x}\otimes J,
\qquad
\Delta_\Omega=\Delta\otimes\operatorname{id}_{n_y}.
$$

Therefore

$$
\mathbb J^*\Delta_\Omega\mathbb J
=
(\operatorname{id}_{n_x}\otimes J^*)
(\Delta\otimes\operatorname{id}_{n_y})
(\operatorname{id}_{n_x}\otimes J)
=
\Delta\otimes J,
$$

where the last equality uses the projection properties of $J$.

Thus

$$
\left(\operatorname{id}-\delta\mathbb J^*\Delta_\Omega\mathbb J\right)^{-1}
=
\left(\operatorname{id}-\delta\Delta\otimes J\right)^{-1}.
$$

Define the projection onto constants in $Y$ by

$$
\mathsf P:=\frac1{n_y}I^*I,
$$

so that

$$
\operatorname{Im}\mathsf P=\operatorname{span}\mathbf 1_y,
\qquad
\mathbf 1_y=(1,\ldots,1),
$$

and

$$
J=\operatorname{id}-\mathsf P,
\qquad
\operatorname{Im}J=\mathbf 1_y^\perp.
$$

Lift these projections to the full space by

$$
\mathbb P:=\operatorname{id}_{n_x}\otimes\mathsf P,
\qquad
\mathbb J:=\operatorname{id}_{n_x}\otimes J.
$$

Then

$$
\operatorname{Im}\mathbb P
=\mathbb R^{n_x}\otimes\operatorname{span}\mathbf 1_y,
$$

which is the subspace of fields that are constant in $Y$, while

$$
\operatorname{Im}\mathbb J
=\mathbb R^{n_x}\otimes\mathbf 1_y^\perp,
$$

which is the subspace of fields with zero sum in $Y$.

On $\operatorname{Im}\mathbb P$,

$$
\operatorname{id}-\delta\Delta\otimes J
=
\operatorname{id},
$$

because $J=0$ on constants in $Y$.

On $\operatorname{Im}\mathbb J$,

$$
\operatorname{id}-\delta\Delta\otimes J
=
\operatorname{id}-\delta\Delta\otimes\operatorname{id}_{n_y}
=
\operatorname{id}-\delta\Delta_\Omega.
$$

Therefore, to solve

$$
(\operatorname{id}-\delta\Delta\otimes J)z=c,
$$

split

$$
z=z_{\mathbb P}+z_{\mathbb J},
\qquad
c=c_{\mathbb P}+c_{\mathbb J},
$$

with

$$
c_{\mathbb P}=\mathbb P c,
\qquad
c_{\mathbb J}=\mathbb J c.
$$

Then

$$
z_{\mathbb P}=c_{\mathbb P},
\qquad
(\operatorname{id}-\delta\Delta_\Omega)z_{\mathbb J}=c_{\mathbb J}.
$$

Hence the final formula is

$$
\boxed{
z=z_{\mathbb P}+z_{\mathbb J},
\qquad
z_{\mathbb P}=\mathbb P c,
\qquad
z_{\mathbb J}=(\operatorname{id}-\delta\Delta_\Omega)^{-1}\mathbb Jc.
}
$$

---

## 8.2. Inverting $\operatorname{id}-\delta\mathbb I^*\Delta\mathbb I$

The next system is

$$
\operatorname{id}-\delta\mathbb I^*\Delta\mathbb I.
$$

The tensor-product structure is

$$
\mathbb I=\operatorname{id}_{n_x}\otimes I.
$$

Hence

$$
\mathbb I^*\Delta\mathbb I
=
(\operatorname{id}_{n_x}\otimes I^*)
(\Delta\otimes\operatorname{id})
(\operatorname{id}_{n_x}\otimes I)
=
\Delta\otimes I^*I.
$$

Since

$$
I^*I=n_y\mathsf P,
$$

we get

$$
\mathbb I^*\Delta\mathbb I
=n_y\Delta\otimes\mathsf P.
$$

Thus:

- on $\operatorname{Im}\mathbb P$, it acts as $n_y\Delta\otimes\operatorname{id}_{n_y}$;
- on $\operatorname{Im}\mathbb J$, it acts as $0$.

Now solve

$$
(\operatorname{id}-\delta\mathbb I^*\Delta\mathbb I)z=c.
$$

Using the decomposition

$$
z=z_{\mathbb P}+z_{\mathbb J},
\qquad
c=c_{\mathbb P}+c_{\mathbb J},
$$

one obtains immediately

$$
z_{\mathbb J}=c_{\mathbb J}=\mathbb Jc.
$$

On $\operatorname{Im}\mathbb P$ one writes

$$
c_{\mathbb P}=\tilde c_{\mathbb P}\otimes\mathbf 1_y,
\qquad
\tilde c_{\mathbb P}=\frac1{n_y}\mathbb I c.
$$

Similarly,

$$
z_{\mathbb P}=\tilde z_{\mathbb P}\otimes\mathbf 1_y.
$$

Then

$$
(\operatorname{id}-\delta n_y\Delta\otimes\operatorname{id}_{n_y})
(\tilde z_{\mathbb P}\otimes\mathbf 1_y)
=
\tilde c_{\mathbb P}\otimes\mathbf 1_y,
$$

or equivalently

$$
(\tilde z_{\mathbb P}-\delta n_y\Delta\tilde z_{\mathbb P})
\otimes\mathbf 1_y
=
\tilde c_{\mathbb P}\otimes\mathbf 1_y.
$$

Thus

$$
\tilde z_{\mathbb P}
=(\operatorname{id}-\delta n_y\Delta)^{-1}
\frac1{n_y}\mathbb I c,
$$

and therefore

$$
z_{\mathbb P}
=
\left[
(\operatorname{id}-\delta n_y\Delta)^{-1}
\frac1{n_y}\mathbb I c
\right]
\otimes\mathbf 1_y.
$$

The final formula is

$$
\boxed{
z=z_{\mathbb P}+z_{\mathbb J},
\qquad
z_{\mathbb P}
=
\left[
(\operatorname{id}-\delta n_y\Delta)^{-1}
\frac1{n_y}\mathbb I c
\right]
\otimes\mathbf 1_y,
\qquad
z_{\mathbb J}=\mathbb Jc.
}
$$

---

## 8.3. Inverting $\operatorname{id}+\delta K^*K$

Finally, for the fidelity proximal step, one needs

$$
\operatorname{id}+\delta K^*K.
$$

Since

$$
K=UF,
$$

we have

$$
K^*K=F^*U^*UF.
$$

Therefore

$$
\operatorname{id}+\delta K^*K
=\operatorname{id}+\delta F^*U^*UF
=F^*(\operatorname{id}+\delta U^*U)F.
$$

Consequently,

$$
\boxed{
(\operatorname{id}+\delta K^*K)^{-1}
=F^*(\operatorname{id}+\delta U^*U)^{-1}F.
}
$$

This is the formula used in the $F_1$ proximal step.

---

## 9. Implementation-level summary

Given an input iterate/block variable

$$
\bar z=(\bar P,\bar Q,\bar f,\bar g),
$$

the resolvents are:

### $J_{\lambda A_1}$

$$
\begin{aligned}
P&=(\operatorname{id}-\lambda^2\mathbb J^*\Delta_\Omega\mathbb J)^{-1}
(\bar P+\lambda\mathbb J^*\operatorname{div}_\Omega\bar f),\\
f&=\bar f+\lambda\nabla_\Omega\mathbb JP,\\
Q&=\bar Q,\\
g&=\bar g.
\end{aligned}
$$

### $J_{\lambda A_2}$

$$
\begin{aligned}
P&=(\operatorname{id}-\lambda^2\mathbb I^*\Delta\mathbb I)^{-1}
(\bar P+\lambda\mathbb I^*\operatorname{div}\bar g),\\
g&=\bar g+\lambda\nabla\mathbb IP,\\
Q&=\bar Q,\\
f&=\bar f.
\end{aligned}
$$

### $J_{\lambda A_3}$

$$
\begin{aligned}
f&=(\operatorname{id}-\lambda^2\Delta_Y)^{-1}
(\bar f+\lambda\operatorname{div}_Y\bar Q),\\
Q&=\bar Q+\lambda\nabla_Yf,\\
P&=\bar P,\\
g&=\bar g.
\end{aligned}
$$

### $J_{\lambda A_0}$

$$
\begin{aligned}
P&=\operatorname{prox}_{\lambda F_1}(\bar P),\\
Q&=\operatorname{prox}_{\lambda F_2}(\bar Q),\\
f&=\operatorname{prox}_{\lambda G_1}(\bar f),\\
g&=\operatorname{prox}_{\lambda G_2}(\bar g).
\end{aligned}
$$

with

$$
\operatorname{prox}_{\lambda F_1}(P)
=(\operatorname{id}+\lambda K^*K)^{-1}(P+\lambda K^*E),
$$

$$
\operatorname{prox}_{\lambda F_2}(Q)
=\operatorname{prox}_{\lambda\alpha_1\|\cdot\|_{2,1}}(Q),
$$

$$
\operatorname{prox}_{\lambda G_1}(f)
=\operatorname{proj}_{\ker S}(f)
\quad\text{or}\quad
\operatorname{proj}_{\ker S\cap C_fB^\infty}(f),
$$

and

$$
\operatorname{prox}_{\lambda G_2}(g)
=\operatorname{proj}_{\alpha_2B^\infty}(g).
$$

---

## 10. Checklist of delicate points

1. **Signs of the gradient/divergence blocks.**  
   The handwritten optimality conditions and the resolvent derivations should be kept under the same convention. The formulas in this file follow the convention inferred from the resolvent derivations.

2. **The $A_3$ coefficient.**  
   The consistent formula is $\operatorname{id}-\lambda^2\Delta_Y$, not $\operatorname{id}-\lambda\Delta_Y$, assuming the equations are $Q=\bar Q+\lambda\nabla_Yf$ and $f=\bar f+\lambda\operatorname{div}_YQ$ in the same convention.

3. **The projections $\mathbb P$ and $\mathbb J$.**  
   They split each $Y$-profile into its constant-in-$Y$ component and its zero-sum component. This is what makes the inverses involving $\mathbb I$ and $\mathbb J$ cheap.

4. **The Fourier inversion for the fidelity prox.**  
   Since $K=UF$, the fidelity prox is diagonal after applying $F$.

5. **The optional positivity projection for $P$.**  
   The handwritten note contains a crossed-out $P=\operatorname{proj}_{\ge0}(\bar P)$. It should only be included if the final algorithm explicitly incorporates the constraint $P\ge0$ in the corresponding block.
