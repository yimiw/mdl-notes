#import "../assets/tmp_sht.typ": *
#show: project.with(authors: ((name: "", email: ""),))

// ========== 参数配置 ==========
#let fsize = 9.5pt
#let hsize1 = 10pt
#let hsize2 = 7.5pt
#let pspace = 0.2em
#let plead = 0.3em
// ===============================

#set text(size: fsize)
#set par(spacing: pspace, leading: plead, justify: true, first-line-indent: 0em)
#show heading.where(level: 1): set text(size: hsize1)
#show heading.where(level: 2): set text(size: hsize2)
#show heading: box
#show heading: set text(fill: rgb("#663399"), weight: "bold")

#show: columns.with(4, gutter: 0.5em)


// #colbreak()

= 1. Adversarial Attacks

#cbox(title: [Targeted FGSM])[
  $x' = x - eta$, $eta = epsilon dot "sign"(nabla_x "loss"_t (x))$
  
  Guarantees $eta in [-epsilon, epsilon]$, $eta$ not minimized
]

#cbox(title: [Untargeted FGSM])[
  $x' = x + eta$, $eta = epsilon dot "sign"(nabla_x "loss"_s (x))$
  
  Guarantees $eta in [-epsilon, epsilon]$, $eta$ not minimized
]

#cbox(title: [Carlini & Wagner])[
  Find targeted adv. sample $x' = x + eta$ _and_ minimize $||eta||_p$ via minimizing $||eta||_p + c dot "obj"_t (x')$, where
  $"obj"_t$ is s.t. $"obj"_t (x') <= 0 arrow.r.double f(x') = t$,
  e.g. $"CE"(x', t) - 1$; $max(0, 0.5 - p_f (x')_t)$
  
  Prior e.g. $x + eta in [0, 1]$: use specialized optimizer (LBFGS-B) or PGD.
  Optimizing $||eta||_infinity$ is hard, use $"ReLU"(sum |eta_i| - tau)$, lower $tau$ gradually.
]

#cbox(title: [PGD])[
  Repeat FGSM with $epsilon_"step"$ and proj. to $x plus.minus epsilon$.
]

= 2. Adversarial Defenses

#cbox(title: [Defense as Optimization])[
  $min_theta EE_((x,y) tilde D) [ max_(x' in S(x)) L(theta, x', y) ]$
  
  usually $S(x) = BB^infinity_epsilon$, $EE approx$ empirical risk.
]

#cbox(title: [PGD Defense algorithm])[
  Run PGD on every batch and use $nabla_theta cal(L)(x_"adv")$ for backprop.
]

#cbox(title: [TRADES defense])[
  $min_theta EE_((x,y) tilde D) [L(theta, x, y) + lambda max_(x' in BB_epsilon (x)) L(theta, x', y)]$
]

= 3. Certification of Neural Networks

Given NN N, precond. $phi$, postcond. $psi$ prove: $forall i quad i models phi arrow.r.double N(i) models psi$ or return a violation.

== 3.1 Complete Methods (always return result)

#cbox(title: [MILP Encoding])[
  Encode NN as MILP instance. Doesn't scale well.
  
  - Affine: $y = W x + b$ is a direct MILP constraint.
  $W x + b <= y <= W x + b$.
  
  - $"ReLU"(x)$: $y <= x - l_x dot (1-a)$, $y >= x$, $y <= u_x dot a$,
  $y >= 0$, $a in {0,1}$, for box bound $x in [l,u]$.
  
  - $a = 0$: $y = 0, x in [l,0]$
  - $a = 1$: $y = x, y in [0,u]$
  
  To check an encoding for $f$, plot constraint regions for all cases of int. variables.
  They should match plot of $f$. Can't use $a dot x$.
  
  $phi = BB^infinity_epsilon (x)$: $x_i - epsilon <= x_i' <= x_i + epsilon, forall i$
  
  precomp. Box bounds: $l_i <= x_i^p <= u_i$
  
  $psi = o_0 > o_1$: MILP objective $min o_0 - o_1$.
]

== 3.2 Incomplete Methods (may abstain)

#cbox(title: [Box])[
  $cal(O)(n^2 L)$: Bounds are $l_infinity$ balls.
  
  $[a,b] +^\# [c,d] = [a+b, c+d]$, $-^\# [a,b] = [-b,-a]$;
  
  $"ReLU"^\# [a,b] = ["ReLU"(a), "ReLU"(b)]$;
  
  $lambda dot^\# [a,b] = [lambda a, lambda b]$ ($lambda >= 0$)
]

#cbox(title: [DeepPoly])[
  $cal(O)(n^3 L^2)$: For each $x_i$ keep constraints:
  
  interval $l_i <= x_i$, $x_i <= u_i$;
  
  relational $a_i^(<= ) <= x_i$, $x_i <= a_i^(>= )$ where $a_i^(<= ), a_i^(>= )$ are of the form $sum_j w_j dot x_j + nu$
  
  - $x_j = "ReLU"^\# (x_i)$: interval constr. $x_i in [l_i, u_i]$:
  
  $u_i <= 0$: $a_j^(<= ) = a_j^(>= ) = 0, l_j = u_j = 0$;
  
  $l_i >= 0$: $a_j^(<= ) = a_j^(>= ) = x_i, l_j = l_i, u_j = u_i$;
  
  $l_i < 0, u_i > 0$: $lambda := u_i \/ (u_i - l_i), x_j <= lambda (x_i - l_i),$
  $alpha in [0, 1], alpha x_i <= x_j, l_j = 0, u_j = u_i$.
  
  Min area: if $u <= - l, alpha = 0$, otherwise $1$.
  
  When proving $y_2 > y_1$, add a layer that computes $y_2 - y_1$ and prove $l_(y_2 - y_1) > 0$.
]

#cbox(title: [Branch & Bound])[
  Split ReLU based on $x_i <= 0$, resulting bound is the worst of two cases.
  Naive split still covers extra space, need constraints. KKT:
  $(max f(x) mid g(x) <= 0) <= max_x min_beta f(x) - beta g(x)$
  
  - $(max_x arrow(a) arrow(x) + c "s.t." -x_i <= 0) <= max_x min_beta arrow(a) arrow(x) + c + beta x_i$
  - $(max_x arrow(a) arrow(x) + c "s.t." x_i <= 0) <= max_x min_beta arrow(a) arrow(x) + c - beta x_i$
  
  Usually you use the weak duality after this. $beta$ is found by GD, and on each step you do full backsubstitution after the split, as the sign in front of symbolic variables can change when $beta$ changes.
]

= 4. Certified Defenses

Produces models that are easier to certify.

== 4.1 DiffAI

#cbox(title: [PGD])[
  $min EE_((x,y) tilde D) [ max_(z in gamma("NN"^\# (S(x)))) L(theta, z, y) ]$
  
  Can use any abstract transformer (Box, DeepPoly).
  
  To find max loss, use abstract loss $L^\# (arrow(z), y)$, where $y = $ target label, $arrow(z) = $ vector of logits:
  
  - $L(z, y) = max_(q != y) (z_q - z_y)$: Compute $d_c = z_c - z_y$ $forall c in cal(C)$, where $z_c$ the abstract logit shape of class $i$. Then compute box bounds of $d_c$ and compute max upper bound: $max_(c in cal(C))(max("box"(d_c)))$
  
  - $L(z,y) = "CE"(z,y)$: Compute box bounds $[l_c, u_c]$ of $z_c$. $forall c in cal(C)$ pick $u_c$ if $c != y$, pick $l_c$ if $c = y$, hence $v = [u_1,.., l_c,.., u_(|cal(C)|)]$.
  Compute $"CE"("softmax"(v), y)$.
]

== 4.2 COLT

#cbox(title: [COLT])[
  Run relaxation up to some layer: $S' = "NN"^\#_(1 dots.h i) (S(x))$, then run PGD on the region
  to train layers $i + 1 dots.h n$.
  For PGD we need to project back to $S'$, which is not efficient for DeepPoly.
]

= 5. Randomized Smoothing for Robustness

#cbox(title: [Smoothed Classifier])[
  Given any classifier $f$, make a smoothed classifier
  $g(x) := arg max_(c_A in Y) PP_epsilon (f(x + epsilon) = c_A)$,
  where $epsilon tilde cal(N)(0, sigma I)$, $p_A (x)$ is the probability under argmax.
  
  If $exists underline(p_(A, x)), overline(p_(B, x)) in [0, 1]$ s.t.
  $p_A (x) >= underline(p_(A, x)) >= overline(p_(B, x)) >=$ $max_(B != A) p_B (x)$,
  then $g(x + delta) = c_A quad forall ||delta||_2 < R_x$ aka
  certification radius $ = delta \/ 2 (Phi^(-1)(underline(p_(A, x))) - Phi^(-1)(overline(p_(B, x))))$.
  
  Calculating $p_(A, x), p_(B, x)$ directly is hard, so we use bounds.
  Calculating $overline(p_(B, x))$ is also hard, so let's assume $underline(p_(A, x)) > 0.5$, then $overline(p_(B, x)) = 1 - underline(p_(A, x))$.
]

#cbox(title: "Certification")[
  $hat(c_A) arrow.l "guess_top_class"(f, sigma, x, n_0)$
  
  $underline(p_A) arrow.l "lower_bound_p"(hat(c_A), f, sigma, x, n, alpha)$
  
  *If* $underline(p_A) > 0.5$:
  
  #h(1em) $R arrow.l sigma Phi^(-1)(underline(p_A))$
  
  #h(1em) *Return* $hat(c_A), R$
  
  *Else*:
  
  #h(1em) *Return* ABSTAIN
]

#cbox(title: [Notes])[
  Top class is estimated via Monte-Carlo.
  Lower bound is estimated by CLT, Chebyshev's inequality or binomial confidence bounds.
  The two function calls involve sampling, the samples should be separate, and $n >> n_0$.
  If the algorithm returns ABSTAIN, one of the following is true:
  - $hat(c_A)$ is wrong, fixed by increasing $n_0$
  - True $p_A <= 0.5$, unfixable
  - Lower bound is too low, fixed by increasing $n$
]

#cbox(title: "Inference")[
  $hat(c_A), n_A, hat(c_B), n_B arrow.l "top_two_classes"(f, sigma, x, n)$
  
  *Return* BinomPValue$(n_A, n_A + n_B, = , 0.5) <= alpha$ ? $hat(c_A)$ : ABSTAIN
]

#cbox(title: [Hypothesis Testing])[
  - NH: true $p("success")$ of $f$ returning $hat(c)_A$ is $0.5$
  - BinomPValue returns $p$-value of null hypothesis, evaluated on $n$ iid samples with $i$ successes.
  - Accept NH if $p$-value is $> alpha$, reject otherwise.
  - $alpha$ small: often accept null hypothesis and ABSTAIN, but more confident in predictions.
  - $alpha$ large: more predictions but more mistakes.
  - Returns wrong class with probability at most $alpha$
]

= 6. Privacy

Common attacks: Model Stealing, Model Extraction (representative inputs),
Data Extraction (exact training samples), Membership Inference (find out if a sample was used for training).

Black-Box MI: Attacker trains many models on the same data distribution, some with entry $x$, some without.
If logits are given, then attacker trains a classifier to distinguish between the two cases.
If not, then do the same with robustness scores.

== 6.1 Federated Learning

#cbox(title: [FedSGD])[
  Entities do training steps on minibatches ${x^k, y^k}$ from private data $cal(D)_k$
  and return gradients $g_k := nabla_theta cal(L)(f_(theta_t) (x^k), y^k)$,
  average on server
  and update the global model $theta_(t+1) := theta_t - gamma g_c$.
  But sent data still contains information about private data.
]

#cbox(title: [Honest but curious server])[
  Server does not manipulate sent weights.
  For batch size 1 and piecewise linear activation functions, the server can learn the data exactly.
  For batch size $> 1$ and some assumptions, a linear combination of some true inputs can be found.
  The general approach is:
  $arg min_(x^*) d(g_k, nabla_theta cal(L)(f_(theta_t) (x^*), y^*)) + alpha_"reg" dot cal(R)(x^*)$
  - $d$ is distance, typically $l_1$, $l_2$ or cosine.
  - $cal(R)$ is a prior based on domain-specific knowledge.
  - Optimization is done via GD.
  - $y^*$ is recovered separately (out of scope).
  - For each categorical feature create an $N$-dim. variable that gets put into $x^*$
  through softmax.
  
  For tables, we can use entropy over many randomly initialized reconstructions as a prior,
  because correct cells are robust to random initializations.
]

#cbox(title: [FedAVG])[
  Client runs $E$ epochs of SGD, sends new weights to server.
  Final weights depend on order of batches, the server does not know it.
  Attack simulates training.
  Prior: the average of samples in one epoch is equal to that in another epoch.
]

== 6.2 Differential Privacy

#cbox(title: [MI protection])[
  $PP(M(cal(D)) in S) approx PP(M(cal(D)') in S)$
]

#cbox(title: [$epsilon$-DP])[
  $M$ is $epsilon$-DP if for all "neighboring" $(a, a')$
  and for any attack $S$
  $p(a) := PP(M(a) in S) <= e^epsilon PP(M(a') in S)$.
  
  As $e^epsilon approx 1 + epsilon$,
  $(1 - epsilon) p(a') approx p(a) approx (1 + epsilon) p(a')$.
  
  By a theorem, $f(a) + "Lap"(0, Delta_1 \/ epsilon)$ is $epsilon$-DP,
  where $Delta_p := max_((a, a') in "Neigh") ||f(a) - f(a')||_p$.
]

#cbox(title: [$epsilon, delta$-DP])[
  $M$ is $epsilon, delta$-DP iff
  $PP(M(a) in S) <= e^epsilon PP(M(a') in S) + delta$
  $forall (a, a') in "Neigh", forall S$. This allows absolute differences (not only relative).
  If $p(a') = 0$, $p(a) != 0$, no $epsilon$-DP mechanism exists, but $epsilon, delta$-DP might.
  
  If output set is discrete, singleton attacks are enough.
  $f(a) + cal(N)(0, sigma^2 I)$ is $epsilon, delta$-DP,
  where $sigma = sqrt(2 log(1.25) \/ delta) dot Delta_2 \/ epsilon$.
]

#cbox(title: [Composition])[
  If $M_1, M_2$ are $epsilon_1, delta_1$-DP and $epsilon_2, delta_2$-DP,
  then $(M_1, M_2)$ and $M_1 compose M_2$ are $epsilon_1 + epsilon_2, delta_1 + delta_2$-DP.
  In particular, if $f$ is a plain function $(0, 0)$-DP, then $f compose M$ is $epsilon, delta$-DP.
  If $A_i$ has user data and $M_i$ is $(epsilon_i, delta_i)$-DP, $M_1 (a_1) dots.h M_k (a_k)$ is $(max_i epsilon_i, max_i delta_i)$-DP.
]

#cbox(title: [DP-SGD])[
  Project gradients for each point onto $l_2$-ball of size $C$ and sum them up.
  Add $cal(N)(0, sigma^2 I)$ to the batch gradient, where
  $sigma = sqrt(2 log(1.25) \/ delta) dot C \/ L \/ epsilon$
  The resulting model is private, even against a white-box attacker with any number of queries.
  Clipping is necessary to bound the sensitivity of the gradient.
]

#cbox(title: [Privacy Amplification])[
  Applying an $(epsilon, delta)$-DP mechanism on a random fraction
  $q = L \/ N$ subset yields a $(tilde(q) epsilon, q delta)$-DP mechanism, where $tilde(q) approx q$.
  
  Due to clipping, sensitivity of the gradient for any point is $C$.
  If $T = 1$ and no subsampling is used, adding/removing a datapoint changes total gradient by at most $C \/ L$.
  Then by the gaussian mechanism the resulting model is $epsilon, delta$-DP.
  If subsampling is used, by privacy amplification, the model is $(tilde(q) epsilon, q delta)$-DP.
  If $T != 1$, by the composition theorem, the model is $(tilde(q) T epsilon, q T delta)$-DP.
  By out of scope theorems, this is
  $(cal(O)(q epsilon sqrt(T log frac(1, delta))), cal(O)(q T delta))$ and
  $(cal(O)(q epsilon sqrt(T)), delta)$-DP.
]

#cbox(title: [PATE: Private Aggregation of Teacher Models])[
  Split data into disjoint partitions and train a model for each.
  Agreggate models via noisy voting into a teacher, which labels public unlabeled data,
  on which we train the final model.
  
  $T$ are teachers, $n_j (x) := |{t(x) = j mid t in T}|$.
  $arg max(n_j (x)) + "Lap"(0, sigma)$ is bad, better
  $arg max(n_j (x) + "Lap"(0, 2 \/ epsilon))$.
  $Delta_1 = 2 arrow.r.double$ model is $(epsilon, 0)$-DP for one query.
  Labeling $T$ data points yields $(epsilon T, 0)$-DP.
  But there are better bounds.
]

#cbox(title: [FedSGD/FedAVG with Noise])[
  clip the gradients/weights and add noise.
  
  DP is closely related to randomized smoothing. We add noise to data, then forward is $epsilon$-DP.
]

= 7. AI Regulation

Key issues: fairness, explainability, data minimization, unlearning (right to be forgotten), copyright.

= 8. Private synthetic data

Data is private, make DP synthetic proxy.

1. *Select* marginal queries we want to measure
2. *Measure* marginal queries using DP
3. *Generate* synthetic data

#cbox(title: [Marginal])[
  *Marginal* on $C subset.eq cal(A)$ (attrs.) is a vector $mu in RR^(n_C)$,
  indexed by $t in Omega_C$, where $Omega_C = product_(i in C) Omega_i$ and $n_C = |Omega_C|$.
  Each entry $mu_t$ is a count $sum_(x in D) [x_C = t]$.
  $M_C : cal(D) arrow RR^(n_C), D arrow.bar mu$ computes the marginal.
  
  $Delta_2 (M_C) = 1$ because adding a row in a dataset can only
  change one element of the vector.
  1-way marginals ($n_C = 1$) are histograms, 2-way marginals are heatmaps.
]

#cbox(title: [Chow-Liu])[
  Mutual information of two variables $X, Y$ is $"I"(X, Y) = sum_(x, y) frac(p(x, y), p(x) p(y))$.
  Chow-Liu algorithm makes a complete graph of features, edge weigths $"I"(X, Y)$.
  Find MST, the optimal 2nd-order approximation.
  Generate by sampling from MST, each node is conditioned on its parent, i.e.
  $p(F_1 = f_1, F_2 = f_2, F_3 = f_3) = p(F_1 = f_1) p(F_2 = f_2 mid F_1 = f_1) p(F_3 = f_3 mid F_1 = f_1)$,
  if $F_1$ is parent of $F_2$ and $F_3$.
  
  Add DP, i.e. add noise to every step of the algorithm.
  MST is done with the exponential mechanism, marginals are measured with Gaussian noise.
]

#cbox(title: [ProgSyn])[
  - Sample random noise $z tilde cal(N)(0, I_p)$
  - Pass $z$ through a generative model $g_theta$
  - Get synthetic dataset $g_theta (z)$
  - Adapt $theta$ to make $g_theta (z)$ close to original $X$
  - Fine-tune $g_theta$ to make $g_theta (z)$ satisfy constraints
]

= 9. Logic and Deep Learning (DL2)

== 9.1 Querying Neural Networks

#cbox(title: [Standard Logic])[
  Use standard logic ($forall, exists, and, or, f: RR^m arrow RR^n,..$) and high-level queries to impose constraints.
  
  $("class"("NN"(i)) = 9) = and.big_(j=1,j != 9)^k "NN"(i)[j] < "NN"(i)[9]$
  
  Use translation $T$ of logical formulas into differentiable loss function $T(phi)$ to be solved with gradient-based optimization to minimize $T(phi)$. Regular SAT solvers can't handle non-small NNs.
]

#cbox(title: [Theorem])[
  $forall x, T(phi)(x) = 0 arrow.l.r.double x models phi$
]

#cbox(title: [Logical Formula to Loss])[
  #set text(size: 7.5pt)
  #table(
    columns: 2,
    [*Logical Term*], [*Loss*],
    [$t_1 <= t_2$], [$max(0, t_1 - t_2)$],
    [$t_1 != t_2$], [$[t_1 = t_2]$],
    [$t_1 = t_2$], [$T(t_1 <= t_2 and t_2 <= t_1)$],
    [$t_1 < t_2$], [$T(t_1 <= t_2 and t_1 != t_2)$],
    [$phi or psi$], [$T(phi) dot T(psi)$],
    [$phi and psi$], [$T(phi) + T(psi)$],
  )
  
  By construction $T(phi)(x) >= 0, forall x, phi$.
  Negation can be implemented by using de Morgan's laws.
]

#cbox(title: [Box constraints])[
  hard to enforce in GD. Use L-BFGS-B and give box constraints to optimizer.
]

== 9.2 Training NN with Background Knowledge

#cbox(title: [Problem statement])[
  Enforce logical property $phi$ when training NN.
  
  find $theta$ that maximizes the expected value of property $EE_(s tilde D) [ forall z . phi(z, s, theta) ]$.
  
  BUT: Universal quantifiers are difficult.
]

#cbox(title: [Reformulation])[
  get the worst violation of $phi$ and minimize its effect, i.e.
  $EE_(s tilde D) [max_z not phi(z, s, theta) ]$.
  
  *Reform. 2:* minimize
  $EE_(s tilde D)[T(phi)(b z, s, theta)]$, where $b z = "argmin"(T(not phi)(z, s, theta))$.
  This is an adv. attack.
  
  $exists$ different $b z$ which minim. $T(not phi)$ which can produce different $T(phi)$.
  $b z !=$ worst example.
  
  Restrict $z$ to a convex set with efficient projs., i.e. $L_infinity$-balls.
  Remove the constraint from $phi$ that restricts $z$ on the convex set and do PGD while projecting $z$ onto the convex set.
]

= 10. Fairness

#cbox(title: [Individual Fairness])[
  A mapping $M : cal(X) arrow Delta(cal(Y))$
  is $(D, d)$-*Lipschitz*, if for every $x_1, x_2 in cal(X)$
  $D(M(x_1), M(x_2)) <= d(x_1, x_2)$.
  If $M$ is a model, it's *individually fair* wrt. $D$ and $d$.
  $d$ is a distance in feature space, $D$ is a metric on probability distributions.
  Choosing metrics is hard.
  
  Lemma: For $h : RR^d arrow [0, 1]$, $x arrow.bar Phi^(-1)(EE_(epsilon tilde cal(N)(0, I))[h(x + epsilon)])$
  is $1$-Lipschitz in $x$.
]

#cbox(title: [Lipschitz Property])[
  Let $L in RR$ be s.t. $D(M(x), M(x')) <= L d(x, x')$ (smaller value is stronger).
  Let $d(x, x') := (x - x')^top S(x - x')$, where $S$ is a symmetric positive definite covariance matrix.
  Let $D(M(x), M(x')) := [M(x) != M(x')]$.
  Then the Lipschitz property is equivalent to $forall delta in BB_S (0, 1 \/ L) quad M(x) = M(x + delta)$,
  where $||x||_S := sqrt(x^top S x)$.
  We have reformulated individual fairness as robustness.
]

== 10.1 Fair Representation Learning

#cbox(title: [FRL])[
  FRL is often more efficient (reuse fair data) and simplifies audits.
  But it has less precise control of the fairness/performance tradeoff,
  is susceptible to adv. attacks by the consumer, can be expensive
  and provides no certification.
]

== 10.2 Learning Certified Individually Fair Representations

#cbox(title: [LCIFR])[
  Keep pros of FRL, but also allow the regulator to certify the fairness of the E2E model
  and allows to define $D$ and $d$ via logical constraints that are accepted by MILP and DL2.
  Example: $d(x, x') = and.big_(i in "Cat" \\ {"race", "gender"}) (x_i = x_i')
  and.big_(j in "Num") |x_j - x_j'| <= alpha$.
  Logic captures cat. features exactly, norms don't.
  
  Let $S_d (x)$ denote the set of all points similar to $x$
  and assume $D(M(x), M(x')) = [M(x) != M(x')]$.
  
  The encoder $f_theta : RR^n arrow RR^k$ is trained using DL2 s.t.
  $forall x' in S_d (x) quad ||f_theta (x) - f_theta (x')||_infinity <= delta$.
  $S_d (x)$ is a complicated set, which we bound by a box in latent space.
  The producer encodes $S_d (x)$ and $f_theta$ as MILP to compute $epsilon$ s.t.
  $f_theta (S_d (x)) subset.eq {z' mid ||f_theta (x) - z'||_infinity <= epsilon}$,
  which gives the consumer a simple robustness problem.
  
  Train encoder using training with background knowledge with classifier to keep latent space useful.
  Train decoder via randomized smoothing.
]

== 10.3 Latent Space Smoothing for individually Fair Representations

#cbox(title: [LSS])[
  Use semantic feature space from a good gen. model encoder
  for similarity formulas for images etc.
  
  Center smoothing produces a bound on the radius of the ball in latent space.
  The E2E model is individually fair with probability $1 - alpha_"rs" - alpha_"cs"$.
]

== 10.4 Group Fairness

#cbox(title: [Definitions])[
  *Demographic parity*: $PP(hat(Y) = 1 mid G = 0) = PP(hat(Y) = 1 mid G = 1)$, where $G$ is a group feature.
  
  *Equal opportunity*: $PP(hat(Y) = 1 mid Y = 1, G = 0) = PP(hat(Y) = 1 mid Y = 1, G = 1)$
  
  *Equalized odds*: Equal opportunity and $PP(hat(Y) = 1 mid Y = 0, G = 0) = PP(hat(Y) = 1 mid Y = 0, G = 1)$
]

#cbox(title: [Postprocessing])[
  Example of *postprocessing*: for a binary classifier with output probability $h(x)$.
  Use separate thresholds for each group, tuned to achieve group fairness.
]

#cbox(title: [In-training])[
  Example of *in-training*: add relaxed fairness constraints that are solved with DL2, i.e.
  $-epsilon <= PP(hat(Y) = 1 mid s = 0) - PP(hat(Y) = 1 mid s = 1) <= epsilon$
]

#cbox(title: [Preprocessing: FRL])[
  Notation: data $(x, s) in RR^d times {0, 1}$, encoder $f : RR^d times {0, 1} arrow RR^(d'), z = f(x, s)$,
  classifier $g : RR^(d') arrow {0, 1}$,
  adversary $h : RR^(d') arrow {0, 1}$ is a classifier that tries to predict the sensitive attribute from data in the latent space,
  $Z_i := {z mid s = i}$, $p_i (z) := PP(z mid s = i)$.
]

#cbox(title: [LAFTR])[
  jointly train $f, g$ and $h$. No guarantees.
  $min_(f, g) max_(h)(cal(L)_"clf" (f(x, s), g) - gamma cal(L)_"adv" (f(x, s), h))$
  
  Use adversary to add guarantees by computing an upper bound on unfairness of any $g$.
  Convert hard constraint (DP, EO) into a soft measure, e.g. for demographic parity:
  $Delta_(Z_0, Z_1) (g) := |EE_(z tilde Z_0) g(z) - EE_(z tilde Z_1) g(z)|$, lower is better.
  Balanced accuracy is $"BA"_(Z_0, Z_1) (h) = frac(1, 2)(EE_(z tilde Z_0) (1 - h(z)) + EE_(z tilde Z_1) h(z)) =$
  $frac(1, 2) integral_Z (p_0 (z)(1 - h(z)) + p_1 (z)h(z))$,
  $h$ chooses $p_0$ or $p_1$.
  The optimal adversary is $h^* (z) := [p_1 (z) >= p_0 (z)]$.
  Theorem: $Delta_(Z_0, Z_1) (g) <= 2 dot "BA"_(Z_0, Z_1) (h^*) - 1$.
  We can't find neither $"BA"$ nor $h^*$ exactly.
]

#cbox(title: [Fair Normalizing Flows])[
  sample $x$ from a known distribution $q$, apply an invertible encoder $z = f(x)$,
  find density of the new distribution by $log p(z) = log q(f^(-1)(z)) + log|det frac(partial f^(-1)(z), partial z)|$.
  Learn normalizing flows $f_0$ and $f_1$ as encoders for $Z_0$ and $Z_1$.
  This lets us find $p_0 (z)$ and $p_1 (z)$, given $q_0 (x), q_1 (x)$.
  They can be estimated with density estimation, e.g. Gaussian Mixture Model.
  Given $p_0 (z), p_1 (z)$, we estimate an UB of $"BA"$ with probability $1 - epsilon$
  by Hoeffding's inequality, and then apply the theorem for UB of $Delta$.
  
  For good bounds, need low accuracy of $h^* arrow.r.double$ low dist. between $Z_0$ and $Z_1$.
  Add $"KL"$ divergence between $p_0$ and $p_1$ (and $"KL"(p_1, p_0)$) to loss of $g$.
  $g$ will be thrown away after training, as it exists only to increase utility of the flows.
  
  The bound holds only when the $q$ estimates are accurate, which is a major limitation.
]

#cbox(title: [Fairness with Restricted Encoders])[
  restrict the space of representations to be finite.
  This allows to get the distribution of sensitive attributes at each $z$, hence we have $p_i (z)$.
  First, we bound $P(s = i)$ using binom. conf. intervals,
  then per-cell balanced accuracy, then $"BA"$. This is done on different datasets to achieve independence.
]

= Appendix

#cbox(title: [De Morgan])[
  $not(phi and psi) = not phi or not psi$;
  $not(phi or psi) = not phi and not psi$
]

#cbox(title: [Ball Relations])[
  $BB_epsilon^1 subset.eq BB_epsilon^2 subset.eq BB_epsilon^infinity subset.eq BB_(epsilon sqrt(d))^(2) subset.eq BB_(epsilon dot d)^1$
]

#cbox(title: [Jensen])[
  $g$ convex: $g(E[X]) <= E[g(X)]$
]

#cbox(title: [Bayes])[
  $P(X|Y) = frac(P(X,Y), P(Y)) = frac(P(Y|X)P(X), P(Y))$
]

#cbox(title: [Matrix Inverse])[
  $A^(-1) = mat(a, b; c, d)^(-1) = 1/(a d - b c) mat(d, -b; -c, a)$
]

#cbox(title: [Norms])[
  $||x||_p = ( sum_(i=1)^d |x_i|^p )^(frac(1, p))$ #h(1em) $||x||_infinity = max_(i in {1,..,d}) |x_i|$
]

#cbox(title: [Softmax])[
  $sigma(z)_i = e^(z_i) \/ sum_(j=1)^(D) e^(z_j)$
]

#cbox(title: [CE loss])[
  $"CE"(arrow(z), y) = - sum_(c=1)^(K) bb(1)[c = y] dot log z_c$
]

#cbox(title: [Implication])[
  $phi arrow.r.double psi arrow.l.r.double not phi or psi$
]

#cbox(title: [Gauss])[
  $cal(N) = frac(1, sqrt((2pi)^d |Sigma|))exp(-frac(1, 2)(x-mu)^T Sigma^(-1) (x-mu))$
  *CDF*: $Phi (v; mu, sigma^2) = integral_(-infinity)^(v) cal(N)(y;mu,sigma^2)d y=Phi (frac(v-mu, sqrt(sigma^2));0,1)$
]

#cbox(title: [Laplace])[
  $cal(L) = frac(1, 2b)exp(-frac(|x-mu|, b))$,
  $Phi (x;mu,b) = 0.5 + 0.5 "sgn"(x - mu)(1 - exp(-frac(|x-mu|, b)))$
]

#cbox(title: [Subadditivity of $sqrt(dot)$])[
  $sqrt(x + y) <= sqrt(x) + sqrt(y)$
]

#cbox(title: [Cauchy Schwarz])[
  $angle.l x,y angle.r <= ||x||_2 dot ||y||_2$
]

#cbox(title: [Hölder's])[
  $||x dot y||_1 <= ||x||_p dot ||y||_q$, if $frac(1, p) + frac(1, q) = 1$
]

#cbox(title: [Minmax])[
  $max_a min_b f(a,b) <= min_b max_a f(a,b)$
]

#cbox(title: [Variance & Covariance])[
  $VV(X)=EE[(X-EE[X])^2]=EE[X^2]-EE[X]^2$
  
  $VV(X + Y) = VV(X) + VV(Y) + 2"Cov"(X,Y)$
  
  $VV(A X) = A VV(X) A^T$,$VV[alpha X]=alpha^2 VV[X]$
  
  $"Cov"(X,Y)=EE[(X-EE[X])(Y-EE[Y])]$
]

#cbox(title: [Distributions])[
  $"Exp"(x|lambda)=lambda e^(-lambda x)$, $"Ber"(x|theta)=theta^x (1-theta)^((1-x))$
  
  Sigmoid: $sigma(x)=1\/(1+e^(-x))$
  
  $a cal(N)(mu_1, sigma_1^2) + cal(N)(mu_2, sigma_2^2) = cal(N)(a mu_1 + mu_2, a^2 sigma_1^2 + sigma_2^2)$
]

#cbox(title: [Normal CDF])[
  $x tilde cal(N)(0, 1) arrow.r.double
  PP(x <= z) = Phi(z),
  PP(x <= Phi^(-1)(z)) = z$.
  $x tilde cal(N)(mu, sigma^2) arrow.r.double
  PP(x <= z) = Phi(frac(z - mu, sigma)),
  PP(x <= mu + sigma Phi^(-1)(z)) = z$
]

#cbox(title: [Chebyshev & Consistency])[
  $PP(|X-EE[X]|>= epsilon)<= frac(VV[X], epsilon^2)$,
  $lim_(n arrow infinity) PP(|hat(mu)-mu |>epsilon)=0$
]

#cbox(title: [Derivatives])[
  $(f g)' = f'g + f g'$; $(f\/g)' = (f'g - f g')\/g^2$
  
  $f(g(x))' = f'(g(x))g'(x)$; $log(x)' = 1\/x$
  
  $diff_x bold(b)^top bold(x) = diff_x bold(x)^top bold(b) = bold(b)$,
  $diff_x bold(x)^top bold(x) = diff_x ||bold(x)||_2^2 = 2 bold(x)$,
  
  $diff_x bold(x)^top bold(A) bold(x) = (bold(A)^top + bold(A)) bold(x)$, $diff_x (bold(b)^top bold(A) bold(x)) = bold(A)^top bold(b)$,
  
  $diff_X (bold(c)^top bold(X) bold(b)) = bold(c) bold(b)^top$,
  $diff_X (bold(c)^top bold(X)^top bold(b)) = bold(b) bold(c)^top$,
  
  $diff_x (|| bold(x)-bold(b) ||_2) = frac(bold(x)-bold(b), ||bold(x)-bold(b)||_2)$,
  $diff_X (||bold(X)||_F^2) = 2 bold(X)$,
  
  $diff_x ||bold(x)||_1 = frac(bold(x), |bold(x)|)$,
  $diff_x ||bold(A) bold(x) - bold(b)||_2^2 = bold(2(A^top A x-A^top b))$,
]

#cbox(title: [MILP encodings])[
  $y = |x|, l <= x <= u$: $y >= x, y >= -x$,
  
  $y <= -x + a dot 2u, y <= x - (1-a) dot 2l$, $a in {0,1}$
  
  $y = max(x_1, x_2), l_1 <= x_1 <= u_1, l_2 <= x_2 <= u_2$:
  
  $y >= x_1, y >= x_2$, $y <= x_1 + a dot (u_2 - l_1)$,
  
  $y <= x_2 + (1 - a) dot (u_1 - l_2), a in {0,1}$
]
