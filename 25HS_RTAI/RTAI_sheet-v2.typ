#import "../assets/tmp_sht.typ": *
#show: project.with(authors: ((name: "", email: ""),))

#let fsize = 10pt
#let hsize1 = 11pt
#let hsize2 = 10.5pt
#let pspace = 0.1em
#let plead = 0.18em

#set text(size: fsize)
#set par(spacing: pspace, leading: plead, justify: true, first-line-indent: 0em)
#show heading.where(level: 1): set text(size: hsize1)
#show heading.where(level: 2): set text(size: hsize2)
#show heading: box
#show heading: set text(fill: rgb("#663399"), weight: "bold")
#show: columns.with(4, gutter: 0.35em)

= 1. Adversarial Attacks

#cbox(title: [FGSM])[
  *Targeted*: $x' = x - epsilon dot "sign"(nabla_x cal(L)(x,t))$ (toward target $t$)
  *Untargeted*: $x' = x + epsilon dot "sign"(nabla_x cal(L)(x,y))$ (away from true $y$)
  Sign normalizes $nabla$→lands on $ell_infinity$ ball *vertex*
  $eta$ not minimized, just $in[-epsilon,epsilon]^d$
]

#cbox(title: [PGD])[
  $x^(k+1)=Pi_(BB_epsilon(x^0))(x^k+alpha dot "sign"(nabla cal(L)))$
  *Init*: $x^0 + "uniform"(-epsilon,epsilon)$
  *Step decay*: $alpha^k = alpha^0/2^k$ (halve each iter)
  *Projection*: $ell_infinity$=clip; $ell_2$=scale to radius
  $ell_2$ proj: $x''=x^0+(epsilon)/(||x'-x^0||_2)(x'-x^0)$ if $||x'-x^0||>epsilon$
  Optimal adv example *always on boundary* (high-dim monotonic)
]



#cbox(title: [C&W])[
  $min_eta ||eta||_p^2 + c dot "OPS"(x+eta,t)$
  $"OPS"=max(0, max_(i!=t)Z_i - Z_t + kappa)$
  OPS$<=0 arrow.r.double$ attack succeeds; $kappa$ controls margin
  *Different from PGD*: minimizes perturbation size, not fixed $epsilon$
  Use LBFGS-B for box constraints; binary search on $c$
]

#cbox(title: [Targeted vs Untargeted])[
  Binary case ($d=2$): equivalent! Away from class 1 = toward class 2
  $d>=3$: NOT equivalent! Untargeted has multiple directions
  Loss relation: $cal(L)(x,t) = -cal(L)(x,y)$ only for 2-class
]

#cbox(title: [GCG (LLM Discrete)])[
  Tokens discrete, can't do PGD directly
  1. One-hot→continuous: compute $nabla_e cal(L)$ in embedding space
  2. Top-K filter: select $K$ tokens with most negative $nabla$
  3. Greedy search: enumerate positions, keep best
  *Use $nabla$ to FILTER, not UPDATE!* Complexity $O(V^k)$ exponential
  Universal suffix: $min_"suf"sum_i cal(L)("Sure"|p_i,"suf")$ transfers to GPT-4
]

#cbox(title: [Norm Relations])[
  $||v||_infinity <= ||v||_2 <= sqrt(n)||v||_infinity$
  $||v||_2 <= ||v||_1 <= sqrt(n)||v||_2$
  $BB_epsilon^1 subset BB_epsilon^2 subset BB_epsilon^infinity$
  $ell_infinity$ constraint $arrow.r.double$ $ell_2$ constraint (converse false)
]

#cbox(title: [AutoAttack])[
  Ensemble: APGD-CE + APGD-DLR + FAB + Square (black-box)
  *Must use for reporting robust accuracy*
  Prevents "overfitting" defense to single attack
]

= 2. Defenses

#cbox(title: [Min-Max Framework])[
  $min_theta EE_{(x,y)} [ max_(x' in S(x)) cal(L)(theta, x', y) ]$
  *Attack*: fix $theta$, find $delta$ (inner max)
  *Defense*: optimize both (outer min)
  *Certify*: replace inner max with relaxation upper bound
]

#cbox(title: [PGD-AT])[
  For each batch: $x_"adv" = "PGD"(x, theta, epsilon)$
  Backprop on $nabla_theta cal(L)(f_theta(x_"adv"), y)$
  Inner: PGD (10-20 steps); Outer: SGD on $theta$
]

#cbox(title: [TRADES])[
  $cal(L) = cal(L)(f(x),y) + lambda max_(x' in BB_epsilon) "KL"(f(x)||f(x'))$
  Separately optimize clean acc and robustness
  $lambda$ trades off; typically $lambda in [1,6]$
  Often better clean-robust Pareto frontier than PGD-AT
]

#cbox(title: [$epsilon$-Robustness & Accuracy])[
  If $exists (x_1,y_1),(x_2,y_2)$ with $y_1!=y_2$ and $||x_1-x_2||_p<=epsilon$:
  *Cannot have both $epsilon$-robust and 100% accurate*
  Proof: if $f$ robust at $x_1$, all points in $BB_epsilon(x_1)$ same label→$x_2$ misclassified
]

= 3. Certification

Core: $forall i: phi(i) arrow.r.double N(i) models psi$

#cbox(title: [Sound vs Complete])[
  *Sound*: Proved $arrow.r.double$ True (no false positive, 底线!)
  *Complete*: True $arrow.r.double$ Provable (no false negative)
  Most practical: Sound but Incomplete (Box, DeepPoly, RS)
  MILP: Sound+Complete but $O(2^k)$
]

#cbox(title: [Crossing ReLU])[
  Input bounds $[l,u]$ with $l<0<u$: *unstable*
  $l>=0$: $y=x$ exact; $u<=0$: $y=0$ exact
  MILP complexity $O(2^k)$ where $k$=*\#Crossing* (NOT total neurons!)
  Reduce $k$: tighter bounds, certified training
]

== 3.1 MILP (Complete)

#cbox(title: [MILP Encoding])[
  *Affine*: $y=W x+b$ directly encoded
  *ReLU* ($l<0<u$): introduce $a in{0,1}$
  $y>=x$, $y<=x-l(1-a)$, $y<=u dot a$, $y>=0$
  $a=1$: $y=x$ (active); $a=0$: $y=0$ (inactive)
  *Specification*: $phi=BB_epsilon^infinity$: $x_i-epsilon<=x'_i<=x_i+epsilon$
  $psi$: prove $o_t > o_j$ $forall j!=t$: minimize $o_t - max_(j!=t)o_j$
]

#cbox(title: [MILP for Other Funcs])[
  *HatDisc/Abs*: $y=|x|$: $y>=x$, $y>=-x$, $y<=x+2u(1-a)$, $y<=-x+2|l|a$
  *Max*: $y=max(x_1,x_2)$: $y>=x_1$, $y>=x_2$, $y<=x_1+a(u_2-l_1)$, $y<=x_2+(1-a)(u_1-l_2)$
  *Binary Step*: like ReLU but output ${0,1}$ not $[0,u]$
]

#cbox(title: [MILP Limitations])[
  $ell_2$ ball is *quadratic* constraint→MILP *incomplete* for $ell_2$!
  Floating-point: theory Sound $neq$ hardware Sound (rounding errors)
  Infinite compute: Box-MILP equiv MILP-MILP (both explore all branches)
]

== 3.2 Relaxation (Incomplete)

#cbox(title: [Box/IBP $O(n^2 L)$])[
  $[a,b]+^\#[c,d]=[a+c,b+d]$; $-^\#[a,b]=[-b,-a]$
  $lambda[a,b]=cases([lambda a,lambda b] & lambda>=0, [lambda b,lambda a] & lambda<0)$
  $"ReLU"^\#[l,u]=["ReLU"(l),"ReLU"(u)]$
  *Affine exact*; ReLU crossing→over-approx (garbage points)
  Loosest but GPU-friendly, parallelizable
]

#cbox(title: [Box Propagation Example])[
  Given $x_1 in[0,0.5]$, $x_2 in[0.2,0.7]$:
  $x_3=x_1+x_2 in[0.2,1.2]$ (non-crossing, $l>=0$)
  $x_4=x_1-x_2 in[-0.7,0.3]$ (*crossing*! $l<0<u$)
  After ReLU: $x_5="ReLU"(x_3) in[0.2,1.2]$; $x_6="ReLU"(x_4) in[0,0.3]$
]

#cbox(title: [DeepPoly $O(n^3 L^2)$])[
  Each $x_i$: interval $l_i<=x_i<=u_i$
  Relational: $a_i^L<=x_i<=a_i^U$ where $a=sum w_j x_j+nu$
  *Affine*: exact, $z<=W x+b<=z$ (upper=lower)
  *ReLU* ($l<0<u$): $lambda=u/(u-l)$
  Upper: $y<=lambda(x-l)$ (fixed, connects $(l,0)$ to $(u,u)$)
  Lower: $y>=alpha x$, $alpha in[0,1]$ (optimizable, $alpha$-CROWN)
  Min area: $alpha=0$ if $|l|>u$; $alpha=1$ otherwise
]

#cbox(title: [Back-Substitution])[
  Recursively expand symbolic bounds to input layer
  *Key*: for $X_j<=sum c_i X_i+d$:
  - If $c_i>0$: substitute upper bound of $X_i$
  - If $c_i<0$: substitute *lower* bound of $X_i$ (opposite!)
  Can stop early using concrete bounds (efficiency)
]

#cbox(title: [DeepPoly Example])[
  $x_5="ReLU"(x_3)$, $x_3 in[-0.5,3.5]$ (crossing)
  Upper: $x_5<=3.5/4(x_3+0.5)=0.875 x_3+0.4375$
  Lower: $x_5>=0$ (if $alpha=0$) or $x_5>=x_3$ (if $alpha=1$)
  Back-sub to get concrete $[l_5,u_5]$
]

#cbox(title: [Single vs Multi-Neuron])[
  *Single*: each neuron independent, fully parallel (GPU)
  *Multi* (PRIMA): captures cross-neuron relations, tighter but serial
  DeepPoly=single-neuron; trades precision for speed
]

#cbox(title: [Triangle vs DeepPoly])[
  Triangle: 3 constraints (exact convex hull), exponential growth
  DeepPoly: 2 constraints (parallelogram), fixed complexity
  Triangle doesn't scale; DeepPoly does
]

== 3.3 Branch & Bound

#cbox(title: [B&B Algorithm])[
  1. *Bound*: compute bounds via DeepPoly/CROWN
  2. If $l>0$: SAFE; if $u<0$: UNSAFE (counterexample)
  3. *Branch*: select unstable ReLU, split on $x_i>=0$ vs $x_i<0$
  4. Recurse on both subproblems
  Worst case: $O(2^k)$; good heuristics crucial
]

#cbox(title: [Branching Heuristics])[
  *Largest interval*: $max(u-l)$ most uncertain
  *Closest to zero*: $min(|l|,|u|)$ most critical
  *$nabla$-based*: $max|nabla_x "obj"|$ most impact
  *Learning-based*: NN predicts best split
]

#cbox(title: [KKT/Lagrangian])[
  $(max_x f(x) "s.t." g(x)<=0) <= max_x min_(beta>=0)[f(x)+beta g(x)]$
  *Weak duality*: $max min <= min max$ (always holds)
  Split constraint $x_i>=0$: add $beta x_i$ to objective
  $beta$ found by $nabla$ descent; need full back-sub each step
]

#cbox(title: [$alpha$-$beta$-CROWN])[
  $alpha$: ReLU lower slope $in[0,1]$, $nabla$-optimizable
  $beta$: Lagrange multiplier $>=0$, encodes split constraints
  *Key*: $alpha,beta$ only affect *Tightness*, NOT *Soundness*!
  Any valid $alpha,beta$ gives sound bound, just looser/tighter
]

= 4. Certified Training

#cbox(title: [DiffAI Framework])[
  $min_theta EE [max_(z in gamma(f^\#(S(x)))) cal(L)(z,y)]$
  Use abstract transformer (Box/DeepPoly) instead of PGD
  *Abstract loss*: optimize over output region (incl. garbage points)
]

#cbox(title: [Abstract Loss $cal(L)^\#$])[
  *Margin loss* $cal(L)=max_(c!=y)(z_c-z_y)$:
  Compute $d_c=z_c-z_y$ for all $c$; take max of upper bounds
  *CE loss*: for each class, take upper (if $c!=y$) or lower (if $c=y$)
  Compute CE on this worst-case logit vector
]

#cbox(title: [Training Paradox])[
  Empirical: Box(86%) > Zonotope(73%) > DeepPoly(70%)
  *Tighter $neq$ better training!*
  Reason: tighter→discrete switching→discontinuous landscape→hard optimize
  Box: loose but smooth $nabla$s
]

#cbox(title: [SABR/COLT])[
  SABR: propagate to layer $k$, freeze; PGD on layers $k+1$ to $n$
  Solves projection problem: $ell_infinity$=clip; DeepPoly shape needs QP
  COLT: similar layer-wise approach with Zonotope
]

#cbox(title: [Certified Training Step])[
  Given network, input spec $x in[l,u]$, weight $w$:
  1. Box propagate: $x_3 in[l_3(w),u_3(w)]$ as function of $w$
  2. Compute worst-case loss: $cal(L)_"worst"=log(1+exp(u_7-l_8))$
  3. $nabla$: $nabla_w cal(L)_"worst"$
  4. Update: $w arrow.l w - eta nabla_w cal(L)_"worst"$
  Bounds are *continuous* in $w$ (linear+max are continuous)
]

= 5. Randomized Smoothing

#cbox(title: [Smoothed Classifier])[
  $g(x)=arg max_c PP_(epsilon tilde cal(N)(0,sigma^2 I))[f(x+epsilon)=c]$
  Base $f$ can be fragile; smoothed $g$ has certified guarantee
  *Theorem is deterministic; estimation is probabilistic!*
]

#cbox(title: [Certified Radius])[
  If $underline(p_A) > 0.5$: $R = sigma dot Phi^(-1)(underline(p_A))$
  $Phi^(-1)$: inverse standard normal CDF (probit)
  $p_A=0.5 arrow.r.double Phi^(-1)(0.5)=0 arrow.r.double R=0$
  $p_A arrow 1 arrow.r.double Phi^(-1)(p_A) arrow infinity arrow.r.double R arrow infinity$
  *$sigma arrow.t$ doesn't always mean $R arrow.t$!* (larger noise→lower $p_A$)
]

#cbox(title: [Two-Stage Sampling])[
  *Stage 1* ($n_0 approx 100$): guess top class $hat(c)_A$
  *Stage 2* ($n approx 10^5$): estimate $underline(p_A)$ via Clopper-Pearson CI
  If $underline(p_A)<=0.5$: ABSTAIN
  *Complexity*: $O(n_"samples")$, independent of network size!
]

#cbox(title: [Inference with Hypothesis Testing])[
  $H_0$: true $p("success")=0.5$
  BinomPValue$(n_A, n, 0.5)$: reject if $<alpha$
  $alpha$ small: more ABSTAIN but higher confidence
  Returns wrong class with prob at most $alpha$
]

#cbox(title: [Why $ell_2$ Only?])[
  Gaussian is *rotation invariant*: $||X||_2$ independent of direction
  →isotropic, equal prob surface is 球→$ell_2$ analytic formula
  Laplace→$ell_1$; Uniform→$ell_oo$: no closed form
]

#cbox(title: [RS vs Convex])[
  *Speed*: RS often *slower* (10k forward passes vs 1 abstract pass)
  *Scalability*: RS works on any size (LLMs); Convex limited to small/medium
  *Guarantee*: RS probabilistic; Convex deterministic
  *Training*: RS no special training; Convex needs certified training
]

#cbox(title: [Common Failures])[
  Wrong top class: $n_0$ too small→increase $n_0$
  $p_A<=0.5$: base model bad under noise→Gaussian adversarial training
  Lower bound too loose: $n$ too small→increase $n$
]

= 6. DP & RS Duality

#cbox(title: [DP vs RS: Same Tools, Opposite Goals])[
  *DP*: make distributions *indistinguishable* $P[M(D)] approx P[M(D')]$
  *RS*: make predictions *distinguishable* $P[G(x)=c] >> P[G(x)!=c]$
  Both use noise mechanisms, exponential bounds
  DP: want hypothesis test power *low*; RS: want confidence *high*
]

#cbox(title: [Lipschitz Connection])[
  Both proofs rely on Lipschitz constant $L$:
  DP: $L$ controls sensitivity→determines noise
  RS: $L$ controls $p_A$ change→determines radius
  DP Noise $prop L/epsilon$; RS Radius $prop sigma/L$
]

= 7. Privacy

#cbox(title: [$epsilon$-DP])[
  $PP(M(D) in S) <= e^epsilon PP(M(D') in S)$ for all neighboring $D,D'$
  $e^epsilon approx 1+epsilon$ for small $epsilon$
  *Laplace*: $f(D)+"Lap"(Delta_1/epsilon)$; $Delta_p=max_(D tilde D')||f(D)-f(D')||_p$
]

#cbox(title: [$(epsilon,delta)$-DP])[
  $PP(M(D) in S) <= e^epsilon PP(M(D') in S) + delta$
  $delta$: tail prob bound, *NOT "leak prob"!*
  Typically $delta << 1/n$
  *Gaussian*: $sigma = (Delta_2 sqrt(2ln(1.25/delta)))/epsilon$
]

#cbox(title: [Neighbor Definitions])[
  $||D-D'||_0<=1$: add/remove one record→Laplace
  $||D-D'||_2<=1$: continuous perturbation ($nabla$s)→Gaussian
]

#cbox(title: [Three Properties])[
  *Post-processing*: $g compose M$ still DP (can't "purify" noise)
  *Composition*: $(epsilon_1+epsilon_2, delta_1+delta_2)$-DP
  *Subsampling*: sample rate $q arrow.r.double (q epsilon, q delta)$
  *Advanced*: $T$ steps→$epsilon_"tot"=O(sqrt(T) epsilon)$ (crucial for training!)
  Independent data: $(max epsilon, max delta)$
]

#cbox(title: [DP-SGD])[
  1. *Clip* each $nabla$: $g_"clip"=g dot min(1, C/||g||_2)$
  2. *Aggregate + noise*: $g_"noisy"=1/L sum g_"clip" + cal(N)(0, sigma^2 C^2/L^2)$
  Clipping bounds sensitivity $Delta_2<=C$
  $sigma = (C sqrt(2ln(1.25/delta)))/(L epsilon)$
  Model private even against white-box attacker
]

#cbox(title: [Privacy Amplification])[
  Apply $(epsilon,delta)$-DP on random subset $q=L/N$:
  Result: $(tilde(q) epsilon, q delta)$-DP where $tilde(q) approx q$
  $T$ steps: $(tilde(q) T epsilon, q T delta)$ or $(O(q epsilon sqrt(T)), delta)$
]

#cbox(title: [PATE])[
  $M$ teachers on disjoint data, noisy voting labels public data, train student
  $n_j(x)=\#{t:t(x)=j}$; output $arg max(n_j(x)+"Lap"(2/epsilon))$
  *Add noise BEFORE argmax!* Sensitivity=2 (NOT $|Y|$!)
  Each query costs $epsilon$; total budget accumulates
]

#cbox(title: [FedSGD vs FedAvg])[
  *FedSGD*: send single-step $nabla$ $g_k$; server averages
  *FedAvg*: client runs $E$ epochs, sends weight diff $Delta theta$
  FedAvg harder to invert (multi-step trajectory unknown)
]

#cbox(title: [DP-FedSGD Noise])[
  Centralized: $sigma_"central"=(C sqrt(2ln(1.25/delta)))/(L epsilon)$
  Distributed ($m$ clients): $sigma_"client"=sqrt(m) dot sigma_"central"$
  Aggregation: $1/m sum g_k$ gives same noise level as centralized
]

= 8. Privacy Attacks

#cbox(title: [Attack Hierarchy])[
  *Attribute Inference*: infer sensitive attr (*no membership needed!*)
  *Data Extraction*: verbatim memorization (K-extractable)
  *MIA*: determine if $x in D_"train"$
  *Dataset Inference*: aggregate weak signals→strong signal
  *$nabla$ Inversion*: reconstruct from $nabla$s (FL)
]

#cbox(title: [MIA Methods])[
  *Shadow Model*: train $K$ shadows, train attack classifier
  *LiRA*: $log(P(ell|x in D)/P(ell|x in.not D))$ likelihood ratio
  *Min-K% Prob*: average of lowest $K$ token probs (LLM)
  *Loss-based*: training data has lower loss
  Practical AUC≈0.5-0.7 (weak!); TPR\@FPR=0.01 only 2%
]

#cbox(title: [$nabla$ Inversion])[
  $x^*=arg min_x ||nabla_theta cal(L)(x,y)-nabla_"obs"||^2+R(x)$
  Prior $R(x)$: TV (image), Perplexity (text), Entropy (tabular)
  FedSGD + BS=1: *exact reconstruction* ($nabla W_1=delta x^top$)
  FedAvg: needs multi-epoch coupling, harder
]

#cbox(title: [Model Stealing/Inversion])[
  *Stealing*: query API, train copy via distillation
  *Inversion*: $x^*=arg max_x P(y_"target"|x)$ reconstruct class representative
  Defense: rate limit, output perturbation, watermarking
]

#cbox(title: [Memorization Factors])[
  Model size↑, Prefix length↑, Repetition↑: more memorization
  Sequence length↑: less (cumulative errors)
]

= 9. Synthetic Data & Marginals

#cbox(title: [Pipeline])[
  1. *Select* marginal queries; 2. *Measure* with DP; 3. *Generate* synthetic
  Marginal $mu_t=sum_(x in D)[x_C=t]$; $Delta_2(M_C)=1$ (one row→one entry)
]

#cbox(title: [Chow-Liu])[
  MI-weighted complete graph→MST→sample along tree
  $p(F_1,F_2,F_3)=p(F_1)p(F_2|F_1)p(F_3|F_1)$
  DP: exponential mechanism for MST, Gaussian for marginals
]

#cbox(title: [Marginal Properties])[
  $(n-1)$-way marginals do *NOT* uniquely describe dataset
  Low-order marginals miss high-order correlations (XOR problem)
  3 columns, all 2-way: $binom(3,2)=3$ queries→$3 epsilon$ total
]

= 10. Logic & DL2

#cbox(title: [Logic→Loss Translation])[
  Theorem: $T(phi)(x)=0 arrow.l.r.double x models phi$
  $t_1<=t_2$: $max(0,t_1-t_2)$; $t_1=t_2$: $(t_1-t_2)^2$
  $phi and psi$: $T(phi)+T(psi)$; $phi or psi$: $T(phi) dot T(psi)$
  By construction $T(phi)>=0$; negation via De Morgan
  *Quantifiers NOT directly supported*; $forall$ via $max$ (worst violation)
]

#cbox(title: [Training with Background Knowledge])[
  Goal: $max_theta EE[forall z.phi(z,s,theta)]$
  Reform: $min_theta EE[T(phi)(hat(z),s,theta)]$ where $hat(z)=arg max T(not phi)$
  This is adversarial attack! Restrict $z$ to $ell_infinity$ ball, PGD+project
]

#cbox(title: [Logic Properties])[
  If $T(not phi)(y)=0$, then $not phi$ satisfied at $y$→$forall x.phi(x)$ FALSE
  $T(phi)(y_1)<=T(phi)(y_2) arrow.r.double T(not phi)(y_1)>=T(not phi)(y_2)$
  Infinite minimizers possible (e.g., $phi$ is tautology)
]

= 11. Fairness

#cbox(title: [Individual Fairness])[
  $(D,d)$-Lipschitz: $D(M(x),M(x'))<=d(x,x')$
  Equivalent to robustness: $forall delta in BB_S(0,1/L): M(x)=M(x+delta)$
  Lemma: $Phi^(-1)(EE[h(x+epsilon)])$ is 1-Lipschitz
]

#cbox(title: [Group Fairness])[
  *Demographic Parity*: $PP(hat(Y)=1|S=0)=PP(hat(Y)=1|S=1)$
  *Equal Opportunity*: above conditioned on $Y=1$ (TPR equal)
  *Equalized Odds*: conditioned on both $Y=0$ and $Y=1$
  Eq Odds $arrow.l.r.double hat(Y) perp S|Y$ (conditional independence)
]

#cbox(title: [$Delta_"EO"$ Calculation])[
  $Delta_"EO"=|"FPR"_0-"FPR"_1|+|"TPR"_0-"TPR"_1|$
  Example: $S=0$: FPR=7/10=0.7, TPR=3/6=0.5
  $S=1$: FPR=2/8=0.25, TPR=16/20=0.8
  $Delta_"EO"=|0.7-0.25|+|0.5-0.8|=0.45+0.3=0.75$
]

#cbox(title: [Adversary Bound])[
  Balanced Accuracy: $"BA"(h)=1/2(EE_(Z_0)(1-h)+EE_(Z_1)h)$
  Optimal adversary: $h^*(z)=[p_1(z)>=p_0(z)]$
  Theorem: $Delta_"EO"(g)<=2 dot "BA"(h^*)-1$
]

#cbox(title: [Eq Odds Proof Sketch])[
  Goal: $PP(hat(Y)=1|S=s,Y=y)$ same for all $s$→$hat(Y) perp S|Y$
  Use: $PP(hat(Y)|Y)=sum_s PP(hat(Y)|S=s,Y)PP(S=s|Y)$
  If $PP(hat(Y)|S,Y)=c$ for all $s$: $PP(hat(Y)|Y)=c$→conditional indep
]

#cbox(title: [LAFTR])[
  $min_(f,g) max_h [cal(L)_"clf"(f,g) - gamma cal(L)_"adv"(f,h)]$
  Use adversary to upper bound unfairness
]

#cbox(title: [LCIFR])[
  Train encoder: $forall x' in S_d(x): ||f(x)-f(x')||_infinity<=delta$
  MILP compute $epsilon$ s.t. $f(S_d(x)) subset {z': ||f(x)-z'||_infinity<=epsilon}$
  Consumer gets simple robustness problem
]

= 12. Watermark & Benchmark

#cbox(title: [Red-Green Watermark])[
  hash(context)+key→split vocab into Green/Red
  *Generate*: add $delta$ bias to Green token logits
  *Detect*: count Green tokens, binomial test *without LLM!*
  $p$-value$<alpha$→watermarked; $alpha$ controls FPR directly
]

#cbox(title: [ITS/SynthID])[
  *ITS*: distortion-free in expectation, but deterministic output
  *SynthID*: distortion-free + non-deterministic
  Tournament sampling: high G-value tokens more likely to win
]

#cbox(title: [Watermark Attacks])[
  *Scrubbing*: paraphrase ~30% tokens removes watermark
  *Spoofing*: modify one word, watermark persists (piggyback)
  *Stealing*: ~30K queries estimate $P_"wm"/P_"base"$, predict Green
]

#cbox(title: [Contamination])[
  *Data*: benchmark in training set (memorize answers)
  *Task*: optimized for task format (not truly solving)
  Detection: N-gram (L1), Perplexity (L2), Completion (L3)
  Outcome-based: compare 2024 vs 2025 performance (time causality)
]

#cbox(title: [VNN-COMP Critique])[
  "Verified 68M params"→check: \#Crossing, accuracy, $epsilon$ size
  Small $epsilon$=fewer crossing=easier; timeout=3600s impractical
  Verified $neq$ practically robust
]

= 13. Post-Training Attacks

#cbox(title: [Quantization Attack])[
  FP32 benign (passes detection), INT8 malicious (activated after deploy)
  Box constraint $[w_"low",w_"high"]$ s.t. quantized value unchanged
  Fine-tune in box with clean data→FP32 looks normal
]

#cbox(title: [Fine-Tuning Attack])[
  $cal(L)=cal(L)_"clean"(theta)+lambda cal(L)_"attack"(theta-nabla cal(L)_"user")$
  Safe now, malicious after user fine-tunes
  Needs Hessian: $ (partial cal(L)) (theta')/partial theta= (partial cal(L)) /partial theta' dot (I-eta nabla^2 cal(L)_"user")$
]

#cbox(title: [Agentic AI / IPI])[
  Indirect Prompt Injection: malicious instruction in tool output
  Agent can't distinguish user instruction vs tool content
  Defense: instruction hierarchy, dual-LLM, command sense
  Tradeoff: security $prop$ 1/capability
]

= 14. Regulation

#cbox(title: [EU AI Act])[
  *Unacceptable*: social credit scoring→prohibited
  *High Risk*: credit scoring, hiring→strict regulation
  *Limited Risk*: chatbots→transparency requirements
  Credit scoring is *High Risk*, NOT prohibited!
]

#cbox(title: [GDPR])[
  Removing PII insufficient→linkage attacks still possible
  Even "anonymized" purchase lists may violate GDPR
]

#cbox(title: [Appendix])[
  *Norms*:$||x||_p=(sum|x_i|^p)^(1/p)$; $||x||_infinity=max|x_i|$
  $cal(N)=(2pi)^(-d/2)|Sigma|^(-1/2)exp(-1/2(x-mu)^top Sigma^(-1)(x-mu))$
  $"Lap"=1/(2b)exp(-|x-mu|\/b)$; Sigmoid$sigma(x)=1/(1+e^(-x))$
  *Softmax&CE*: $sigma(z)_i=e^(z_i)\/sum_j e^(z_j)$; $"CE"(z,y)=-log sigma(z)_y=-z_y+log sum e^(z_j)$
  *Derivatives*: $diff_x b^top x=b$; $diff_x x^top x=2x$; $diff_x x^top A x=(A+A^top)x$
  $diff_x||A x-b||_2^2=2A^top(A x-b)$

  *不等式*: Cauchy-Schwarz:$angle.l x,y angle.r<=||x||_2||y||_2$
  Hölder:$||x dot y||_1<=||x||_p||y||_q$,$1/p+1/q=1$
  Jensen:$g$凸$arrow.r.double g(EE[X])<=EE[g(X)]$
  Chebyshev:$PP(|X-EE[X]|>=epsilon)<=VV[X]/epsilon^2$
  Minmax:$max min<=min max$(Weak Duality)
  Hoeffding:$PP(|hat(X)-EE[X]|>=epsilon)<=2exp(-2n epsilon^2/(b-a)^2)$

  *prob*: $VV(X)=EE[X^2]-EE[X]^2$; $VV(a X+b Y)=a^2 VV(X)+b^2 VV(Y)+2a b"Cov"$
  Bayes:$P(X|Y)=P(Y|X)P(X)/P(Y)$
  $Phi(z)=PP(cal(N)(0,1)<=z)$; $Phi^(-1)(0.5)=0$; $Phi^(-1)(0.975)approx 1.96$
  *Matrix*: $mat(a, b; c, d)^(-1)=1/(a d-b c)mat(d, -b; -c, a)$

  *MILP编码*: $y=|x|$:$y>=x,y>=-x$,$y<=x+2u(1-a),y<=-x+2|l|a$,$a in{0,1}$
  $y=max(x_1, x_2)$:$y>=x_1,y>=x_2$,$y<=x_1+a(u_2-l_1),y<=x_2+(1-a)(u_1-l_2)$

  *Logic*: De Morgan:$not(phi and psi)=not phi or not psi$; $not(phi or psi)=not phi and not psi$
  Implication:$phi arrow.r.double psi equiv not phi or psi$
  Ball:$BB^1_epsilon subset.eq BB^2_epsilon subset.eq BB^infinity_epsilon subset.eq BB^2_(epsilon sqrt(d))$
]

#cbox(title: [⚠️ Traps])[
  MILP complexity $O(2^k)$, $k$=*Crossing count*!
  RS theorem deterministic, estimation probabilistic
  $sigma arrow.t$ doesn't always $R arrow.t$ ($p_A$ drops!)
  GCG uses $nabla$ to *filter*, not *update*
  $n_0$ (guess class ~100) vs $n$ (estimate prob ~100k)
  PATE: noise *before* argmax, $Delta_1=2$
  Tighter $neq$ better training (Box trains best)
  Back-sub: negative coeff→opposite bound
  MILP incomplete for $ell_2$ (quadratic)
  PGD $neq$ CW: different objectives
  FGSM always on $ell_infinity$ boundary
  FedSGD easier to invert than FedAvg
  $delta$ is tail mass bound, not leak prob
  Gaussian DP needs $ell_2$ sensitivity
  Floating-point: theory Sound $neq$ hardware Sound
  Credit scoring is High Risk, NOT Unacceptable
  MIA AUC≈0.5-0.7 (basically random)
  Universal suffix transfers across models
]
// // = 大题
// #cbox(title: [⚙️PGD步骤])[
//   *Step 1*: 算初始logits $z_i$和分类
//   *Step 2*: 算Loss对$x$的梯度 $nabla_x cal(L)$:
//   对$cal(L)=-z_t^2+sum_(i!=t)z_i^2, (partial cal(L))  /(partial x_j)=sum_i ( (partial cal(L)) /(partial z_i))((partial z_i)/(partial x_j))$
//   ; $ (partial cal(L)) /(partial z_t)=-2z_t$; $ (partial cal(L)) /(partial z_(i!=t))=2z_i$
//   *Step 3*: update $x^"temp"=x^k pm eta dot "sign"(nabla)$ (targeted用$-$, untargeted用$+$)
//   *Step 4*: proj回$BB_epsilon(x^0)$
//   $ell_infinity$: $x_i^"new"="clip"(x_i^"temp", x_i^0-epsilon, x_i^0+epsilon)$
//   ; $ell_2$: if $||x^"temp"-x^0||>epsilon$: $x^"new"=x^0+epsilon(x^"temp"-x^0)/(||x^"temp"-x^0||)$

//   *Step 5*: 检查是否攻击成功 ($arg max z$改变?)
//   *Step 6*: 下一轮$eta^(k+1)=eta^k\/2$ (if decay)
// ]
// #cbox(title: [⚙️ MILP验证步骤])[
//   *检验编码正确性*：画约束区域图！1. 令$a=0$: 约束简化成什么？解区域是什么？; 2. 令$a=1$: 约束简化成什么？解区域是什么？; 3. 合并两个区域，应该恰好等于函数图像
//   *修复non-uniqueness* (如HatDisc在$x=0$两个值)：添加约束$x>=epsilon(1-a)$强制$x=0$时$a=1$
// ]

// #cbox(title: [⚙️ Binary Step编码])[
//   $sigma(x)=cases(1 & x>=0, 0 & x<0)$; $x in[l,u]$, $l<0<u$
//   *Case $l>=0$*: $y=1$ (constant)
//   *Case $u<0$*: $y=0$ (constant)
//   *Case $l<0<u$*: 需要$a in{0,1}$
//   $y>=a$, $y<=a$, $x>=l dot a$, $x<=u dot a + l(1-a)$...
//   (类似ReLU但输出${0,1}$不是$[0,u]$)
// ]

// #cbox(title: [⚙️ DeepPoly计算完整流程])[
//   *Forward pass* (算concrete bounds): 1. Input: $x_1 in[l_1,u_1]$, $x_2 in[l_2,u_2]$; 2. Affine: $x_3=x_1+x_2-0.5$ → $x_3 in[l_1+l_2-0.5, u_1+u_2-0.5]$; 3. 判断ReLU类型: $l_3<0<u_3$? → crossing!; 4. ReLU symbolic: upper $x_5<=lambda(x_3-l_3)$, lower $x_5>=alpha x_3$
  
//   *Back-substitution* (精化bounds):
//   1. 从output开始: $x_7=-x_5+x_6+3$; 2. 要算$u_7$→max $x_7$→min $x_5$, max $x_6$; 3. 替换$x_5,x_6$的symbolic bounds; 4. 继续替换直到只剩input变量; 5. 在input domain上优化(取端点!)
  
//   *符号规则*: 算upper bound时: 正系数$c_i>0$: 用$x_i$的upper bound; 负系数$c_i<0$: 用$x_i$的*lower* bound!
// ]

// #cbox(title: [⚙️ DeepPoly数值例子])[
//   $x_1,x_2 in[0,2]$; $x_3=x_1+x_2-0.5 in[-0.5,3.5]$ (crossing)
//   $x_5="ReLU"(x_3)$: $lambda=3.5/4=0.875$
//   Upper: $x_5<=0.875(x_3+0.5)=0.875x_3+0.4375$
//   Lower: $x_5>=0$ (选$alpha=0$因为$|l|=0.5<u=3.5$? 不对，$|l|<u$时选$alpha=1$)
//   实际: $|{-0.5}|=0.5<3.5$→min area用$alpha=1$: $x_5>=x_3$
  
//   Back-sub $x_5$到input:
//   Upper: $x_5<=0.875(x_1+x_2-0.5)+0.4375=0.875x_1+0.875x_2$
//   Max at $x_1=x_2=2$: $u_5=3.5$
// ]

// #cbox(title: [⚙️ Certified Training计算题])[
//   *题型*: 给网络结构和weight $w$，用Box传播，算worst-case loss，做一步GD
  
//   *Step 1*: Box传播 (bounds是$w$的函数!)
//   $x_3=w x_1+b$ → $x_3 in[w l_1+b, w u_1+b]$ if $w>=0$
//   (注意$w<0$时上下界交换!)
  
//   *Step 2*: ReLU后bounds
//   $x_5="ReLU"(x_3)$: $l_5=max(0,l_3)$, $u_5=max(0,u_3)$
  
//   *Step 3*: Worst-case loss (CE with logits $x_7,x_8$, target=$x_8$)
//   $cal(L)_"worst"=log(1+exp(u_7-l_8))$ (max $x_7$, min $x_8$)
  
//   *Step 4*: 梯度 (chain rule through bounds)
//   $partial cal(L)/partial w=partial cal(L)/partial u_7 dot partial u_7/partial w+...$
  
//   *Step 5*: 更新 $w_"new"=w-eta partial cal(L)/partial w$
  
//   *连续性*: Bounds是$w$的连续函数 (linear+max都连续)
// ]

// #cbox(title: [⚙️ RS认证计算])[
//   *Given*: $sigma$, 采样结果$n$次中$n_A$次是class A
//   *Step 1*: 估计$hat(p)_A=n_A/n$
//   *Step 2*: 计算置信下界$underline(p_A)$ (Clopper-Pearson或正态近似)
//   正态近似: $underline(p_A)=hat(p)_A-z_(alpha/2)sqrt(hat(p)_A(1-hat(p)_A)/n)$
//   *Step 3*: if $underline(p_A)<=0.5$: ABSTAIN
//   *Step 4*: $R=sigma dot Phi^(-1)(underline(p_A))$
  
//   *常用值*: $Phi^(-1)(0.5)=0$, $Phi^(-1)(0.84)approx 1$, $Phi^(-1)(0.975)approx 1.96$
// ]

// #cbox(title: [⚙️ 为什么$sigma$大不一定$R$大])[
//   $R=sigma dot Phi^(-1)(p_A)$
//   $sigma$↑ → 直接效应: $R$↑
//   $sigma$↑ → 噪声大 → $p_A$↓ → $Phi^(-1)(p_A)$↓ → $R$↓
//   两个效应相反! 存在最优$sigma^*$
// ]

// #cbox(title: [⚙️ DP敏感度计算])[
//   *$Delta_1$* (L1): 改变一条记录，输出向量L1变化max
//   *$Delta_2$* (L2): 改变一条记录，输出向量L2变化max
  
//   *Mean*: $f(D)=1/n sum x_i$; 加/删一个$x$: $Delta_1=||x||_1/n$
//   若$x$有界$||x||_1<=B$: $Delta_1=B/n$
  
//   *$nabla$* (一个sample): $Delta_2<=C$ (after clipping!)
  
//   *PATE投票*: 改变一个教师→一票变化→$n_j$改变$(+1,-1)$→$Delta_1=2$
// ]

// #cbox(title: [⚙️ DP预算计算])[
//   *Simple Composition*: $k$个$(epsilon,delta)$-DP query → $(k epsilon, k delta)$
//   *3 columns, 2-way marginals*: $binom(3,2)=3$ queries → 总预算$3epsilon$
  
//   *Subsampling*: 采样率$q=L/N$
//   $(epsilon,delta)$-DP mechanism → $(approx q epsilon, q delta)$-DP
  
//   *Advanced* ($T$ steps): $(O(sqrt(T)epsilon), delta)$ 而非 $(T epsilon, T delta)$
// ]

// #cbox(title: [⚙️ $nabla$ Inversion可行性])[
//   *FedSGD + BS=1*: $nabla_(W_1)cal(L)=delta dot x^top$
//   → 可精确恢复$x$! (解线性系统)
  
//   *FedSGD + BS>1*: 只能恢复$sum x_i$的线性组合
//   *FedAVG*: 多步更新，需要模拟整个轨迹，更难
  
//   *Binary classification ($d=2$)*: $nabla$符号直接揭示label!
//   *Multi-class ($d>3$)*: $nabla$是向量，无法唯一确定label
// ]

// #cbox(title: [⚙️ $Delta_"EO"$计算步骤])[
//   *Given*: 数据表(Dataset)和预测表(Predictions)
  
//   *Step 1*: 算各组FPR (False Positive Rate)
//   $"FPR"_s=P(hat(Y)=1|Y=0,S=s)=(\#"预测1且真实0")/(\#"真实0")$
  
//   *Step 2*: 算各组TPR (True Positive Rate)
//   $"TPR"_s=P(hat(Y)=1|Y=1,S=s)=(\#"预测1且真实1")/(\#"真实1")$
  
//   *Step 3*: $Delta_"EO"=|"FPR"_0-"FPR"_1|+|"TPR"_0-"TPR"_1|$
  
//   *Example*:
//   $S=0,Y=0$: 10人中7人预测1 → $"FPR"_0$=0.7
//   $S=0,Y=1$: 6人中3人预测1 → $"TPR"_0$=0.5
//   $S=1,Y=0$: 8人中2人预测1 → $"FPR"_1$=0.25
//   $S=1,Y=1$: 20人中16人预测1 → $"TPR"_1$=0.8
//   $Delta_"EO"=|0.7-0.25|+|0.5-0.8|=0.75$
// ]

// #cbox(title: [⚙️ BA与$Delta$关系])[
//   *Adversary* $h(z,y)$尝试从$z$预测$S$
//   *定义*: $h(z,0)=1-g(z)$, $h(z,1)=g(z)$
  
//   *BA计算*:
//   $"BA"=1/2["accuracy on" S=0 + "accuracy on" S=1]$
  
//   For $Y=0$: $h$预测$S=0$的prob=$P(g=0|S=0,Y=0)$; 预测$S=1$的prob=$P(g=1|S=1,Y=0)$
  
//   *Theorem*: $Delta_"EO"(g)<=2"BA"(h^*)-1$
//   验证: 若$Delta_"EO"=0.75$, BA=?(算出BA后代入验证)
// ]