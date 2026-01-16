#import "sheet-v2.typ": *
#import "@preview/xarrow:0.3.1": xarrow
#show: project.with(authors: ((name: "", email: ""),))

// ========== 参数配置 ==========
#let fsize = 9pt //7.5pt
#let hsize1 = 10pt //8pt
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

// ========== 术语缩写 ==========
#block(stroke: 0pt + black, inset: 3pt, width: 100%)[
  #set text(size: 7pt) //6pt
  = Dict
  *BALD*:Bayesian Active Learning by Disagreement; *BLR*:Bayesian Linear Reg; *BN*:Bayesian Network; *BNN*:Bayesian NN; *BO*:Bayesian Opt; *BP*:Belief Propagation; *CPD*:Cond Prob Dist; *DAG*:Directed Acyclic Graph; *DBE*:Detailed Balance Eq; *DDIM*:Denoising Diffusion Implicit Models; *DDPG*:Deep Deterministic PG; *DDPM*:Denoising Diffusion Prob Models; *DQN*:Deep Q-Net; *ECE*:Expected Calibration Error; *EI*:Expected Improvement; *ELBO*:Evidence Lower Bound; *GP*:Gaussian Process; *HMM*:Hidden Markov Model; *KF*:Kalman Filter; *KL*:Kullback-Leibler; *LDM*:Latent Diffusion; *LOTV*:Law of Total Var; *MALA*:Metropolis-Adjusted Langevin; *MAP*:Max A Posteriori; *MCMC*:Markov Chain MC; *MDP*:Markov Decision Process; *MH*:Metropolis-Hastings; *MI*:Mutual Info; *MLE*:Max Likelihood Est; *MPE*:Most Probable Explanation; *PF*:Particle Filter; *PI*:Prob of Improvement; *POMDP*:Partially Observable MDP; *RBF*:Radial Basis Fnc; *RFF*:Random Fourier Features; *SGLD*:Stoch Grad Langevin Dyn; *SWAG*:Stoch Weight Avg Gaussian; *TD*:Temporal Diff; *UCB*:Upper Confidence Bound; *VE*:Var Elimination; *VI*:Variational Inference;
  $k_(X X'):= k(X,X')$; $K_y:=K_(X X)+sigma_n^2 I$
  //1/√2:0.707; √2:1.414; √3:1.732; ln2:0.693; ln3:1.099; 1/e:0.368; $e$:2.718; $(1-1/e)$:0.632
]
#let EE = $bb(E)$
#let PP = $bb(P)$
#let VV = $bb(V)$
#let opt = text([$*$], fill: red)
// ==============================

= Probability Fundamentals
#cbox(title: [Axioms])[
  $PP(Omega)=1$; $PP(A)>=0$; Disjoint: $PP(union.big_i A_i)=sum_i PP(A_i)$
  *Product*: $PP(X_(1:n))=PP(X_1)product_(i=2)^n PP(X_i|X_(1:i-1))$
  *Sum*: $PP(X)=sum_y PP(X, y)$
  *Bayes*: $PP(X|Y)=(PP(Y|X)PP(X))/(PP(Y))$
  *Cond Indep*: $X perp Y|Z arrow.l.r PP(X, Y|Z)=PP(X|Z)PP(Y|Z)$
]

#cbox(title: [Gaussian $cal(N)(mu,Sigma)$])[
  $cal(N)(x;mu,Sigma)=1/sqrt((2pi)^d det(Sigma))exp(-1/2(x-mu)^top Sigma^(-1)(x-mu))$
  *Marginal*: $X_A tilde cal(N)(mu_A, Sigma_(A A))$
  *Conditional*: $X_A|X_B tilde cal(N)(mu_(A|B), Sigma_(A|B))$
  $mu_(A|B)=mu_A+Sigma_(A B)Sigma_(B B)^(-1)(x_B-mu_B)$
  $Sigma_(A|B)=Sigma_(A A)-Sigma_(A B)Sigma_(B B)^(-1)Sigma_(B A)$
  *Linear*: $Y=M X tilde cal(N)(M mu, M Sigma M^top)$
  *Sum*: indep $X+X' tilde cal(N)(mu+mu', Sigma+Sigma')$
]

#cbox(title: [$EE$, Var, Cov])[
  $EE[A X+b]=A EE[X]+b$; *Tower*: $EE_Y[EE_X[X|Y]]=EE[X]$
  $"Var"[X]=EE[(X-EE[X])^2]$; $"Cov"[X,Y]=EE[X Y]-EE[X]EE[Y]$
  $"Var"[X+Y]="Var"[X]+"Var"[Y]+2"Cov"[X,Y]$
  *LOTV*: $"Var"[X]=EE["Var"[X|Y]]+"Var"[EE[X|Y]]$
]

#cbox(title: [Info Theory])[
  *Entropy*: $H[p]=-EE_p[log p(x)]$; Gauss $H=1/2 log((2pi e)^d det Sigma)$
  *KL*: $"KL"(p||q)=EE_p[log p/q]>=0$; need $"supp"(q) subset.eq "supp"(p)$
  *Forward* $"KL"(p||q)$: mean-seeking覆盖; *Reverse* $"KL"(q||p)$: mode-seeking过confident
  *MI*: $I(X;Y)=H[X]-H[X|Y]=H[Y]-H[Y|X]>=0$, symmetric
  *Cond MI*: $I(X;Y|Z)=H[X|Z]-H[X|Y,Z]$
  *Gauss MI*: $I[X;Y]=1/2 log det(I+sigma_n^(-2)Sigma)$ for $Y=X+epsilon$

  *MLE*: $hat(theta)_"MLE"=arg max_theta sum_i log p(y_i|x_i,theta)$

  *MAP*: $hat(theta)_"MAP"=arg min_theta underbrace(-log p(theta), "reg")+underbrace(ell_"nll", "fit")$
  Gaussian prior→L2; Laplace prior→L1
]

= Bayesian Linear Regression

#cbox(title: [Model])[
  $y=w^top x+epsilon$, $epsilon tilde cal(N)(0,sigma_n^2)$; Prior: $w tilde cal(N)(0,sigma_p^2 I)$
]

#cbox(title: [Posterior])[
  $w|X,y tilde cal(N)(mu,Sigma)$ where
  $Sigma^(-1)=sigma_n^(-2)X^top X+sigma_p^(-2)I$; $mu=sigma_n^(-2)Sigma X^top y$
  Note: $Sigma$只依赖$X$, 不依赖$y$
]

#cbox(title: [Prediction])[
  $y_*|x_*,X,y tilde cal(N)(x_*^top mu, x_*^top Sigma x_*+sigma_n^2)$
]

#cbox(title: [Connection])[
  `MAP`=Ridge with $lambda=sigma_n^2/sigma_p^2$; Online update: $O(n d^2)$
]

= Gaussian Processes

#cbox(title: [Def])[
  $f tilde cal(G P)(mu,k)$: any finite subset jointly Gaussian.
  $f_A tilde cal(N)(mu_A, K_(A A))$, $[K_(A A)]_(i j)=k(x_i,x_j)$
]

#cbox(title: [GP Regression])[
  $y tilde cal(N)(0,K_(X X)+sigma_n^2 I)=cal(N)(0,K_y)$
  *Mean*: $mu_*(x)=k(x,X)K_y^(-1)y$
  *Cov*: $k_*(x,x')=k(x,x')-k(x,X)K_y^(-1)k(X,x')$
  *Predictive*: $y_* tilde cal(N)(mu_*, k_*+sigma_n^2)$
]

#cbox(title: [Kernels])[
  *Linear*: $k(x,x')=x^top x'+sigma_0^2$
  *RBF*: $k=exp(-||x-x'||^2/(2ell^2))$ smooth无限可微
  *Exponential*: $k=exp(-||x-x'||/ell)$ rough
  *Matérn*: $nu=0.5$→Exp, $nu arrow infinity$→RBF, $nu$控制smoothness
  *Periodic*: $k=sigma^2 exp(-2/ell^2 sin^2(pi|x-x'|/p))$
  *Closure*: $k_1+k_2$, $k_1 dot k_2$, $c dot k$, $exp(k)$仍valid kernel
  *Stationary*: $k(x,x')=k(x-x')$; *Isotropic*: $k=k(||x-x'||)$
]

#cbox(title: [Marginal Lik])[
  $log p(y|X)=-1/2 y^top K_y^(-1)y-1/2 log det(K_y)+C$
  Balance: Datafit(前) vs Complexity(后)
]

#cbox(title: [Approx $O(n^3)$ → lower])[
  *RFF*: $k(x-x')approx phi(x)^top phi(x')$, $O(n m^2+m^3)$
  Bochner: stationary kernel ↔ Fourier of non-neg measure
  *Inducing Pts*: subset $m<<n$ points for approx
]

= Variational Inference

#cbox(title: [Goal])[
  Approx $p(theta|D)$ with $q(theta|lambda)$ by min $"KL"(q||p)$
]

#cbox(title: [ELBO])[
  $cal(L)=EE_q[log p(y|theta)]-"KL"(q(theta)||p(theta))$
  $log p(y)=cal(L)+"KL"(q||p(dot|D))>=cal(L)$
  Max ELBO ⇔ Min KL to posterior
  *Derivation*: Jensen's $log EE_q[p/q]>=EE_q[log p/q]$
]

#cbox(title: [Gaussian KL])[
  $"KL"(cal(N)_p||cal(N)_q)=1/2[tr(Sigma_q^(-1)Sigma_p)+(mu_p-mu_q)^top Sigma_q^(-1)(mu_p-mu_q)-d+log(det Sigma_q/det Sigma_p)]$
  *Product*: $"KL"(Q_X Q_Y||P_X P_Y)="KL"(Q_X||P_X)+"KL"(Q_Y||P_Y)$
]

#cbox(title: [Reparam Trick])[
  $theta=g(epsilon;lambda)$, $epsilon tilde phi$
  $EE_(theta tilde q)[f(theta)]=EE_epsilon[f(g(epsilon;lambda))]$
  *Gaussian*: $theta=mu+sigma dot.circle epsilon$, $epsilon tilde cal(N)(0,I)$
  Enable gradient: $nabla_lambda EE_q[f]=EE_phi[nabla_lambda f(g(epsilon;lambda))]$
]

#cbox(title: [Laplace Approx])[
  $q(theta)=cal(N)(hat(theta),Lambda^(-1))$
  $hat(theta)="MAP"$; $Lambda=-nabla^2 log p(hat(theta)|D)$ (Hessian)
  Good at mode, overconfident elsewhere
]

= Markov Chains & MCMC

#cbox(title: [MC basics])[
  *Markov*: $X_(t+1) perp X_(1:t-1)|X_t$
  *Stationary* $pi$: $pi=pi P$
  *Irreducible*: all states reachable from any state
  *Aperiodic*: $gcd{t: P^t(x,x)>0}=1$
  *Ergodic*=Irreducible+Aperiodic: unique $pi>0$, $lim_(t arrow infinity)q_t=pi$
]

#cbox(title: [DBE])[
  $pi(x)P(x'|x)=pi(x')P(x|x')$
  If satisfied→$pi$ stationary, chain *reversible*
  *Proof*: sum over $x'$得$pi(x)=sum_(x')pi(x')P(x|x')$
]

#cbox(title: [Ergodic Thrm])[
  $1/n sum_(i=1)^n f(X_i) arrow.r^("a.s.") EE_(x tilde pi)[f(x)]$
  Hoeffding: error prob decays $exp(-n)$
]

#cbox(title: [MH Algorithm])[
  Propose $x' tilde R(x'|x)$. Accept w.p.:
  $alpha(x'|x)=min{1,(q(x')R(x|x'))/(q(x)R(x'|x))}$
  Stationary: $p(x) prop q(x)$ (unnormalized OK)
  Satisfies DBE→correct stationary dist
]

#cbox(title: [Gibbs Sampling])[
  Iterate: $x_i^((t+1)) tilde p(x_i|x_(-i)^((t)))$
  Special MH with acceptance=1
  *Practical*: 顺序scan all vars, sample each from conditional
]

#cbox(title: [Langevin & SGLD])[
  *Langevin*: $R(x'|x)=cal(N)(x';x-eta nabla f(x),2eta I)$ where $p prop e^(-f)$
  *MALA*: MH-corrected Langevin, poly-time for log-concave
  *SGLD*: $theta_(t+1)=theta_t+epsilon_t(nabla log p(theta)+nabla log p(D|theta))+sqrt(2epsilon_t)xi$
  Converge: $sum_t epsilon_t=infinity$, $sum_t epsilon_t^2<infinity$; 常用$epsilon_t in Theta(t^(-1/3))$
]

#cbox(title: [Gibbs Distribution])[
  $p(x)=1/Z exp(-f(x))$, $f$=energy function
  Posterior always interpretable as Gibbs
]

= Bayesian Neural Networks

#cbox(title: [Model])[
  Prior: $theta tilde cal(N)(0,sigma_p^2 I)$
  *Homoscedastic*: $y|x,theta tilde cal(N)(f(x;theta),sigma^2)$ fixed noise
  *Heteroscedastic*: $y tilde cal(N)(f_mu (x;theta),exp(f_sigma (x;theta)))$ input-dependent noise
]

#cbox(title: [Hetero NLL])[
  $-log p(y|x,theta)=C+1/2[log sigma^2(x)+((y-mu(x))^2)/(sigma^2(x))]$
  Model can "blame" noise but pays $log sigma$ penalty防collapse
]

#cbox(title: [MAP for BNN])[
  $hat(theta)_"MAP"=arg min 1/(2sigma_p^2)||theta||^2+1/(2sigma_n^2)sum_i(y_i-f(x_i;theta))^2$
  Weight decay = Gaussian prior
]

#cbox(title: [Prediction])[
  $p(y_*|x_*,D)approx 1/m sum_(j=1)^m p(y_*|x_*,theta^((j)))$, $theta^((j)} tilde q$
  MC approx of posterior predictive
]

#cbox(title: [Uncertainty Decomp])[
  *Total Var*=*Aleatoric*+*Epistemic*
  Aleatoric(data noise): $1/m sum_j sigma^2(x_*,theta^((j)))$
  Epistemic(model uncertainty): $1/m sum_j(mu(x_*, theta^((j)))-bar(mu))^2$
  where $bar(mu)=1/m sum_j mu(x_*, theta^((j)))$
]

#cbox(title: [MC Dropout])[
  $q_j(theta_j)=p delta_0(theta_j)+(1-p)delta_(lambda_j)(theta_j)$
  Test时keep dropout→multiple forward passes→uncertainty
]

#cbox(title: [SWAG])[
  Store running avg of SGD iterates: $mu,Sigma$
  Space: $O(d^2)$ covariance vs $O(T d)$ all models
]

#cbox(title: [Calibration])[
  Goal: Confidence ≈ Accuracy
  *ECE*: $sum (|B_m|)/n |"acc"(B_m)-"conf"(B_m)|$
  *Temp Scaling*: $z/T$ on logits; $T>1$→less confident
]

= Active Learning

#cbox(title: [Objective])[
  $I(S)=I(f_S;y_S)=H[f_S]-H[f_S|y_S]$
  NP-hard; Greedy gives $(1-1/e)$-approx (submodular, monotone)
]

#cbox(title: [Strategies])[
  *Uncertainty Sampling*: $x=arg max H[y_x|D]$
  Cannot distinguish aleatoric vs epistemic
  *BALD*: $x=arg max I(theta;y_x|D)=H[y_x|D]-EE_theta[H[y_x|theta]]$
  Finds where models *disagree* about $y_x$
  *Hetero*: $x=arg max sigma^2_"epistemic"/sigma^2_"aleatoric"$
]

#cbox(title: [Submodular])[
  $F(A union {x})-F(A)>=F(B union {x})-F(B)$ for $A subset.eq B$
  Diminishing returns; MI is submodular
]

= Bayesian Optimization

#cbox(title: [Regret])[
  $R_T=sum_(t=1)^T (f^#opt -f(x_t))$
  Goal: sublinear $R_T/T arrow 0$
]

#cbox(title: [Acquisition Fncs])[
  *UCB*: $x_(t+1)=arg max[mu_t(x)+beta_t sigma_t(x)]$
  $beta_t=0$: pure exploit; $beta_t arrow infinity$: uncertainty sampling
  Regret: $R_T=O(sqrt(T gamma_T))$
  *PI*: $"PI"(x)=Phi((mu(x)-f(x^+))/(sigma(x)))$
  *EI*: $"EI"(x)=(mu-f^+)Phi(Z)+sigma phi(Z)$, $Z=(mu-f^+)/sigma$
  *Thompson*: Sample $tilde(f) tilde p(f|D_t)$, pick $arg max tilde(f)$
]

#cbox(title: [Info Gain $gamma_T$])[
  Linear: $gamma_T=O(d log T)$
  RBF: $gamma_T=O((log T)^(d+1))$
  Matérn($nu>1/2$): $gamma_T=O(T^(d/(2nu+d))(log T)^(2nu/(2nu+d)))$
]

= MDP & RL Foundations

#cbox(title: [MDP])[
  $(cal(S),cal(A),P,R,gamma)$: states, actions, $P(s'|s,a)$, reward, discount
  *Value*: $V^pi(s)=EE[sum_(t>=0)gamma^t R_t|s_0=s,pi]$
  *Q-fnc*: $Q^pi(s, a)=R(s,a)+gamma sum_(s')P(s'|s,a)V^pi(s')$
]

#cbox(title: [Bellman Eqs])[
  *Expectation*: $V^pi(s)=R(s,pi(s))+gamma sum_(s')P(s'|s,pi(s))V^pi(s')$
  *Optimality*: $V^#opt (s)=max_a[R(s,a)+gamma sum_(s')P(s'|s,a)V^#opt (s')]$
  $Q^#opt (s,a)=R(s,a)+gamma sum_(s')P(s'|s,a)max_(a')Q^#opt (s',a')$
  *Matrix*: $bold(v)^pi=(bold(I)-gamma bold(P)^pi)^(-1)bold(r)^pi$
]

#cbox(title: [Bellman's Thrm])[
  $pi^#opt$ optimal iff greedy w.r.t. own $V^pi$:
  $pi^#opt (s)=arg max_a Q^#opt (s,a)$
]

#cbox(title: [PI & VI])[
  *Policy Iter*: (1)Eval $V^pi$ exactly(solve LSE), (2)$pi arrow$greedy. Fewer iters, $O(n^3)$/iter.
  *Value Iter*: $V arrow max_a[r+gamma P V]$. More iters, $O(n^2 m)$/iter.
  Both converge to optimal; VI gives $epsilon$-optimal
]

#cbox(title: [POMDP])[
  *Belief*: $b_t(x)=P(X_t=x|y_(1:t),a_(1:t-1))$
  *Bayes Filter*: $b_(t+1)(x) prop o(y_(t+1)|x)sum_(x')P(x|x',a_t)b_t(x')$
  Belief-state MDP: reward $rho(b, a)=EE_(x tilde b)[r(x,a)]$
]

= Tabular RL

#cbox(title: [Model-based])[
  $hat(P)(x'|x,a)=N(x'|x,a)/N(a|x)$, $hat(r)(x,a)=$avg rewards
  Converges but needs many samples
]

#cbox(title: [Q-Learning (Off-policy)])[
  $Q(s,a) arrow.l Q(s,a)+alpha(r+gamma max_(a')Q(s',a')-Q(s,a))$
  Uses $max$ (ideal best $a'$); off-policy
]

#cbox(title: [SARSA (On-policy)])[
  $Q(s,a) arrow.l Q(s,a)+alpha(r+gamma Q(s',a')-Q(s,a))$
  Uses actual $a'$ from policy; on-policy
]

#cbox(title: [TD Learning])[
  $V(s) arrow.l V(s)+alpha(r+gamma V(s')-V(s))$
  *As SGD*: $ell=1/2(V(s)-(r+gamma V(s')))^2$
  Converges if Robbins-Monro: $sum alpha_t=infinity$, $sum alpha_t^2<infinity$
]

#cbox(title: [Exploration])[
  *$epsilon$-greedy*: prob $epsilon$ random, else best
  *Optimistic Init*: $Q=R_max/(1-gamma)$
  *Rmax*: unknown$(s,a) arrow R_max$, PAC guarantee
]

= Deep RL

#cbox(title: [DQN])[
  $cal(L)=(r+gamma max_(a')Q_(theta^-)(s',a')-Q_theta(s, a))^2$
  *Target Net* $theta^-$: stabilize; *Experience Replay*: break correlation
  *Double DQN*: selection $theta$, eval $theta^-$; reduces overestimation
]

#cbox(title: [Policy Gradient])[
  $nabla_theta J=EE_(tau tilde pi_theta)[sum_t nabla log pi_theta(a_t|s_t)G_t]$
  $nabla log P(tau)=sum_t nabla log pi(a_t|s_t)$ (dynamics cancel!)
  *REINFORCE*: MC estimate, high variance
  *Baseline*: $G_t-b(s_t)$, $b=V(s)$ optimal; unbiased
]

#cbox(title: [Actor-Critic])[
  *Actor*: $pi_theta(a|s)$; *Critic*: $V_phi(s)$ or $Q_phi(s, a)$
  $nabla J approx EE[nabla log pi(a|s)(Q(s,a)-V(s))]$
  Critic bootstrap减variance但引入bias
]

#cbox(title: [Advanced])[
  *TRPO*: $max EE[(pi_theta/pi_"old")A^(pi_"old")]$ s.t. $"KL"<=delta$
  *DDPG*: continuous actions, deterministic $mu_theta(s)$
  *Adv Fnc*: $A^pi(s, a)=Q^pi(s, a)-V^pi(s)$
]

= Bayesian Networks

#cbox(title: [Def])[
  DAG $G$ + CPDs $P(X_v|"Pa"_(X_v))$
  *Joint*: $P(X_(1:n))=product_i P(X_i|"Pa"_i)$
  Variable order重要 for compact representation
]

#cbox(title: [D-Separation])[
  $X perp Y|Z$ iff all paths blocked by $Z$
  *Active trails* (path通):
  Chain $X arrow Y arrow Z$: Y *not* observed
  Fork $X arrow.l Y arrow Z$: Y *not* observed
  Collider $X arrow Y arrow.l Z$: Y or descendant *is* observed
]

#cbox(title: [Inference])[
  *VE*: Sum out non-query vars; complexity = treewidth
  *BP (Sum-Product)*:
  $mu_(v arrow u) prop product mu_(u' arrow v)$
  $mu_(u arrow v) prop sum f_u product mu_(v' arrow u)$
  Exact on trees; loopy可能不converge
  *Max-Product*: replace $sum$ with $max$ for MPE/MAP
]

#cbox(title: [Approx Inference])[
  *Rejection Sampling*: discard samples不符evidence, inefficient if rare
  *Likelihood Weighting*: weight by evidence prob
  *Gibbs*: sample each var from conditional given rest
]

#cbox(title: [Learning])[
  *Params*: $hat(theta)_(X_i|"Pa"_i)="count"(X_i,"Pa"_i)/"count"("Pa"_i)$ (MLE)
  *Structure*: Score-based, MLE score偏好fully connected
  *BIC*: $S_"BIC"=sum hat(I)(X_i;"Pa"_i)-(log N)/(2N)|G|$
  *Chow-Liu*: max spanning tree on MI weights→optimal tree BN
]

/*
非今年考点也
= Particle Filtering

#cbox(title: [Algo])[
  1.*Predict*: $x_i' tilde P(X_(t+1)|x_(i,t))$ propagate
  2.*Weight*: $w_i=P(y_(t+1)|x_i')$ likelihood
  3.*Resample*: N particles prop to $w_i$
  Approx: $P(x) approx 1/N sum_i delta_(x_i)(x)$
]

= Kalman Filter

#cbox(title: [System])[
  $X_(t+1)=F X_t+epsilon_t$, $epsilon_t tilde cal(N)(0,Sigma_x)$
  $Y_t=H X_t+eta_t$, $eta_t tilde cal(N)(0,Sigma_y)$
]

#cbox(title: [Update])[
  *Predict*:
  $hat(mu)_(t+1)=F mu_t$; $hat(Sigma)_(t+1)=F Sigma_t F^top+Sigma_x$
  *Kalman Gain*:
  $K_(t+1)=hat(Sigma)_(t+1)H^top(H hat(Sigma)_(t+1)H^top+Sigma_y)^(-1)$
  *Update*:
  $mu_(t+1)=hat(mu)_(t+1)+K_(t+1)(y_(t+1)-H hat(mu)_(t+1))$
  $Sigma_(t+1)=(I-K_(t+1)H)hat(Sigma)_(t+1)$
]

#cbox(title: [Intuition])[
  Small obs noise→large $K$→trust obs
  Small proc noise→small $K$→trust predict
  Complexity: $O(d^3)$/step (matrix inv)
  KF = closed-form Bayesian filter for linear Gaussian
]
*/

= Diffusion Models

#cbox(title: [Setup])[
  *Forward*: data→noise (fixed, no learning)
  *Backward*: noise→data (learned generation)
  Latent var model: $x_(1:T)$ are latents, $x_0$ is data
]

#cbox(title: [Forward Process])[
  $q(x_t|x_(t-1))=cal(N)(x_t;sqrt(1-beta_t)x_(t-1),beta_t I)$
  $x_t=sqrt(1-beta_t)x_(t-1)+sqrt(beta_t)epsilon_t$
  Schedule: $beta_t in (0,1)$单调增, $beta_1 approx 10^(-4)$, $beta_T approx 0.02$
]

#cbox(title: [Closed-Form Marginal ⭐])[
  Define: $alpha_t=1-beta_t$, $bar(alpha)_t=product_(s=1)^t alpha_s$
  $q(x_t|x_0)=cal(N)(sqrt(bar(alpha)_t)x_0,(1-bar(alpha)_t)I)$
  *Reparam*: $x_t=sqrt(bar(alpha)_t)x_0+sqrt(1-bar(alpha)_t)epsilon$, $epsilon tilde cal(N)(0,I)$
  As $t arrow T$: $bar(alpha)_T arrow 0$, $x_T tilde cal(N)(0,I)$ indep of $x_0$
]

#cbox(title: [Reverse Process])[
  $p_lambda(x_(t-1)|x_t)=cal(N)(mu_lambda(x_t, t),sigma_t^2 I)$
  Prior: $p(x_T)=cal(N)(0,I)$
  Generate: sample $x_T$, iteratively denoise to $x_0$
]

#cbox(title: [Forward Posterior])[
  $q(x_(t-1)|x_t,x_0)=cal(N)(tilde(mu)_t,tilde(beta)_t I)$
  $tilde(mu)_t=(sqrt(bar(alpha)_(t-1))beta_t)/(1-bar(alpha)_t)x_0+(sqrt(alpha_t)(1-bar(alpha)_(t-1)))/(1-bar(alpha)_t)x_t$
  $tilde(beta)_t=((1-bar(alpha)_(t-1))beta_t)/(1-bar(alpha)_t)$
  Key: given $x_0,x_t$, forward posterior is Gaussian (tractable)
]

#cbox(title: [ELBO & Loss])[
  $cal(L)="const"-sum_(t=2)^T underbrace("KL"(q(x_(t-1)|x_t,x_0)||p_lambda(x_(t-1)|x_t)), L_t)$
  Two Gaussians same var: $"KL" prop ||mu_1-mu_2||^2$
]

#cbox(title: [⭐Noise Prediction ])[
  Predict $epsilon$ instead of $mu$ (more stable):
  From $x_t=sqrt(bar(alpha)_t)x_0+sqrt(1-bar(alpha)_t)epsilon$:
  $tilde(mu)_t=1/sqrt(alpha_t)(x_t-(beta_t)/sqrt(1-bar(alpha)_t)epsilon)$
  *Simple Loss*: $L_"simple"=EE_(t,x_0,epsilon)[||epsilon-epsilon_lambda(x_t, t)||^2]$
]

#cbox(title: [Training Algo])[
  Repeat: sample $x_0 tilde p_"data"$, $t tilde "Unif"{1,...,T}$, $epsilon tilde cal(N)(0,I)$
  $x_t=sqrt(bar(alpha)_t)x_0+sqrt(1-bar(alpha)_t)epsilon$
  $nabla_lambda||epsilon-epsilon_lambda(x_t, t)||^2$
]

#cbox(title: [Sampling Algo])[
  $x_T tilde cal(N)(0,I)$
  For $t=T,...,1$: $z tilde cal(N)(0,I)$ if $t>1$ else $z=0$
  $x_(t-1)=1/sqrt(alpha_t)(x_t-(beta_t)/sqrt(1-bar(alpha)_t)epsilon_lambda(x_t, t))+sigma_t z$
]

#cbox(title: [Connection])[
  $epsilon_lambda(x_t, t) approx -sqrt(1-bar(alpha)_t)nabla_(x_t)log q(x_t)$
  *Denoising = Score matching*
]

#cbox(title: [Variants])[
  *LDM*: diffusion in VAE latent space, more efficient
  *DDIM*: deterministic sampling, fewer steps
  *Cond Gen*: $epsilon_lambda(x_t, t, c)$, Classifier-Free Guidance:
  $tilde(epsilon)=(1+w)epsilon_lambda(x_t, t, c)-w epsilon_lambda(x_t, t)$
]

//||========================================||
= QuickCheck:
- *VI*: Approx posterior via ELBO. Laplace `MAP`, Reparam for grad.
- *MCMC*: Sample posterior. MH accept/reject, Gibbs coordinate, Langevin uses $nabla$.
- *GP*: Prior over fncs, closed-form posterior. RBF smooth, Matérn tunable.
- *BNN*: Prior on weights, MC predictive. Aleatoric=data noise, Epistemic=model.
- *Active*: Max MI, BALD for disagreement, submodular→greedy$(1-1/e)$.
- *BO*: UCB balance explore/exploit, EI expected gain, Thompson sample.
- *BN*: DAG factorization, d-sep for indep, BP exact on trees.
- *KF*: Linear Gaussian, Kalman gain trades predict vs observe.
- *Diffusion*: Forward=noise, Backward=denoise, train predict $epsilon$.
- *On/Off*: On=SARSA,REINFORCE,PPO; Off=Q-learn,DQN,SAC
- *Bellman*: $V=R+gamma P V$;
