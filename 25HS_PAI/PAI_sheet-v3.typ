#import "../assets/tmp_sht.typ": *
#show: project.with(authors: ((name: "", email: ""),))

// ========== 参数配置 ==========
#let fsize = 9.5pt //7.5pt
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


= 1. Probability & Info Theory

#cbox(title: [MLE, MAP])[
  $hat(theta)_"MLE"=arg max_theta product_i p(y_i|x_i,theta)=arg max_theta sum_i log p(y_i|x_i,theta)$

  *GaussianNoiseLinReg*: $y=theta^top x+epsilon$, $epsilon tilde cal(N)(0,sigma^2)$
  MLE ≡ 最小二乘: $hat(theta)_"MLE"=arg min sum_i (y_i-theta^top x_i)^2=(X^top X)^(-1)X^top y$

  $hat(theta)_"MAP"=arg max_theta p(theta|D)=arg max_theta p(D|theta)p(theta)$
  $
    =arg min_theta underbrace(-log p(theta), "正则")+underbrace(-log p(D|theta), "拟合")
  $

  *Prior→Regularizer*:
  Gaussian $cal(N)(0,sigma_p^2 I)$ → L2: $lambda/(2)||theta||^2$, $lambda=1/sigma_p^2$
  ; $"Laplace"(0,b)$ → L1: $lambda||theta||_1$, $lambda=1/b$
]

#cbox(title: [Posterior ∝ Likelihood × Prior])[

  $p(theta|D) prop p(D|theta)dot p(theta)$, log之,
  $log p(theta|D)=log p(D|theta)+log p(theta)-log p(D)$, where $log p(D)$"const w.r.t" $theta$.
]

#cbox(title: [Prob])[
  *Product*: $PP(X_(1:n))=PP(X_1)product_(i=2)^n PP(X_i|X_(1:i-1))$
  *Bayes*: $PP(X|Y)=(PP(Y|X)PP(X))/(PP(Y))$
  *Cond Indep*: $X perp Y|Z arrow.l.r PP(X, Y|Z)=PP(X|Z)PP(Y|Z)$
  *Tower*: $EE_Y [EE_X[X|Y]]=EE[X]$
  *LOTV*: $VV[X]=EE[VV[X|Y]]+VV[EE[X|Y]]$
]

#cbox(title: [Gaussian性])[
  $cal(N)(x;mu,Sigma)=(exp(-1/2(x-mu)^top Sigma^(-1)(x-mu)))/sqrt((2pi)^d|Sigma|)$
  *Marginal*: $X_A tilde cal(N)(mu_A,Sigma_(A A))$
  *Conditional*: $X_A|X_B tilde cal(N)(mu_(A|B),Sigma_(A|B))$
  $mu_(A|B)=mu_A+Sigma_(A B)Sigma_(B B)^(-1)(x_B-mu_B)$;
  $Sigma_(A|B)=Sigma_(A A)-Sigma_(A B)Sigma_(B B)^(-1)Sigma_(B A)$

  *Linear*: $Y=M X tilde cal(N)(M mu,M Sigma M^top)$

  *Sum*: indep $X+X' tilde cal(N)(mu+mu',Sigma+Sigma')$

  *Product* (两高斯相乘): $cal(N)(mu_1,Sigma_1)dot cal(N)(mu_2,Sigma_2) prop cal(N)(mu,Sigma)$
  $Sigma^(-1)=Sigma_1^(-1)+Sigma_2^(-1)$ (precision相加!)
  $mu=Sigma(Sigma_1^(-1)mu_1+Sigma_2^(-1)mu_2)$
  *Precision Matrix*: $Lambda:=Sigma^(-1)$ (inverse covariance)
  对角项: 条件方差倒数; 非对角: 条件相关性
]

#cbox(title: [Var & Cov])[
  $VV[X]=EE[(X-EE[X])^2]$; $"Cov"[X,Y]=EE[X Y]-EE[X]EE[Y]$
  $VV[a X+b Y]=a^2 VV[X]+b^2 VV[Y]+2a b"Cov"[X,Y]$
  $VV[a X-b Y]=a^2 VV[X]+b^2 VV[Y]-2a b"Cov"[X,Y]$
  $VV[X+Y]=VV[X]+VV[Y]+2"Cov"[X,Y]$
  $VV[X-Y]=VV[X]+VV[Y]-2"Cov"[X,Y]$
]

#cbox(title: [Info Theory])[
  *Entropy*: $H[p]=-EE_p [log p(x)]$; Gauss: $H=1/2 log((2pi e)^d|Sigma|)$
  *KL*: $"KL"(p||q)=EE_p [log p/q]>=0$; Need $"supp"(q)subset.eq"supp"(p)$
  Forward $"KL"(p||q)$: mean-seeking; Reverse $"KL"(q||p)$: mode-seeking
  *MI*: $I(X;Y)=H[X]-H[X|Y]=H[Y]-H[Y|X]>=0$
  *Info Gain公式*:
  $I(X;Y,Z)=I(X;Y)+I(X;Z|Y)$
  $I(X;Y,Z)-I(X;Y)=H[X|Y]-H[X|Y,Z]$ (条件减少熵!)
  *Info Never Hurts*: $I(X;Y)>=0$ and $H[X|Y]<=H[X]$
  观测$Y$不会增加$X$的不确定性
  *Cond MI*: $I(X;Y|Z)=H[X|Z]-H[X|Y,Z]$
  *Gauss MI*: $I[X;Y]=1/2 log det(I+sigma_n^(-2)Sigma)$ for $Y=X+epsilon$
]

= 2. BLR: Linear Kernel GP

#cbox(title: [Model])[
  $y=w^top x+epsilon$, $epsilon tilde cal(N)(0,sigma_n^2)$
  *Prior*: $w tilde cal(N)(0,sigma_p^2 I)$ (L2正则/weight decay)
  *Posterior*: $w|X,y tilde cal(N)(mu,Sigma)$
  $Sigma^(-1)=sigma_n^(-2)X^top X+sigma_p^(-2)I$ (只依赖$X$!)
  $mu=sigma_n^(-2)Sigma X^top y$
]

#cbox(title: [Prediction])[$x_*^top Sigma x_*$,"epistemic";$sigma_n^2$,"aleatoric":
  $y_*|x_*,X,y tilde cal(N)(x_*^top mu,x_*^top Sigma x_*)+sigma_n^2)$;
  *MAP=Ridge*: $hat(w)=(X^top X+lambda I)^(-1)X^top y$, $lambda=sigma_n^2/sigma_p^2$
  *BLogR*: Logistic Regression无闭式解 (非高斯likelihood)
  需VI/Laplace/MCMC近似posterior
]

#cbox(title: [Online Update (Woodbury)])[
  $(A+x x^top)^(-1)=A^(-1)-(A^(-1)x x^top A^(-1))/(1+x^top A^(-1)x)$ $O(d^2)$
  New data$(x,y)$: $Sigma_(t+1)=Sigma_t-(Sigma_t x x^top Sigma_t)/(1+x^top Sigma_t x)$
  $mu_(t+1)=Sigma_(t+1) (Sigma_t^(-1)mu_t+y x)$
]

= 3. Gaussian Processes

// #cbox(title: [Def])[
// $f tilde cal(G P)(mu,k)$. 有限子集$f_A tilde cal(N)(mu_A,K_(A A))$
// *量纲*: $[k(x,x')]=[f]^2=[y]^2$ (协方差=值平方)
// $k(x,x')="Cov"[f(x),f(x')]$
// ]

#cbox(title: [Def])[
  $y_i = f(x_i) + epsilon_i$, noise $epsilon_i tilde cal(N)(0, sigma_n^2)$.
  *Prior*: $f tilde cal(G P)(mu(x), k(x, x'))$.
  Finite set $A={x_1, ..., x_m}$, the vector $f(A)$多维Gaussian, $f(X) tilde cal(N)(mu(A), K_(A A))$, $[K_(A A)]_(i j)=k(x_i,x_j) in RR^(m times m)$.
  $k(x_i, x_i)$:each points自由度/方差; $k(x_i, x_j)$ points间通信/耦合强度.
]

#cbox(title: [GPR])[
  trainingdata-set $A={x_1, ..., x_m}$, observed value-set $y_A$, prior mean $mu(x)$, prior mean vector $bold(mu(A))$.
  $y tilde cal(N)(0,K_(A A)+sigma_n^2 I)=cal(N)(0,K_y)$
  *Mean*: $mu_*(x_*)= mu(x_*) + k(x_*,A)K_y^(-1)(y_A -mu_A)$
  *Cov*: $k_*(x_*,x')=k(x_*,x')-k(x_*,A)K_y^(-1)k(A,x')$
  *Predictive*: $y_* tilde cal(N)(mu_*, k_*(x,x') +sigma_n^2)$
]

#cbox(title: [GPRposterior ])[
  $y tilde cal(N)(0,K_(X X)+sigma_n^2 I)=cal(N)(0,K_y)$
  *Mean*: $mu'(x)=k(x,X)K_y^(-1)y$
  *Cov*: $k'(x,x')=k(x,x')-k(x,X)K_y^(-1)k(X,x')$
  *Predictive*: $y_* tilde cal(N)(mu',k'+sigma_n^2)$
]


#cbox(title: [Kernels])[
  *Valid Kernel*: 1. PSD: $K$所有$lambda_i$$>=0$; 2. Closure: $k_1+k_2$, $c k$ ($c>0$), $k_1 dot k_2$, $exp(k)$ valid; 3. Gram matrix: $x^top K x>=0$ $forall x$

  *Linear*: $k(x,x')=x^top x'+sigma_0^2$有rank=1协方差matrix(→BLR!).
  *RBF*: $k=exp((-||x-x'||^2) /(2ell^2))$ smooth $sigma_f^2$: 纵向振幅; $ell$: 横向平滑
  *Laplace*: $k = exp(-r / ell)$ Rough, sharp peaks, $C^0$ cont
  *Cosine*: $k = cos(2 pi r / p)$ (Periodic, no decay)
  *Exponential*: $k=exp(-||x-x'||\/ell)$ rough
  *Matérn*: $nu=0.5$→Exp, $nu arrow infinity$→RBF, $nu$控制smoothness
  *Periodic*: $k=sigma^2 exp(-2/ell^2 sin^2((pi|x-x'|)/p))$
  *Closure*: $k_1+k_2$, $k_1 dot k_2$, $c dot k$, $exp(k)$仍valid kernel
  *Stationary*: $k(x,x')=k(x-x')$; *Isotropic*: $k=k(||x-x'||)$
]

#cbox(title: [Marginal似然])[
  $log p(y|X,theta)=-1/2 y^top K_y^(-1)y-1/2 log|K_y|+C$
  Data fit (第一项) vs Complexity (第二项)
  *多峰非凸*: 多个local最优 (不同$ell$可能相似似然)
]

#cbox(title: [limit分析])[
  $sigma_n^2 arrow infinity$: posterior →先验 (noise淹没data)
  $ell^2 arrow infinity$ (Linear): 信号>>noise→最小二乘回归
  只有1train点: 精确插值该点，elsewhere高不确定
]

#cbox(title: [Sparse GP])[
  $N$data, $M$inducing: $O(N M^2+M^3)$ vs 标准$O(N^3)$
  SoR/FITC/VFE/RFF: 低秩核近似
]

#cbox(title: [Precision Matrix])[
  $Lambda:=Sigma^(-1)$ (Covariance的逆)
  *diag项* $Lambda_(i i)$: 条件方差$1/VV[X_i|X_(-i)]$
  *non-diag* $Lambda_(i j)$: 条件相关性 (给定其他变量)
  ; $Lambda$稀疏→条件独立结构 (Graphical Lasso); Gaussian Product: $Sigma^(-1)=Sigma_1^(-1)+Sigma_2^(-1)$ (precision相加!)
]

= 4. VI & ELBO

#cbox(title: [动机])[
  积分$integral p(y_*|w)p(w|D)d w$难算
  *Laplace*: 峰值$cal(N)(w_"MAP",H^(-1))$
  *VI*: 用$q(theta)$近似$p(theta|D)$, min $"KL"(q||p)$
  *MCMC*: 直接采样$w_s tilde p(w|D)$
]

#cbox(title: [ELBO])[
  $cal(L)=EE_q [log p(y|theta)]-"KL"(q(theta)||p(theta))$;
  $log p(y)=cal(L)+ "KL"(q||p(theta|y))$
  Max ELBO ⇔ Min KL to posterior

  *等价OR*:
  $arg max_q cal(L) equiv arg min_q "KL"(q||p(theta|y))$
  $equiv arg min_q EE_q [log q(theta)-log p(y,theta)]$
]

#cbox(title: [Min KL ≡ Max Likelihood])[
  当$q(theta)=delta(theta-hat(theta))$ (point estimate):
  $min_theta "KL"(delta||p(theta|y))equiv max_theta p(y|theta)p(theta)$ (MAP!)
  当prior uniform: MAP→MLE
  *非参数族*: VI受$q$族限制，无法精确recover真实posterior
  例: 真实posterior multimodal, $q$强制unimodal Gaussian→mode collapse
]

#cbox(title: [Gaussian KL])[
  1维: $"KL"(cal(N)_p||cal(N)_q)=1/2[(mu_p-mu_q)^2/sigma_q^2+sigma_p^2/sigma_q^2-1-log(sigma_p^2/sigma_q^2)]$
  多维: $"KL"=1/2["tr"(Sigma_q^(-1)Sigma_p)+(mu_p-mu_q)^top Sigma_q^(-1)(mu_p-mu_q)-d+log(|Sigma_q|/|Sigma_p|)]$
]

#cbox(title: [Reparam Trick])[
  *公式*: $theta=g(epsilon;lambda)$, $epsilon tilde p(epsilon)$ (indep $lambda$)
  $EE_(q_lambda)[f(theta)]=EE_p [f(g(epsilon;lambda))]$
  $nabla_lambda EE_q [f]=EE_p [nabla_lambda f(g(epsilon;lambda))]$ 移入Grad;
  *Gaussian*: $theta=mu+sigma dot.circle epsilon$, $epsilon tilde cal(N)(0,I)$
  $nabla_mu EE[f(theta)]=EE[nabla_mu f(mu+sigma epsilon)]$
  $nabla_sigma EE[f(theta)]=EE[nabla_sigma f(mu+sigma epsilon)]$
  *性质*: 连续可微变量; Unbiased; Low variance

  *Score Fnc*: $EE[f nabla log q]$ 离散/连续均可; High variance
]

#cbox(title: [Laplace近似])[
  $q(theta)=cal(N)(hat(theta),Lambda^(-1))$
  $hat(theta)="MAP"$; $Lambda=-nabla^2 log p(hat(theta)|D)$ (Hessian)
  Unimodal分布准确; Multimodal失效; Mode处好, elsewhere过confident
]

= 5. BNN & Uncertainty

#cbox(title: [model])[
  *Prior*: $theta tilde cal(N)(0,sigma_p^2 I)$
  *Homoscedastic*: $y|x,theta tilde cal(N)(f(x;theta),sigma_n^2)$ (固定noise)
  *Heteroscedastic*: $y tilde cal(N)(f_mu,exp(f_sigma))$ (输入依赖)
  $sigma^2(x)=exp(f_sigma (x;theta))$ 保证$>0$
]

#cbox(title: [Posterior Log-Density])[
  $log p(theta|D) prop log p(theta)+sum_i log p(y_i|x_i,theta)$
  $=-1/(2sigma_p^2)||theta||^2-sum_i [1/2 log sigma^2(x_i)+1/2((y_i-mu(x_i))^2)/(sigma^2(x_i))]$
  model可"blame"noise但付$log sigma$代价
]

#cbox(title: [Comparison $σ²$])[
  $sigma_p^2$: Prior variance (weight prior)
  $sigma_n^2$: Aleatoric noise (观测noise, 固定)
  $sigma^2(x)$: Heteroscedastic noise (输入依赖)
  $sigma_"epi"^2$: Epistemic (model不确定, data增加→减少)
  $sigma_"ale"^2$: Aleatoric (datanoise, data增加不变)
  ; epistemic=认知=可学习; aleatoric=偶然=不可减
]

#cbox(title: [Uncertainty])[
  $VV_"total" [y]=underbrace(EE_theta [VV[y|theta]], "aleatoric")+underbrace(VV_theta [EE[y|theta]], "epistemic")$
  *BLR闭式*: $sigma_"epi"^2=x^top Sigma_"post" x$; $sigma_"ale"^2=sigma_n^2$
  *GPR闭式*: $sigma_"epi"^2=k'(x,x)$; $sigma_"ale"^2=sigma_n^2$
  *MC近似* ($m$采样): $theta_j tilde q(theta)$
  Aleatoric: $1/m sum_j sigma^2(x,theta_j)$
  Epistemic: $1/m sum_j (mu(x, theta_j)-macron(mu))^2$
]

#cbox(title: [MC Dropout ])[
  $q_j=p delta_0+(1-p)delta_lambda$ (Bernoulli mask)
  *Training*: Dropout开
  *Inference*: Dropout保持开→多次forward→uncertainty estimate
  *本质*: Variational inference! $q$是Bernoulli×Gaussian混合
  近似$p(theta|D)$但限制在特定族
  *vs Gaussian Dropout*: 加性noise$w+epsilon$ vs 乘性mask $w dot m$
]

#cbox(title: [方法对比])[
  *SWAG*: SGD轨迹avg, $O(d^2)$ 存$Sigma$
  *Ensembles*: 多model独立train
  *Calibration*: ECE=$sum|"acc"-"conf"|$; Temp Scaling: $z/T$
]

= 6. Active Learning & BO

#cbox(title: [Info Gain目标])[
  $I(S)=I(f_S;y_S)=H[f_S]-H[f_S|y_S]$
  *Submodular*: $F(A union{x})-F(A)>=F(B union{x})-F(B)$, $A subset.eq B$
  Greedy: $(1-1/e)$-approx; NP-hard最优
]

#cbox(title: [对比])[
  *Uncertainty*: $x=arg max H[y_x|D]$
  Homo时OK; Hetero失效 (混淆aleatoric/epistemic)
  *BALD*: $x=arg max I(theta;y_x|D)=H[y_x|D]-EE_theta [H[y_x|theta]]$
  找model disagreement
  *Hetero修正*: $I(f;y|x)=1/2 log(1+sigma_"epi"^2\/sigma_"ale"^2)$
  考虑SNR而非纯variance
]

#cbox(title: [BO Acquisition])[
  *UCB*: $mu+beta sigma$; $beta=0$→exploit; $beta arrow infinity$→explore
  *PI*: $Phi((mu-f^+)/sigma)$ 保守
  *EI*: $(mu-f^+)Phi(Z)+sigma phi(Z)$ 平衡
  *Thompson*: 采样$tilde(f) tilde p(f|D)$, $arg max tilde(f)$
]

#cbox(title: [Regret & Info Gain])[
  $R_T=sum(f^opt-f(x_t))$; Sublinear: $R_T/T arrow 0$
  $R_T=O(sqrt(T gamma_T))$ for UCB
  Linear: $gamma_T=O(d log T)$
  RBF: $gamma_T=O((log T)^(d+1))$
]

= 7. MDP Foundations

#cbox(title: [MDP定义])[
  $(cal(S),cal(A),P,R,gamma)$: states, actions, transitions, reward, discount
  *Policy* $pi:cal(S)arrow cal(A)$ (或$pi(a|s)$ stochastic)
  *Stationary*: $pi$与时间$t$无关
  *Deterministic*: $pi(s)$ 单值; *Stochastic*: $pi(a|s)$ prob分布
]

#cbox(title: [Value Fnc])[
  $V^pi (s)=EE^pi [sum_(t=0)^infinity gamma^t R_t|s_0=s]$
  $Q^pi (s,a)=EE^pi [sum_(t=0)^infinity gamma^t R_t|s_0=s,a_0=a]$
  *V&Q*: $V^pi (s)=sum_a pi(a|s)Q^pi (s,a)$
  $Q^pi (s,a)=R(s,a)+gamma sum_(s')P(s'|s,a)V^pi (s')$
]

#cbox(title: [Bellman Expectation Eq])[
  $V^pi (s)=sum_a pi(a|s)[R(s,a)+gamma sum_(s')P(s'|s,a)V^pi (s')]$

  *Q value*版: $Q^pi (s,a)=R(s,a)+gamma sum_(s')P(s'|s,a)sum_(a')pi(a'|s')Q^pi (s',a')$

  *Matrix*: $bold(v)^pi=(bold(I)-gamma bold(P)^pi)^(-1)bold(r)^pi$ ($O(n^3)$求解)
]

#cbox(title: [BOE])[
  *Bellman算子*: $gamma$-contraction in $||dot||_infinity$

  $V^opt (s)=max_a [R(s,a)+gamma sum_(s')P(s'|s,a)V^opt (s')]$;
  $Q^opt (s,a)=R(s,a)+gamma sum_(s')P(s'|s,a)max_(a')Q^opt (s',a')$;

  *V&Q*: $V^opt (s)=max_a Q^opt (s,a)$
  $pi^opt (s)=arg max_a Q^opt (s,a)$
]

#cbox(title: [Optimal Policy定理])[
  $pi^opt$ optimal ⇔ greedy w.r.t. own $V^pi$
  $pi^opt (s)=arg max_a sum_(s')P(s'|s,a)V^opt (s')$
  有限MDP+$gamma<1$: 存在deterministic stationary $pi^opt$
]

#cbox(title: [PI vs VI对比])[
  *Policy Iter*:
  (1) Eval: 解LSE $V^pi=(I-gamma P^pi)^(-1)r^pi$ 精确
  (2) Improve: $pi'=arg max_a Q^pi$ greedy
  Fewer iters; $O(n^3)$/iter; 收敛到*exact* $pi^opt$
  *单调性*: $V^(pi_(k+1))>=V^(pi_k)$ 严格改进

  *Value Iter*:
  $V_(k+1)(s)=max_a [R(s,a)+gamma sum_(s')P(s'|s,a)V_k(s')]$
  More iters; $O(n^2 m)$/iter; 收敛到$epsilon$-optimal
  *收敛*: $||V_(k+1)-V_k||_infinity<epsilon arrow V_k approx V^opt$
]

#cbox(title: [Reward变])[
  *Scaling*: $R'=alpha R$ ($alpha>0$) → $pi^opt$不变, $V'=alpha V$
  *平移*: $R'=R+c$ → $pi^opt$**可能**变! $V'=V+c/(1-gamma)$
  $c>0$+$gamma arrow 1$→偏好长轨迹
  *Potential-based*: $F=gamma phi(s')-phi(s)$, $R'=R+F$ → $pi^opt$不变
]

#cbox(title: [POMDP])[
  *概念*: POMDP不可直接用VI/PI, 需转Belief-MDP (连续状态)
  *Belief*: $b_t (x)=PP(X_t=x|y_(1:t), a_(1:t-1))$
  *Bayes Filter*: $b_(t+1) prop o(y_(t+1)|x)sum_(x')P(x|x',a_t)b_t (x')$
  Belief-state MDP: $rho(b, a)=EE_(x tilde b)[r(x,a)]$

]

= 8. Tabular RL

#cbox(title: [Q-Learning])[Off-policy, Model-free;
  $
    Q_"new"=(1-alpha)Q_"old"+alpha[r+gamma max_(a')Q_"old"(s',a')]
  $
  *notation*: $Q_t$=处理$t$个样本后; $Q_0=0$ (或optimistic init)
  *用`max`*: 理想最优$a'$ (off-policy!)
  *Convergence*: Robbins-Monro ($sum alpha_t=infinity$, $sum alpha_t^2<infinity$) + 所有$(s,a)$访问无限次

  $bold(Q_(t+1)) (s,a)=bold(Q_t) (s,a)+alpha[r+gamma max_(a') bold(Q_t)(s',a')-bold(Q_t)(s,a)]$若$(s,a)$ visited;otherwise $Q_t (s,a)$不动.
]

#cbox(title: [SARSA (On-policy, Model-free)])[
  $Q_(t+1) (s,a)=Q_t (s,a)+alpha[r+gamma Q_t (s',a')-Q_t (s,a)]$
  *用实际$a'$*: policy执行的action (on-policy!)
  更保守; $a'$来自$epsilon$-greedy/$pi$
]

#cbox(title: [TD Learning (Policy Eval)])[
  $V_(t+1) (s)=V_t (s)+alpha[r+gamma V_t (s')-V_t (s)]$
  *As SGD*: $ell=1/2[V(s)-(r+gamma V(s'))]^2$
  $nabla_V ell=(V(s)-r-gamma V(s'))dot 1$
]

#cbox(title: [Model-based (学MDP)])[
  $hat(P)(s'|s,a)=N(s'|s,a)/N(s,a)$ (count visits)
  $hat(R)(s,a)=(sum_"visits"r)/N(s,a)$
  然后用$hat(P),hat(R)$做VI/PI
  *vs Model-free*: Q-learning直接学$Q$, 不估$P,R$
]

#cbox(title: [Exploration])[
  *$epsilon$-greedy*: prob $epsilon$ random
  *Optimistic Init*: $Q_0(s,a)=R_max/(1-gamma)$ (乐观探索)
  *Rmax*: unknown$(s,a) arrow R_max$; PAC保证
  *H-UCRL*: 乐观选最强model (OFU原则)
]

= 9. Deep RL

#cbox(title: [DQN])[Off-policy.
  *Target Net* $theta^-$: 每$C$步更新→稳定
  *Exp Replay*: buffer随机采样→打破时间相关
  $cal(L)=(r+gamma max_(a')Q_(theta^-)(s',a')-Q_theta (s,a))^2$
  *Double DQN*: $a^*=arg max Q_theta$; 用$Q_(theta^-)(s',a^*)$评估
  减maximization bias (Q-learning过估计)
]

#cbox(title: [Policy$nabla$Thrm])[
  $
    nabla_theta J=EE_(tau tilde pi_theta)[sum_(t=0)^T nabla_theta log pi_theta (a_t|s_t)G_t]
  $
  , deduction: $nabla log P(tau|theta)=sum_t nabla log pi(a_t|s_t)$
  $P(tau)=mu(s_0)product pi(a_t|s_t)product P(s_(t+1)|s_t,a_t)$
  dynamics$P(s'|s,a)$抵消(对$theta$求导为0)
  $G_t=sum_(t'=t)^T gamma^(t'-t)r_(t')$ (reward-to-go, 因果律!)
]

#cbox(title: [REINFORCE (On-policy)])[
  $theta arrow.l theta+alpha sum_t nabla log pi_theta (a_t|s_t)G_t$
  *MC估计*: 完整轨迹$tau$计算$G_t$
  *High variance*(引入baseline)
]

#cbox(title: [Baseline (Variance Reduction)])[
  $nabla J=EE[sum nabla log pi(G_t-b(s_t))]$ Unbiased if $b$与$a_t$无关!
  *Proof*: $EE_a [b(s)nabla log pi(a|s)]=b(s)nabla sum_a pi(a|s)=0$
  *Optimal*: $b(s)=V^pi (s)$ (若$nabla$近似常数)
  *Advantage*: $A^pi (s,a)=Q^pi (s,a)-V^pi (s)$
  $nabla J=EE[sum nabla log pi dot A]$
]

#cbox(title: [Actor-Critic])[
  *Actor*: $pi_theta (a|s)$; *Critic*: $V_phi (s)$ 或 $Q_phi (s,a)$
  $nabla J approx EE[nabla log pi(a|s)(Q(s,a)-V(s))]$
  Critic bootstrap→减variance但引入bias (若$V_phi$不准)
  *A2C*: Advantage Actor-Critic, on-policy
]

#cbox(title: [DDPG(连续 $cal(A)$)])[
  *Deterministic policy*: $a=mu_theta (s)$ (非prob!)
  *train加噪*: $a_"explore"=mu_theta (s)+cal(N)$ (OUnoise或Gaussian)
  无noise→无探索 (deterministic policy缺陷)
  *Actor*: $nabla_theta J=EE[nabla_a Q_phi (s,a)|_(a=mu)nabla_theta mu_theta (s)]$
  *Critic*: $(r+gamma Q_(phi^-)(s',mu_(theta^-))-Q_phi)^2$
  *Off-policy*: Exp Replay+Target Nets
  *vs MPC*: MPC硬算$H$步$G=sum r$; DDPG用TD $G=r+gamma Q$
]
#cbox(title: [$nabla$ Estimator Bias-Var])[
  #set text(size: 7pt)
  #table(
    columns: (auto, auto, auto, auto),
    [*Method*], [*Bias*], [*Var*], [*适用*],
    [MC ($G_t$)], [Unbiased], [High], [完整轨迹],
    [TD bootstrap], [Biased], [Low], [单步],
    [Baseline $G-b$], [Unbiased], [Lower], [$b$与$a$无关],
    [Actor-Critic], [Biased], [Low], [Critic不准时],
    [Reparam], [Unbiased], [Low], [连续可微],
    [Score/REINFORCE], [Unbiased], [High], [通用],
  )
  *Critic bias*: 若$V_phi approx V^pi$准确则unbiased
  *Baseline条件*: $b(s)$只依赖state, 不依赖action!
]

#cbox(title: [Advanced])[
  *PPO*: Clip$(pi/pi_"old")$ 限制update幅度; On-policy
  *SAC*: Entropy regularization $+lambda H(pi)$; Off-policy
  *TRPO*: $max EE[(pi/pi_"old")A]$ s.t. $"KL"<=delta$
]

#cbox(title: [RL Algo Check])[
  *On-policy*: SARSA, REINFORCE, A2C, PPO (data来自当前$pi$)
  *Off-policy*: Q-learn, DQN, DDPG, SAC, TD3 (用旧data/buffer)
  *Model-based*: Rmax, H-UCRL, PETS, Dyna-Q (学$P,R$)
  *Model-free*: Q-learn, PG, DQN (直接学$Q$/$pi$)
  *OFU (乐观探索)*: Rmax, H-UCRL (未知→高奖励)
  *Gradient估计*: Score Fnc (REINFORCE): High var, 离散/连续均可;Reparam (DDPG): Low var, 需连续可微

  *MCTS*: 蒙特卡洛树搜索, 模拟轨迹→UCB选择
  *MPC*: Model Predictive Control, horizon $H$→执行第1步→replan
  *MCTS vs MPC*: 都需model; MPC确定性规划, MCTS随机搜索
]

= 10. Diffusion Models

#cbox(title: [Def])[
  *Forward*: data→noise (固定$q$, 无学习)
  *Backward*: noise→data (学$p_lambda$)
  隐变量model: $x_(1:T)$ latents, $x_0$ data
]

#cbox(title: [Forward])[
  $q(x_t|x_(t-1))=cal(N)(sqrt(1-beta_t)x_(t-1),beta_t I)$
  $x_t=sqrt(1-beta_t)x_(t-1)+sqrt(beta_t)epsilon_t$
  Define: $alpha_t=1-beta_t$; $macron(alpha)_t=product_(s=1)^t alpha_s$


  *Closed-Form Marginal*: $q(x_t|x_0)=cal(N)(sqrt(macron(alpha)_t)x_0,(1-macron(alpha)_t)I)$
  $x_t=sqrt(macron(alpha)_t)x_0+sqrt(1-macron(alpha)_t)epsilon$
  $t arrow T$: $macron(alpha)_T arrow 0$, $x_T tilde cal(N)(0,I)$
]

#cbox(title: [Backward])[
  $p_lambda (x_(t-1)|x_t)=cal(N)(mu_lambda (x_t,t),sigma_t^2 I)$
  Prior: $p(x_T)=cal(N)(0,I)$
  *generate*: $x_T tilde cal(N)(0,I)$ → 迭代denoise → $x_0$
]

#cbox(title: [Forward Posterior])[
  $q(x_(t-1)|x_t,x_0)=cal(N)(tilde(mu)_t,tilde(beta)_t I)$ (给定$x_0,x_t$可算!)
  $tilde(mu)_t=(sqrt(macron(alpha)_(t-1))beta_t x_0+sqrt(alpha_t)(1-macron(alpha)_(t-1))x_t)/(1-macron(alpha)_t)$
  $tilde(beta)_t=((1-macron(alpha)_(t-1))beta_t)/(1-macron(alpha)_t)$
]

#cbox(title: [ELBO & noise预测])[
  $cal(L)="const"-sum_(t=2)^T "KL"(q(x_(t-1)|x_t,x_0)||p_lambda (x_(t-1)|x_t))$
  两Gauss间KL $prop ||mu_1-mu_2||^2$
  *Issue*: 直接预测$mu_lambda$不稳定 (目标依赖$x_0$)
  *Solution*: 预测*noise *$epsilon$!
  从$x_t=sqrt(macron(alpha)_t)x_0+sqrt(1-macron(alpha)_t)epsilon$反解:
  $tilde(mu)_t=1/sqrt(alpha_t)(x_t-(beta_t)/sqrt(1-macron(alpha)_t)epsilon)$
  *simple$cal(L)$*: $L_"simple"=EE_(t,x_0,epsilon)[||epsilon-epsilon_lambda (x_t,t)||^2]$
  *train*=predict noise; Backward=denoise;
]

#cbox(title: [train & 采样])[
  *Train*: 采样$(x_0,t,epsilon)$ → 算$x_t$ → $nabla||epsilon-epsilon_lambda||^2$
  *Sample*: $x_T tilde cal(N)(0,I)$
  For $t=T,...,1$: $z tilde cal(N)(0,I)$ if $t>1$ else $z=0$
  $x_(t-1)=1/sqrt(alpha_t)(x_t-(beta_t)/sqrt(1-macron(alpha)_t)epsilon_lambda (x_t,t))+sigma_t z$
]

#cbox(title: [与BNN连接])[
  *Latent var model*: $x_(1:T)$ 类似BNN hidden layers,
  *Reparam应用*: $x_t=sqrt(macron(alpha)_t)x_0+sqrt(1-macron(alpha)_t)epsilon$, where
  $epsilon$reparametrization变量;
  *VI框架*: Max ELBO → Min KL$(q||p_lambda)$
  Forward $q$已知 → Backward $p_lambda$待学
]

#cbox(title: [变体])[
  *LDM*: VAE, latent space diffusion (Stable Diffusion)
  *DDIM*: 确定性($z=0$), 加速20-50步
  *Cond*: $epsilon_lambda (x_t,t,c)$; Classifier-Free Guidance
]

#block(stroke: 0pt + black, inset: 3pt, width: 100%)[
  #set text(size: 9pt)
  = 速查 Quick Ref
  *Hoeffding*:$PP(|hat(X)-EE[X]|>=epsilon)<=2exp(-2n epsilon^2/(b-a)^2)$
  仅适用*iid* data; MCMC有自相关不适用
  *Bellman*: Expectation用$pi$求和; Optimality用$max$; $V=sum pi Q$; $Q=R+gamma P V$
  *PI vs VI*: PI精确Eval+少iters; VI单步+多iters; 都$O(n^3)$或$O(n^2 m)$
  *Q-learn vs SARSA*: Off用$max$ vs On用实际$a'$; 前者过估计后者保守
  *PG关键*: 动态抵消→$nabla=sum nabla log pi G_t$; Baseline减方差不bias (与$a$无关!)
  *DDPG*: 确定$mu$→必须加噪探索(OU/Gaussian); Off-policy+连续动作
  *Diffusion*: Forward固定加噪→Backward学denoise; 预测$epsilon$而非$mu$; ELBO=VI框架
  *Uncertainty*: Epi=model(data↑→↓); Ale=noise(data↑不变); Total=Ale+Epi
  *Info*: Never Hurts $H[X|Y]<=H[X]$; $I(X;Y,Z)=I(X;Y)+I(X;Z|Y)$
  *Kernel验证*: PSD (特征值$>=0$); Closure (和/积/指数); Gram $x^top K x>=0$
  *MAP vs MLE*: MAP=MLE+正则; Gauss prior→L2; Laplace→L1
  *ELBO*: Max $cal(L)$ ⇔ Min KL$(q||p)$ ⇔ Max似然(point estimate时)
  *Reparam*: $theta=mu+sigma epsilon$ → Grad移入$EE$ → Low var; Score高var
  *VI局限*: 受$q$族限制→无法精确recover (如multimodal→unimodal)
  *RL分类*: On=SARSA/REINFORCE/A2C/PPO; Off=Q/DQN/DDPG/SAC; Model-based=Rmax/PETS
  *OFU*: 乐观探索=未知高奖励; Rmax/H-UCRL; Optimistic Init $Q_0=R_max/(1-gamma)$
  *MCTS vs MPC*: 树搜索 vs 确定规划; 都需model; MPC执行1步replan

  Linear kernel GP ≡ BLR; Uniform prior时: MAP=MLE; VI能精确recover任意posterior ✗ (受$q$族限制); Entropy正则化→偏好stochastic/uniform; 边缘似然关于hyperparams是凸的 ✗ (通常多峰非凸!); 预测noise ≡ 预测mean(等价但noise更stable), Forward process无需learning. Contextual Bandit = MDP with $|S|=1$, $gamma=0$. $partial/(partial theta)log|K|="tr"(K^(-1)partial K/(partial theta))$
  $partial/(partial theta)(y^top K^(-1)y)=-y^top K^(-1)(partial K)/(partial theta)K^(-1)y$

  // *Jensen*: $f$ convex → $f(EE[X])<=EE[f(X)]$
  // $f$ concave → $f(EE[X])>=EE[f(X)]$
  // 应用: $log EE[X]>=EE[log X]$ (ELBO推导)

  // *Covariance*:
  // $M M^top$对角 ⇔ $M$行向量正交 ⇔ 独立(Gaussian)
  // $det(Sigma)=0$ ⇔ 退化(线性相关)

]


#block(stroke: 0pt + black, inset: 3pt, width: 100%)[
  #set text(size: 8pt)
  = Dict
  *A2C*:Advantage Actor-Critic; *BALD*:Bayesian Active Learning by Disagreement; *BLR*:Bayesian Linear Reg; *BNN*:Bayesian NN; *BO*:Bayesian Opt; *DDIM*:Denoising Diffusion Implicit; *DDPG*:Deep Deterministic PG; *DDPM*:Denoising Diffusion Prob; *DQN*:Deep Q-Net; *ECE*:Expected Calibration Error; *EI*:Expected Improvement; *ELBO*:Evidence Lower Bound; *GPR*:GP Regression; *H-UCRL*:Hoeffding-UCB RL; *LDM*:Latent Diffusion; *LOTV*:Law of Total Var; *MAP*:Max A Posteriori; *MCTS*:Monte Carlo Tree Search; *MI*:Mutual Info; *MLE*:Max Likelihood Est; *MPC*:Model Predictive Control; *NLL*:Negative Log-Likelihood; *OU*:Ornstein-Uhlenbeck; *PETS*:Probabilistic Ensemble w/ Trajectory Sampling; *PI*:Prob of Improvement; *PPO*:Proximal Policy Opt; *RBF*:Radial Basis Fnc; *RFF*:Random Fourier Features; *SAC*:Soft Actor-Critic; *SNR*:Signal-Noise Ratio; *SWAG*:Stoch Weight Avg Gaussian; *TD*:Temporal Diff; *UCB*:Upper Confidence Bound; *VI*:Variational Inference;
  ln2≈0.693; ln3≈1.099; 1/e≈0.368; √2≈1.414; (1-1/e)≈0.632;
  $a X+b Y tilde cal(N)(a mu_X+b mu_Y, a^2 sigma_X^2+b^2 sigma_Y^2)$ (indep);
  Bern [$p$], [$p(1-p)$]; Binomial($n,p$ [$n p$], [$n p(1-p)$];
  Poisson($lambda$) [$lambda$], [$lambda$];
  Exp($lambda$ [$lambda e^(-lambda x)$], [$1/lambda$], [$1/lambda^2$];
]

#cbox(title: [RL Algo Property])[
  #set text(size: 6.5pt)
  #table(
    columns: 6,
    [*Algo*], [*On/Off*], [*Model*], [*Data Eff*], [*Complexity*], [*Bias/Var*],
    [Q-learn], [Off], [Free], [High], [$O(|S||A|)$], [Unbiased],
    [SARSA], [On], [Free], [Low], [$O(|S||A|)$], [Unbiased],
    [DQN], [Off], [Free], [High], [Func Approx], [Biased(FA)],
    [DDPG], [Off], [Free], [High], [Func Approx], [Biased(FA)],
    [REINFORCE], [On], [Free], [Low], [Func Approx], [Unbiased/HiVar],
    [A2C], [On], [Free], [Med], [Func Approx], [Biased(Critic)],
    [PPO], [On], [Free], [Med], [Func Approx], [Biased(Clip)],
    [SAC], [Off], [Free], [High], [Func Approx], [Biased(Entropy)],
    [Rmax], [Both], [Based], [Low], [$O(|S|^3)$], [Unbiased],
    [PETS], [Off], [Based], [High], [Ensemble], [Biased(Model)],
  )
  *$nabla$ Estimator*:Score Fnc (REINFORCE): Unbiased, High Var, 离散/连续; Reparam (DDPG/SAC): Biased(若FA), Low Var, 连续only; Baseline: Unbiased iff 与action无关
]
