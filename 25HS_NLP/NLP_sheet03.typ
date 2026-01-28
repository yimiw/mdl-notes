#import "../assets/tmp_sht.typ": *
#show: project.with(authors: ((name: "", email: ""),))

// ========== Font Size ==========
#let fsize = 9pt
#let hsize1 = 9.5pt
#let hsize2 = 9pt
#let pspace = 0.15em
#let plead = 0.25em
// ================================

#set text(size: fsize)
#set par(spacing: pspace, leading: plead, justify: true, first-line-indent: 0em)
#show heading.where(level: 1): set text(size: hsize1)
#show heading.where(level: 2): set text(size: hsize2)
#show heading: box
#show heading: set text(fill: rgb("#663399"), weight: "bold")
#show: columns.with(4, gutter: 0.5em)

= 0. Intro   
#cbox(title: [Hypergraph View])[
  Computation graph = labeled acyclic *hypergraph*.
  Edges can have multiple sources/targets.
  *Complexity*: same time as $f$; space higher (store intermediates)
  vec-vec: $O(d)$; mat-vec: $O(n m)$; mat-mat: $O(n m ell)$
]
#cbox(title: [NLL $nabla$ = 0])[
  $sum_(i=1)^n bold(f)(x_i,y_i)=sum_(i=1)^n EE_(y|x_i,theta)[bold(f)(x_i,y)]$
  Observed features = Expected features
  *Hessian*: $bold(H)=sum_i "Cov"_(y|x_i,theta)[bold(f)(x_i,y)]$ (PSD!)
]
#cbox(title: [DAG Properties])[
  Topological order唯一确定; DP子问题独立拆分可行; Gradient反向传播良定义(no cycles)
  *Hypergraph*: 函数式计算自然表示, multi-inputs→one output
]

= 1. Backpropagation

#cbox(title: [Chain])[
  $d/(d x)[f(g(x))]=f'(g(x))g'(x)$
  *Jacobian*: $f: RR^n -> RR^m$, $(d y)/(d x)=[(d y)/(d x_1),...,(d y)/(d x_n)] in RR^(m times n)$
  *Multivar*: $(d y_i)/(d x_j)=sum_(k=1)^m (d y_i)/(d z_k)(d z_k)/(d x_j)$
]

#cbox(title: [Bauer Path])[
  $(d y_i)/(d x_j)=sum_(p in cal(P)(j,i)) product_((k,l) in p) (d z_l)/(d z_k)$
  $cal(P)(j,i)$=all paths $j->i$; worst $O(m^n)$, $m$平均出度, $n$路径长度
]

#cbox(title: [Forward vs Reverse])[
  *Forward*: expand $(d)/(d x)$ recursively, same flow as fwd
  *Reverse*: 2 passes—fwd compute vals, bwd compute grads
  *Complexity*: same time as $f$; higher space (store intermediates)
]

#cbox(title: [Primitives])[
  *Sum*: $(d(a+b))/(d a)=1$; *Prod*: $(d(a b))/(d a)=b$
]

= 2. Log-Linear Models

#cbox(title: [Prob Basics])[
  *Bayes*: $p(y|x)=(p(x|y)p(y))/(integral p(x|y)p(y) d y)$
  Posterior $prop$ Prior $times$ Likelihood
  *Marginal*: $p(x)=sum_y p(x,y)$
  *Expectation*: $EE[f(x)]=sum_x f(x)p(x)$
]

#cbox(title: [Log-Linear Model])[
  $p(y|x,theta)=(exp(theta dot f(x,y)))/(Z(theta))$
  $Z(theta)=sum_(y' in Y)exp(theta dot f(x,y'))$
  $log p(y|x,theta)=theta dot f(x,y)-log Z$ (linear in log space!)
  *Discrete MLE*: $p(y|x)="count"(x,y)/"count"(x)$ (sparse问题)
]

#cbox(title: [MLE $nabla$])[
  $theta_"MLE"=arg min_theta -sum_(n=1)^N log p(y_n|x_n,theta)$

观测特征count = 期望特征count → *Expectation Matching*
  $(d cal(L))/(d theta_k)=-sum_n f_k (x_n,y_n)+sum_n sum_(y')p(y'|x_n;theta)f_k (x_n,y')$
]

#cbox(title: [MAP & Ridge])[
  $hat(theta)_"MAP"=arg min[-log p(theta)-log p(D|theta)]$
  Gaussian prior $cal(N)(0,sigma_p^2 I)$ → L2: $lambda/2||theta||^2$
  Laplace prior → L1 regularization
]

#cbox(title: [Softmax])[
  $"sftm"(h,y,T)=(exp(h_y\/T))/(sum_(y')exp(h_(y')\/T))$
  $T->0$: argmax; $T->infinity$: uniform
  $log "sftm"=h_y-log sum_(y')exp(h_(y'))$ (logsumexp)
]

#cbox(title: [MLP Architecture])[
  *Problem*: Log-linear needs linearly separable data
  *Solution*: Learn non-linear feature fn
  $bold(h)_k=sigma_k (bold(W)_k^top bold(h)_(k-1))$, $bold(h)_1=sigma_1 (bold(W)_1^top bold(e)(x))$
  Output: $"sftm"(bold(theta)^top bold(h)_n)$
]

#cbox(title: [Activations])[
  $sigma(x)=1/(1+exp(-x))$, $nabla sigma=sigma(1-sigma)$
  *tanh*: $(1-e^(-2x))/(1+e^(-2x))$, $nabla=1-tanh^2$
  Sigmoid/tanh vanishing gradient→ use ReLU
  *Backprop*: $(partial ell)/(partial bold(W)_k)=(partial ell)/(partial y)(partial y)/(partial bold(h)_n)(product_(m=k+1)^n sigma'_m bold(W)_m)sigma'_k bold(h)_(k-1)$
]

#cbox(title: [Exp Family & MaxEnt])[
  $p(x|theta)=1/(Z(theta))h(x)exp(theta dot phi(x))$
  *Max Entropy*: $H(p)=-sum_x p(x)log p(x)$
  选最大熵分布=最少假设=Laplace原则
  *优势*: Conjugate priors; Sufficient stats; Convex log-partition → unique MLE
]

= 3. Language Models

#cbox(title: [Structured Prediction])[
  *Kleene $V^*$*: infinite set of finite-length strings from $V$
  *Language Model*: weighted prefix tree, each sentence=unique path
  $p(y)=1/Z product_(t=1)^(|y|)"weight"_(y_(<=t))$
]

#cbox(title: [Local Normalization])[
  $Z=1$ when children edges sum to 1 at each node
  *Consistency*: $p("EOS"|y_(<t),V^*)>epsilon>0$
  $p(|y|=infinity)<=lim_(t->infinity)(1-epsilon)^t=0$ (tight)
]

#cbox(title: [N-gram Model])[
  $p(y_t|y_(<t))=p(y_t|y_(t-1),...,y_(t-n+1))$
  *Markov*: $P(t_i|t_(1:i-1))=P(t_i|t_(i-1))$ (1st order)
  $=(exp(w_(y_t) dot h_t))/(sum_(y' in V)exp(w_(y') dot h_t))$, $h_t in RR^d$
  *Bengio*: $h_t=f(e("hist"))$, $e("hist")=[e(y_(t-1));e(y_(t-2));...]$
]

#cbox(title: [RNN])[
  $h_t=f(h_(t-1),e(y_(t-1)))$ (implicit infinite context)
  *Vanilla*: $h_t=sigma(W_1 h_(t-1)+W_2 e(y_(t-1)))$
  *BPTT*: unroll through time, sum grads over timesteps
]

= 4. Word Embeddings

#cbox(title: [Encoding])[
  *One-hot*: $v in O(|V|)$, only word=1
  *Bag-of-words*: pooled one-hot (sum/mean/max)
  *N-grams*: vectors huge—every combo needs slot
  *Pipeline*: Embedding → Pooling → NN → Softmax
]

#cbox(title: [Skip-gram])[
  *Preprocess*: word-context pairs ($k times C$ many), window $k$
  $p(c|w)=1/(Z(w))exp(e_"wrd"(w) dot e_"ctx"(c))$, $O(2|V|k)$ params
  *Bilinear*: linear if all-but-one vars held constant
  *Similarity*: $cos(u_i,u_j)$
]

= 5. CRF & POS Tagging

#cbox(title: [As Graph])[
  Fully connected graph w/ POS-tag nodes per layer
  $"score"(chevron.l D,N,V,... chevron.r,w)=theta f(t,w)$
  *score*$(t,w)$= unnormalized log-prob = $sum_n$trans$+$emit
  Problem: $O(|cal(T)|^N)$ paths in normalizer
]

#cbox(title: [CRF Model])[
  $p(t|w)=(exp("score"(t,w)))/(sum_(t' in cal(T)^N)exp("score"(t',w)))$
  *Decomposition*: $"score"(t,w)=sum_(n=1)^N "score"(chevron.l t_(n-1),t_n chevron.r,w,n)$
  $p(t|w) prop product_(n=1)^N exp{"score"(chevron.l t_(n-1),t_n chevron.r,w)}$
]

#cbox(title: [DP推导: $O(|T|^N)->O(N|T|^2)$])[
  Goal: $Z=sum_(bold(t) in cal(T)^N) exp"score"(bold(t),bold(w))$
  *Step1*: 可加分解 $"score"=sum_n "score"_n$
  *Step2*: $Z=sum_(bold(t)) exp sum_n "score"_n=sum_(bold(t)) product_n exp"score"_n$ (exp)
  *Step3*: $=sum_(t_1)...sum_(t_N) product_n exp"score"_n$ (展开)
  *Step4*: $=sum_(t_1) exp"score"_1 times (sum_(t_2)...)$ (distrib把内层sum推进去)
  *若3-gram*: 依赖$t_(n-2),t_(n-1),t_n$ → $O(N|cal(T)|^3)$
]

#cbox(title: [Forward Algorithm])[
  $alpha[0,t]=exp("score"("BOS"->t))$ (init w/ BOS trans)
  for $n=1,...,N-1$; for $t_n in cal(T)$:
  #h(1em) $alpha[n,t_n]=plus.o.big_(t_(n-1)) alpha[n-1,t_(n-1)] times.o exp("score")$
  return $plus.o.big_t alpha[N-1,t]$ (sum last column!)
  *直觉*: prefix之和, 从seq开头走到当前状态的所有走法score总和
]

#cbox(title: [Backward Algorithm])[
  $forall t_N: beta[N,t_N] <- bold(1)$
  for $n=N-1,...,0$; for $t_n in cal(T)$:
  #h(1em) $beta[n,t_n] <- plus.o.big_(t_(n+1)) exp("score"_(n+1)) times.o beta[n+1,t_(n+1)]$
  return $beta[0,"BOS"]$ (single value!)
  *Complexity*: $O(N|cal(T)|^2)$
]

#cbox(title: [Fwd vs Bwd Asymmetry])[
  *Init*: Bwd直接$bold(1)$; Fwd需BOS转移
  *Term*: Bwd返回$beta[0,"BOS"]$单值; Fwd需$plus.o$整列
  *原因*: BOS显式存在, EOS不显式处理
]

#cbox(title: [Viterbi Decoding])[
  $delta[n,t]=max_(t_(n-1))[delta[n-1,t_(n-1)]+"score"(t_(n-1),t)]$
  每步枚举$t$和$t_(n-1)$→$|cal(T)|^2$种trans
  *Backtrack*: 存$"argmax"$指针bp, 从$arg max_t delta[N,t]$回溯
]

#cbox(title: [Common Semirings])[
  #set text(size: 6.5pt)
  #table(
    columns: 7,
    [*Name*], [$bb(K)$], [$plus.o$], [$times.o$], [$bold(0)$], [$bold(1)$], [*用途*],
    [Real], [$RR_(>=0)$], [$+$], [$times$], [$0$], [$1$], [$Z$ partition],
    [Viterbi], [$RR union {-infinity}$], [$max$], [$+$], [$-infinity$], [$0$], [最优path],
    [Log], [$RR union {pm infinity}$], [lse], [$+$], [$-infinity$], [$0$], [$log Z$],
    [Boolean], [${0,1}$], [$or$], [$and$], [$0$], [$1$], [可达性],
    [Counting], [$NN$], [$+$], [$times$], [$0$], [$1$], [路径数],
    [Tropical], [$RR union {infinity}$], [$min$], [$+$], [$infinity$], [$0$], [最短路],
  )
]

#cbox(title: [Semiring Definition])[
  $chevron.l bb(K),plus.o,times.o,bold(0),bold(1) chevron.r$ where:
  1. $(bb(K),plus.o,bold(0))$: *comm monoid* (assoc+comm+identity)
  2. $(bb(K),times.o,bold(1))$: *monoid* (assoc+identity)
  3. *Distrib*: $(x plus.o y) times.o z=(x times.o z) plus.o (y times.o z)$
  4. *Annihilator*: $bold(0) times.o x=x times.o bold(0)=bold(0)$
  // *陷阱*: $bold(0)=bold(1)$必失败!
]

#cbox(title: [Semiring意义])[
  $plus.o$: *分治* (split points合并, OR/MAX/+)
  $times.o$: *连接* (左右子树组合, AND/$times$/+)
  $bold(0)$: 吸收元, 消除invalid; $bold(1)$: 单位元, null不破坏
]

#cbox(title: [Monoid判定])[
  1. *Closure*: $a times.o b in bb(K)$
  2. *Assoc*: $(a times.o b) times.o c=a times.o (b times.o c)$
  3. *Identity*: $exists bold(e): a times.o bold(e)=bold(e) times.o a=a$
]

#cbox(title: [Kleene Star])[
  $a^*=plus.o.big_(n=0)^infinity a^(times.o n)=bold(1) plus.o a times.o a^*$
  Real上$|a|<1$: $a^*=1/(1-a)$ (geometric series)
  Tropical: $a^*=0$ if $a>=0$ (正环不帮助)
  用于globally normalized LM
]

= 6. CFG Parsing

#cbox(title: [Constituents])[
  Multi-word units as single unit
  *Tests*: Pronoun substitution, Clefting, Answer ellipsis
  Ambiguity: PP attachment, modifier scope
]

#cbox(title: [CFG Definition])[
  $G=chevron.l cal(N),cal(S),Sigma,cal(R) chevron.r$
  Non-terminals, start symbol, terminals, production rules
  *CNF*: $N_1->N_2 N_3$ or $N->a$; $O(4^N)$ trees (Catalan)
]

#cbox(title: [Weighted CFG])[
  *Global*: $p(t)=1/Z product_(r in t)exp("score"(r))$
  $Z=sum_(t' in cal(T))product_(r')exp("score"(r'))$ (可能$infinity$!)
  *Probabilistic*: local norm $sum_k p(alpha_k|N)=1$
]

#cbox(title: [CKY Chart索引])[
  Position在*words之间*: $0|w_1|1|w_2|2|...|N$
  $"Chart"[i,k,X]$: span $[i,k)$覆盖$w_i,...,w_(k-1)$
  *长度*: $k-i$; *对角线*: $k-i=1$ (单词)
  *Fill order*: 按span长度递增($ell=1,2,...,N$)
  同一长度内任意顺序 (topo order自由度)
  *Goal*: $"Chart"[0,N,S]$
]

#cbox(title: [CKY algo])[
  $O(N^3|R|)$, needs CNF
  *Terminal*: $C[i,i+1,X]=exp"score"(X->w_i)$ for $X->w_i in cal(R)$
  *Binary*: for span$=2,...,N$; for $i=1,...,N-"span"$:
  $k<-i+"span"$; for $j=i+1,...,k-1$; for $X->Y Z in cal(R)$:
  $C[i,k,X] plus.o exp{"score"} times.o C[i,j,Y] times.o C[j,k,Z]$
]

#cbox(title: [CKY Chart 3×3 Example])[
  Sentence: $w_1 w_2 w_3$
  #set text(size: 6.5pt)
  #table(
    columns: 4,
    [], [1], [2], [3],
    [0], [$C[0,1]$], [$C[0,2]$], [$C[0,3]$←*goal*],
    [1], [], [$C[1,2]$], [$C[1,3]$],
    [2], [], [], [$C[2,3]$],
  )
  *Fill*: diag first, then by span length
]

= 7. Dependency Parsing

#cbox(title: [Dependency Tree])[
  Directed spanning tree, root degree 1
  *Constraints*: Single head; Connected; Acyclic
  *Projective*: arcs不交叉 (嵌套/并列) → CKY可用
  *Non-projective*: arcs可交叉 → 必须用CLE/MTT
  \# spanning trees: $O((n-1)^(n-2))$
]

#cbox(title: [Edge-Factored Model])[
  *Arc-factored*: 每条边独立打分，树score=边score之和
  *优点*: global优化分解为local边决策
  *局限*: 无法捕捉sibling/grandparent effects
  $"score"(t,bold(w))=sum_((i->j) in t) "score"(i->j,bold(w))+"score"(r,bold(w))$
  $p(t|w)=1/Z product_((i->j) in t)exp("score"(i,j,w))exp("score"(r,w))$
]

#cbox(title: [CLE关键步骤])[
  *Goal*: max spanning arborescence (directed MST)
  1. For each node $v$, pick max incoming edge
  2. If no cycle → done (it's a tree)
  3. If cycle → *contract* cycle to supernode
  4. *Reweight*: $omega'(u->v)=omega(u->v)-omega_"in-cycle"(v)$
  5. Recursively solve contracted graph
  6. *Expand*: break cycle at min-loss edge
  *Complexity*: $O(N^2)$ or $O(E+N log N)$
]

#cbox(title: [Cayley Formula])[
  *无向$K_n$*: $n^(n-2)$棵spanning trees
  *有向+固定root*: $n^(n-2)$棵arborescences
  *有向+任意root*: $n times n^(n-2)=n^(n-1)$棵
]

#cbox(title: [Graph Laplacian $L$])[
  $L_(i j)=cases(
    "Degree"(i) & i=j "(对角线)",
    -1 & i!=j and i tilde j "(有边)",
    0 & "otherwise"
  )$
  *trick*: 只看非对角$-1$判断边存在
  *MTT*: \#spanning trees = $det(hat(L))$ (any minor)
]

#cbox(title: [Weighted Laplacian (MTT)])[
  $A_(i j)=exp("score"(i->j))$, $rho_j=exp("score"(j,w))$
  $L_(i j)=cases(
    rho_j & i=1 "(root row)",
    sum_(k!=j) A_(k j) & i=j "(in-degree)",
    -A_(i j) & "else"
  )$
  $Z=det(L)$, 复杂度$O(n^3)$
]



#cbox(title: [Root Constraint])[
  CLE base允许多root outgoing arcs
  *Naive*: 对每条root arc分别运行CLE → $O(N dot "CLE")$
  *Clever* (Gabow): swap score=next-best - current
  删除swap score最小的多余root edge
]

#cbox(title: [MTT vs CLE])[
  // #set text(size: 6.5pt)
  // #table(
  //   columns: 3,
    [*维度*], [*MTT*], [*CLE*],
    [目标], [$Z=sum_t exp("score")$], [$t^*=arg max$],
    [算法], [$det(tilde(L))$], [Greedy+Contract],
    [复杂度], [$O(N^3)$], [$O(N^2)$],
  // )
]

= 8. Semantic Parsing

#cbox(title: [Syntax vs Semantics])[
  *Syntax*: structural org (parse tree)
  *Semantics*: underlying meaning
  *Logical form*: quantifiers, vars, boolean, predicates
  *Compositionality*: meaning of whole = fn of parts
]

#cbox(title: [Lambda Calculus])[
  *Terms*: 变量$x$; 抽象$lambda x.M$; 应用$(M N)$
  *$beta$-reduction*: $(lambda x.M)N ->_beta M[x:=N]$
  *$alpha$-conversion*: 重命名bound变量避免capture
  *$beta$-infinity*: $F=lambda x((x x)x)$, $F F=...$不终止
]

#cbox(title: [$beta$-reduction步骤])[
  1. 找到$(lambda x.M)N$形式的redex
  2. 在$M$中找所有被该$lambda x$绑定的$x$
  3. 将这些$x$替换为$N$
  *注意*: 可能需先$alpha$-convert避免变量捕获!
]

// #cbox(title: [$alpha$-conversion何时需要])[
//   $(lambda x.lambda y.x y)y$ 直接reduce得$lambda y.y y$—*错!*
//   原本外层$y$是free, 现在变bound了
//   *正确*: 先$alpha$-convert: $lambda y.x y ->_alpha lambda z.x z$
//   再reduce: $(lambda x.lambda z.x z)y ->_beta lambda z.y z$
// ]

#cbox(title: [Free vs Bound Variables])[
  $"FV"(x)={x}$; $"FV"(lambda x.M)="FV"(M)-{x}$
  $"FV"(M N)="FV"(M) union "FV"(N)$
  *Bound*: 在某$lambda$的scope内
  *Free*: 不在任何abstraction的scope内
]

#cbox(title: [Combinatory Logic])[
  $bold(I) x=x$; $bold(K) x y=x$; $bold(S) x y z=x z(y z)$
  $bold(B) x y z=x(y z)$ (comp); $bold(C) x y z=x z y$ (flip)
  $bold(T) x y=y x$ (type-raising)
  $bold(I)=bold(S) bold(K) bold(K)$ (S,K构成complete basis)
]

#cbox(title: [CCG Rules])[
  *Application*:
  $X\/Y space Y => X$ (>前向); $Y space X backslash Y => X$ (\<后向)
  *Composition*:
  $X\/Y space Y\/Z => X\/Z$ ($bold(B)_>$)
  *Type-raising*: $X => T\/(T backslash X)$ ($bold(T)_>$)
  rules是universal, language-specific全在lexicon
]

#cbox(title: [CCG Category直觉])[
  $S backslash "NP"$: 左边要NP → 产出S (intransitive)
  $(S backslash "NP")\/"NP"$: 右边要NP → $S backslash "NP"$ (transitive)
  *Slash方向*: $\/$ 向右找arg; $backslash$ 向左找arg
]

#cbox(title: "Derivation with Semantics")[
      Lexicon:
      - Mary : NP : $"Mary"$
      - likes : $(S backslash "NP") \/ "NP"$ : $lambda y. lambda x. "Likes"(x, y)$
      - John : NP : $"John"$

      Parse "Mary likes John":
      ```
      Mary        likes                    John
      NP:Mary (S\NP)/NP:λy.λx.Likes(x,y)  NP:John
                 ───────────────────────────── >
                            S\NP:λx.Likes(x,John)
      ──────────────────────────────────────── <
                        S:Likes(Mary,John)
      ```
    ]

#cbox(title: [LIG构造策略])[
  *问题*: CFG无法"计数" ($a^n b^n c^n$中$n$相等)
  *LIG*: 用stack记录计数信息
  *策略1*: 两端向中间—先生成首尾, 再生成中间
  *策略2*: 左向右—前半部分push, 后半部分pop
  *Example* $a^n b^n c^n d^n$:
  $S[sigma]->a S[f sigma]d$; $S[sigma]->T[sigma]$
  $T[f sigma]->b T[sigma]c$; $T[]->epsilon$
]

#cbox(title: [FOL Translation])[
  $forall$配$=>$:全称限定条件; 
  $exists$配$and$: 存在某具体对象;
  否则$exists$配$=>$:往往荒谬; $or$配$and$:要求满足多个条件.
]

= 9. WFST & Lehmann

#cbox(title: [Transducer Def])[
  $T=chevron.l Q,Sigma,Omega,lambda,rho,delta chevron.r$
  $Q$: states; $Sigma$: input; $Omega$: output
  $lambda: Q->RR$: initial; $rho: Q->RR$: final
  $delta: Q times (Sigma union epsilon) times (Omega union epsilon) times Q -> RR$
  *$epsilon$-transition*: no input/output consumed
]

#cbox(title: [FSA vs FST])[
  *WFSA* (单带): read only, $"score"(pi)=sum_n "score"(tau_n)$
  *WFST* (双带): read input + write output
  *Unambiguous*: $|Pi(x,y)|<=1$
  *Ambiguous*: $|Pi(x,y)|>1$ → need semiring
]

#cbox(title: [Path Score])[
  $"score"(pi)=lambda(q_"start")+sum_(n=1)^(|pi|)"score"(tau_n)+rho(q_"end")$
  $p(y|x)=1/Z sum_(pi in Pi(x,y))exp("score"(pi))$
  $Z=sum_(y' in Omega^*)sum_(pi')exp("score"(pi'))$ (infinite!)
]

#cbox(title: [Matrix Mult View])[
  $C=A times.o B$: $C_(i j)=plus.o.big_k (A_(i k) times.o B_(k j))$
  *Tropical*: $C_(i j)=min_k (A_(i k)+B_(k j))$
  *Inside*: $C_(i j)=sum_k (A_(i k) times B_(k j))$
  Naive $W^N$: $O(N^4)$ → Lehmann fixes to $O(N^3)$
]

#cbox(title: [Lehmann递推直觉])[
  $bold(R)_(i k)^((j))$: 从$q_i$到$q_k$, 仅经过${q_1,...,q_j}$的paths总权
  *分解*:
  $ bold(R)_(i k)^((j)) = bold(R)_(i k)^((j-1)) plus.o (bold(R)_(i j)^((j-1)) times.o (bold(R)_(j j)^((j-1)))^* times.o bold(R)_(j k)^((j-1))) $
  不经$q_j$ + (到$q_j$ + 在$q_j$循环任意次 + 离开$q_j$)
  
  $ Z = plus.o.big_(i,k in Q) lambda(q_i) times.o bold(R)_(i k) times.o rho(q_k)$
  
  $lambda$: initial weights
  $rho$: final weights
  $bold(R)_(i k)$: Lehmann算出的all-paths权重
]

#cbox(title: [Floyd-Warshall])[
  *Key*: allow中间node $k$ incrementally
  $ "dist"_k [i][j]=min("dist"_(k-1)[i][j], "dist"_(k-1)[i][k]+"dist"_(k-1)[k][j]) $
  Runtime: $O(N^3)$
  *FW*是Lehmann在Tropical的特例（$a^*=0$ 循环不帮助）
]

#cbox(title: [Lehmann algo])[
  *Generalized FW* for any closed semiring:
  $ W^((k))_(i j) = W^((k-1))_(i j) plus.o W^((k-1))_(i k) times.o (W^((k-1))_(k k))^* times.o W^((k-1))_(k j) $
  *定义*: $bold(R)_(i k)^((j))$=从$q_i$到$q_k$, 仅经过${q_1,...,q_j}$的paths的semiring-sum
  *直觉*: 经过${1,...,j}$的paths = 不经$j$ $plus.o$ 经$j$
  后者分解: $i->j$(不经$j$) + $j$上cycles + $j->k$(不经$j$)
  Runtime: $O(|Q|^3)$
]

#cbox(title: [Pathsum & Z])[
  $Z(cal(T))=plus.o.big_(i,k in Q) lambda(q_i) times.o bold(R)_(i k) times.o rho(q_k)$
  $Z=alpha^top (plus.o.big_(omega in Sigma^*) W^((omega)))^* beta$
  *Why Lehmann?* Direct sum over infinite paths impossible
]

#cbox(title: [Composition])[
  $cal(T)(x,y)=plus.o.big_(z in Omega^*) cal(T)_1(x,z) times.o cal(T)_2(z,y)$
  *Transliteration*: 3 transducers cascade
  $cal(T)_x compose cal(T)_theta compose cal(T)_y$
]

#cbox(title: [Acyclic WFSA Backward])[
  *前提*: DAG可做topological sort
  1. 按reverse topo order遍历nodes $q_M,...,q_1$
  2. $beta[q_m]<-rho(q_m) plus.o plus.o.big_((q_m,a,w,q') in delta) w times.o beta[q']$
  3. return $plus.o.big_(q in I) lambda(q) times.o beta[q]$
  *Complexity*: $O(|Q|+|delta|)$ (linear!)
]

= 10. Transformers & MT

#cbox(title: [Seq2Seq])[
  $z="encoder"(x)$, $y|x tilde"decoder"(z)$
  $p(y|x)=product_(t=1)^T p(y_t|x,y_1,...,y_(t-1))$
  *Information Bottleneck*: $z$ fixed-length → Attention解决
]

#cbox(title: [Attention])[
  $alpha^T V=sum_i alpha_i v_i^T$ (soft retrieval)
  $alpha_i="sftm"("score"(q,k_i))$
  $K=V=H^((e))$, $q_t=h_t^((d))$, $c=alpha^T V$
]

#cbox(title: [Self-Attention])[
  $bold(Q)=bold(X)bold(W)_Q$, $bold(K)=bold(X)bold(W)_K$, $bold(V)=bold(X)bold(W)_V$
  $"SelfAtt"="sftm"((bold(Q)bold(K)^top)/sqrt(d_k))bold(V)$
  *$sqrt(d_k)$*: 防止点积过大导致softmax饱和; $sigma^2$.
  *Complexity*: $O(n d^2+d n^2)$

  Permutation Equivariance:  若 $f$ 是 permutation equivariant，则对任意 permutation $pi$, $f(pi(X)) = pi(f(X))$, 若$bold(Q)$ fixed（如常数矩阵），则attention permutation invariant.即打乱输入顺序，输出以相同方式打乱. 
  设 $bold(P)$ 是 permutation matrix, 则:
  $"Attn"(bold(P X)) = "sftm"( 1/ sqrt(d) (bold(P X) bold(W)_Q)(bold(P X) bold(W)_K)^top) (bold(P X) bold(W)_V) 
    = "sftm"(bold(P) bold(Q) bold(K)^top bold(P)^top/ sqrt(d)) bold(P) bold(V) = bold(P) "sftm"(bold(Q) bold(K)^top / sqrt(d)) bold(V) = bold(P) "Attn"(bold(X))
  $
  
]

#cbox(title: [Positional Encoding])[
  $bold(P)_(p,2i)=sin(p\/10000^(2i\/d))$
  $bold(P)_(p,2i+1)=cos(p\/10000^(2i\/d))$
  motiv: Transformer无recurrence, 无法区分位置
]

#cbox(title: [Encoder-Decoder架构])[
  *Encoder*: $+bold(P) -> "MHSA" -> + -> "LN" -> "MLP" -> + -> "LN"$
  *Decoder*: +masked self-attn + cross-attn
  *Masked*: 只attend到左边positions (causal)
  *Cross-attn*: $Q$来自decoder, $K,V$来自encoder
  *Residual*: $x+"Layer"(x)$ 缓解vanishing gradient
]

#cbox(title: [Decoding Strategies])[
  $y^*=arg max_(y in cal(Y))"score"(x,y)$
  W/o assumptions: $O(|Sigma|^(n_max))$ paths
  *Greedy*: 每步$arg max$ (次优, 快)
  *Beam*: 保持$k$-best candidates
  *Nucleus/Top-p*: 从累积prob$>=p$的tokens中sample
  *Temperature*: $T<1$ sharper; $T>1$ uniform
  *Eval*: BLEU (n-gram overlap), METEOR
]

#cbox(title: [MT Pipeline])[
  1. *Tokenize*: subword (BPE/WordPiece)
  2. *Embed*: token→vector + positional
  3. *Encode*: Transformer encoder
  4. *Decode*: autoregressive, $p(y_n|y_(<n),bold(z))$
  5. *Search*: beam/nucleus sampling
  *Train*: MLE, $-sum log p(y_n|y_(<n),bold(x))$
]

= 11. Modeling Choices

#cbox(title: [Prob vs Non-Prob])[
  *Prob*: leverage prob theory, needs assumptions
  CRF, RNN, N-gram models
  *Non-Prob*: interpretable, uncertainty unclear
  Perceptron, SVM, CFG rules
]

#cbox(title: [Disc vs Generative])[
  *Discriminative*: model boundary $p(y|x)$
  *Generative*: model own dist $p(x,y)$
]

#cbox(title: [Local vs Global Norm])[
  *Local*: efficient train, biased predictions
  *Global*: needs $Z$, unbiased
  Independence assumptions control complexity
]

#cbox(title: [Regularization])[
  *LogLoss*: $ell(y,y')=log(1+e^(-y dot y'))$
  *Exp-Loss*: $ell(y,y')=e^(-y dot y')$
  *L1/L2*: weight penalties (Laplace/Gaussian prior)
]

#cbox(title: [Evaluation Metrics])[
  *Prec*: $P_"true"\/P_"all"$; *Recall*: $P_"true"\/(P_"true"+N_"false")$
  *Acc*: $(P_"true"+N_"true")\/N$
  *F-score*: $((1+beta^2)("prec" dot "recall"))\/(beta^2"prec"+"recall")$
]

#cbox(title: [Statistical Tests])[
  $p=2 min(P(T>=t|H_0),P(T<=t|H_0))$; Rej if $p<alpha$
  *Power*: $P("reject" H_0|H_1)$
  *Multiple tests*: $P(|"FalseRej"|>0)=1-(1-alpha)^K$
  *Bonferroni*: $alpha^*=alpha\/K$
  *McNemar*: $chi^2=((b-c)^2)/(b+c) tilde chi_1^2$
]

= 12. Bias & Fairness &Eval

#cbox(title: [Bias Sources])[
  *Labeling*: reproduce annotator bias
  *Sample selection*: training fits certain profile
  *Task definition*: excludes certain groups
  *Imbalanced test*: loss ignores minorities
]


#cbox(title: [BLEU Score])[
  $"BLEU"="BP" times exp(sum_(n=1)^N w_n log p_n)$
  $p_n$: n-gram precision (clipped count)
  $"BP"=cases(1 & c>r, e^(1-r\/c) & "otherwise")$
  $c$=候选长度, $r$=参考长度, $w_n=1\/N$
  *Clipped*: $"count"_"clip"=min("count"_"pred", max_"ref" "count"_"ref")$
  防止重复词刷分
]

#cbox(title: [Model Taxonomy])[
  *Probabilistic*: 建模$p(Y|X)$或$p(X,Y)$
  - *Discriminative*: 直接$p(Y|X)$ (LogReg, CRF)
  - *Generative*: joint $p(X,Y)=p(Y)p(X|Y)$ (N-gram, HMM)
  *Non-Prob*: Learned (SVM, MLP) / Handcrafted (CFG)
]

#cbox(title: [Confusion Matrix Metrics])[
  #set text(size: 7pt)
  #table(
    columns: 3, inset: 2pt,
    [], [Pred +], [Pred −],
    [Actual +], [TP], [FN],
    [Actual −], [FP], [TN],
  )
  $"Prec"="TP"\/("TP"+"FP")$; $"Recall"="TP"\/("TP"+"FN")$
  $"F"_1=2 dot ("Prec" dot "Recall")\/("Prec"+"Recall")$
  *为何不用Acc?* Class imbalance; 不同错误代价不同
]

#cbox(title: [K-Fold CV])[
  数据分$K$份, 每次取第$k$份为test, 其余train
  *Test set size*: $N\/K$
  *Train set size*: $N times (K-1)\/K$
  *Total models*: $K$
  *Nested CV*: Inner loop调参, Outer loop评估
]

#cbox(title: [McNemar's Test])[
  比较两个classifiers在同一数据集上表现
  #set text(size: 7pt)
  #table(
    columns: 3, inset: 2pt,
    [], [B Correct], [B Wrong],
    [A Correct], [$n_(00)$], [$n_(01)$],
    [A Wrong], [$n_(10)$], [$n_(11)$],
  )
  $chi^2=((|n_(01)-n_(10)|-1)^2)/(n_(01)+n_(10))$
  只关注disagreement cells $n_(01),n_(10)$
  要求$n_(01)+n_(10)>=25$
]

#cbox(title: [Permutation Test])[
  1. 原始数据训练, 记录performance $P_0$
  2. Repeat $B>=1000$次: permute labels, 重训, 记录$P_b$
  3. p-value $approx$ fraction of $P_b>=P_0$
  *tip*: 若labels有信息, 原始模型应显著优于permuted
]

= 补充
#cbox(title: [Edit Distance FSA ])[
  *状态*: $(i,e)$ 位置×编辑次数, 共$O(d N)$个
  *转移*:
  匹配$s_i$→$(i+1,e)$; 
  插Σ→$(i,e+1)$;
  删ε→$(i+1,e+1)$; 
  替Σ$backslash s_i$→$(i+1,e+1)$
  *终态*: $i=N$所有状态
  *口诀*: 插读不动, 删ε跳, 替错跳
]

#cbox(title: [Semiring速判 ])[
  *先验$bold(0) plus.o a = a$*
  $min$单位元=$+infinity$ (非$0$/$-infinity$)
  $max$单位元=$-infinity$
  
  *Kleene*: Real $a^*=1/(1-a)$; Bool $a^*=1$
]

#cbox(title: [BPTT ])[
  $(partial h_t)/(partial h_k)=product_i "diag"(sigma')R$
  $R^n=Q D^n Q^(-1)$
  $|lambda|<1$→消失; $|lambda|>1$→爆炸
]

#cbox(title: [FOL ])[
  $forall$配$=>$; $exists$配$and$
  "所有X都Y": $forall x.X(x)=>Y(x)$
  "有些X是Y": $exists x.X(x) and Y(x)$
]

#cbox(title: [$beta$-reduce ])[
  $(lambda x.M)N -> M[x:=N]$
  $(lambda x.(x x))(lambda z.x)=(lambda z.x)(lambda z.x)=x$
]

#cbox(title: [Fwd vs Bwd不对称性])[
  #set text(size: 7pt)
  #table(
    columns: 3,
    [*项目*], [*Forward*], [*Backward*],
    [初始化], [$alpha[0,t]=exp("score"("BOS"->t))$], [$beta[N,t]=bold(1)$ 全1],
    [递推], [$alpha[n,t]=plus.o.big_(t') alpha[n-1,t'] times.o exp$], [$beta[n,t]=plus.o.big_(t') exp times.o beta[n+1,t']$],
    [终止], [$plus.o.big_t alpha[N,t]$ 需sum], [$beta[0,"BOS"]$ 单值],
  )
  *原因*: BOS显式存在, EOS隐式处理使用场景
  *Forward*: 单独计算$Z$ (partition function)
  *Backward*: 单独计算suffix概率
  *两者结合*: 计算marginals $p(t_n=t|bold(w))$
    $p(t_n=t|bold(w)) = (alpha[n,t] times beta[n,t]) / Z$
]


#block(stroke: 0pt, inset: 3pt, width: 100%)[
  #set text(size: 8pt)
  = Quick Ref
  *Chain*: $(d)/(d x)[f(g(x))]=f'(g)g'(x)$; Bauer: sum over all paths
  *Softmax*: $exp(h_y)\/sum exp(h_(y'))$; $T->0$=argmax
  *Log-Linear*: $p(y|x)=exp(theta dot f)\/Z$; MLE matches expected features
  *DP*: distrib把$O(|T|^N)->O(N|T|^2)$; 3-gram则$O(N|T|^3)$
  *Fwd/Bwd*: Fwd init BOS+sum last col; Bwd init $bold(1)$+single value
  *Viterbi*: max instead of sum + backpointer
  *CKY*: $O(N^3|R|)$; CNF; diag first, span递增
  *MTT*: $Z=det(L)$ in $O(n^3)$; *CLE*: greedy+contract $O(n^2)$
  *Lehmann*: $R^((j))=R^((j-1)) plus.o R times.o R^* times.o R$; $O(|Q|^3)$
  *Kleene*: Inside $1/(1-a)$; Tropical $0$ if $a>=0$
  *CCG*: $X\/Y space Y=>X$ (>); $Y space X backslash Y=>X$ (<)
  *$beta$-reduce*: $(lambda x.M)N->M[x:=N]$; 先$alpha$-convert避免捕获
  *Self-Attn*: $"sftm"(Q K^T\/sqrt(d))V$; $O(n d^2+d n^2)$
  *Cayley*: 固定root $n^(n-2)$; 任意root $n^(n-1)$
]
