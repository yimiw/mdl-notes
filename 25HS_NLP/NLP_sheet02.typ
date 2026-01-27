#import "../assets/tmp_sht.typ": *
#show: project.with(authors: ((name: "", email: ""),))

// ========== Font Size ==========
#let fsize = 8.5pt
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

= 1. Backpropagation

#cbox(title: [Chain Rule])[
  $d/(d x)[f(g(x))]=f'(g(x))g'(x)$
  *Jacobian*: $f: RR^n -> RR^m$, $(d y)/(d x)=[(d y)/(d x_1),...,(d y)/(d x_n)] in RR^(m times n)$
  *Multivar*: $(d y_i)/(d x_j)=sum_(k=1)^m (d y_i)/(d z_k)(d z_k)/(d x_j)$
]

#cbox(title: [Bauer Path Formula])[
  $(d y_i)/(d x_j)=sum_(p in cal(P)(j,i)) product_((k,l) in p) (d z_l)/(d z_k)$
  $cal(P)(j,i)$=all paths $j->i$; worst $O(m^n)$
  *Computation Graph*: DAG w/ function nodes, edges=variable flow
]

#cbox(title: [Forward vs Reverse])[
  *Forward*: expand $(d)/(d x)$ recursively, same flow as fwd
  *Reverse*: 2 passes—fwd compute vals, bwd compute grads
  *Complexity*: same time as $f$; higher space (store intermediates)
]

#cbox(title: [Primitives])[
  *Sum*: $(d(a+b))/(d a)=1$; *Prod*: $(d(a b))/(d a)=b$
  // $log 1=0$, $log 0=-infinity$, $exp 0=1$, $sin 0=0$, $cos 0=1$
]

= 2. Log-Linear Models

#cbox(title: [Prob Basics])[
  *Bayes*: $p(y|x)=(p(x|y)p(y))/(integral p(x|y)p(y) d y)$
  *Marginal*: $p(x)=sum_y p(x,y)$
  *Expectation*: $EE[f(x)]=sum_x f(x)p(x)$
]

#cbox(title: [Log-Linear Model])[
  $p(y|x,theta)=(exp(theta dot f(x,y)))/(Z(theta))$, $Z(theta)=sum_(y' in Y)exp(theta dot f(x,y'))$
  $log p(y|x,theta)=theta dot f(x,y)-log Z$ (linear in log space!)
  *Discrete MLE*: $p(y|x)="count"(x,y)/"count"(x)$ (sparse问题)
]

#cbox(title: [MLE $nabla$])[
  $theta_"MLE"=arg min_theta -sum_(n=1)^N log p(y_n|x_n,theta)$
  $(d cal(L))/(d theta_k)=-sum_n f_k (x_n,y_n)+sum_n sum_(y')p(y'|x_n;theta)f_k (x_n,y')$
  观测特征计数 = 期望特征计数 → *Expectation Matching*
]

#cbox(title: [Softmax])[
  $"softmax"(h,y,T)=(exp(h_y\/T))/(sum_(y')exp(h_(y')\/T))$
  $T->0$: argmax; $T->infinity$: uniform
  $log "softmax"=h_y-log sum_(y')exp(h_(y'))$ (logsumexp)
  $(d log "softmax")/(d theta)=f(x,y)-sum_i "softmax"(theta^top f,i)f(x,i)$
]


#cbox(title: [MLP Architecture])[
  *Problem*: Log-linear needs linearly separable data
  *Solution*: Learn non-linear feature fn
  $bold(h)_k=sigma_k (bold(W)_k^top bold(h)_(k-1))$, $bold(h)_1=sigma_1 (bold(W)_1^top bold(e)(x))$
  Output: $"softmax"(bold(theta)^top bold(h)_n)$
]

#cbox(title: [Sigmoid & Activations])[
  $sigma(x)=1/(1+exp(-x))$, $nabla sigma=sigma(1-sigma)$
  *tanh*: $(1-e^(-2x))/(1+e^(-2x))$, $nabla=1-tanh^2$,
  Sigmoid/tanh, vanishing gradient→ use ReLU
  *Backprop(MLP)*: $(partial ell)/(partial bold(W)_k)=(partial ell)/(partial y)(partial y)/(partial bold(h)_n)(product_(m=k+1)^n sigma'_m bold(W)_m)sigma'_k bold(h)_(k-1)$
]

#cbox(title: [Learning Pipeline])[
  Embedding → Pooling (sum/mean/max) → NN → Softmax
]

#cbox(title: [Exp Family & MaxEnt])[
  $p(x|theta)=1/(Z(theta))h(x)exp(theta dot phi(x))$
  *Max Entropy*: $H(p)=-sum_x p(x)log p(x)$
  选最大熵分布=最少假设=Laplace原则
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
  Problem: $O(|cal(T)|^N)$ paths in normalizer
]

#cbox(title: [CRF Model])[
  $p(t|w)=(exp("score"(t,w)))/(sum_(t' in cal(T)^N)exp("score"(t',w)))$

  *Decomposition*: $"score"(t,w)=sum_(n=1)^N "score"(chevron.l t_(n-1),t_n chevron.r,w,n)$
  $p(t|w) prop product_(n=1)^N exp{"score"(chevron.l t_(n-1),t_n chevron.r,w)}$
]

#cbox(title: [Forward-Backward DP])[
  $forall t_n: beta(w,t_N,N)<-1$
  for $n<-N-1,...,0$:
  $beta(w,t_n,n)<-sum_(t_(n+1) in cal(T))exp("score") times beta(w,t_(n+1))$
]

#cbox(title: [Viterbi Decoding])[
  $beta(w,t_n)<-max_(t_(n+1))exp("score") times beta(w,t_(n+1))$
  *Structured CRF*: $log p=sum_i ("score"(t^((i)),w^((i)))-max_(t')"score"(t',w^((i))))$
]
#cbox(title: [Semiring Definition])[
  $angle.l bb(K),plus.o,times.o,bold(0),bold(1) angle.r$ where:
  1. $(bb(K),plus.o,bold(0))$: *comm monoid* (assoc+comm+identity)
  2. $(bb(K),times.o,bold(1))$: *monoid* (assoc+identity)
  3. *Distrib*: $(x plus.o y) times.o z=(x times.o z) plus.o (y times.o z)$
  4. *Annihilator*: $bold(0) times.o x=x times.o bold(0)=bold(0)$
]

#cbox(title: [Semiring意义])[
  $plus.o$: *分治* (split points合并, OR/MAX/+)
  $times.o$: *连接* (左右子树组合, AND/$times$/+)
  $bold(0)$: 吸收元, 消除invalid; $bold(1)$: 单位元, null不破坏
]

#cbox(title: [Monoid判定])[
  1. *Closure*: $a times.o b in bb(K)$; 2. *Assoc*: $(a times.o b) times.o c=a times.o (b times.o c)$; 3. *Identity*: $exists bold(e): a times.o bold(e)=bold(e) times.o a=a$
  // *陷阱*: 减法不assoc; identity须在集合内
  // $angle.l NN,+,0 angle.r$✓; $angle.l ZZ,-,0 angle.r$✗(不assoc)
  // $angle.l Sigma^*,"concat",epsilon angle.r$✓(非交换!)
]

#cbox(title: [Semiring判定])[
  1. $plus.o$-monoid (comm): $a plus.o b=b plus.o a$ ; 2. $times.o$-monoid; 3. Distributivity (左右皆需); 4. Annihilation: $bold(0) times.o x=bold(0)$
  *陷阱*: $bold(0)=bold(1)$必失败!
  // $angle.l RR_(>=0),max,+,0,0 angle.r$✗ ($bold(0)=bold(1)$矛盾)
]

#cbox(title: [Closed Semiring & Kleene\*])[
  $a^*=plus.o.big_(n=0)^infinity a^(times.o n)=bold(1) plus.o a times.o a^*$
  Real上$|a|<1$: $a^*=1/(1-a)$ (geometric series)
  用于globally normalized LM
]

#cbox(title: [DP推导])[
  Goal: $Z=sum_(bold(t))exp"score"(bold(t),bold(w))$
  *Step1*: 可加分解 $"score"=sum_n "score"_n$
  *Step2*: $Z=sum_(bold(t))product_n exp"score"_n$
  $=sum_(t_1)exp"score"_1 times (sum_(t_2)...)$ (distrib!)
  $O(|cal(T)|^N) -> O(N|cal(T)|^2)$
  *若依赖3-gram*: $O(N|cal(T)|^3)$
]

#cbox(title: [Common Semirings])[
  #set text(size: 6pt)
  #table(
    columns: 7,
    // 表头
    [*Name*],
    [$bb(K)$],
    [$plus.o$],
    [$times.o$],
    [$bold(0)$],
    [$bold(1)$],
    [*用途*],


    // 数据行
    [Real],
    [$RR_(>=0)$],
    [$+$],
    [$times$],
    [$0$],
    [$1$],
    [$Z$ partition fn],

    [Viterbi],
    [$RR union {-infinity}$],
    [$max$],
    [$+$],
    [$-infinity$],
    [$0$],
    [最优path/解码],

    [Log],
    [$RR union {pm infinity}$],
    [lse],
    [$+$],
    [$-infinity$],
    [$0$],
    [$log Z$ 数值稳定],

    [Boolean],
    [${0,1}$],
    [$or$],
    [$and$],
    [$0$],
    [$1$],
    [可达性/存在性],

    [Counting],
    [$NN$],
    [$+$],
    [$times$],
    [$0$],
    [$1$],
    [路径数/歧义度],

    [Tropical],
    [$RR union {infinity}$],
    [$min$],
    [$+$],
    [$infinity$],
    [$0$],
    [最短路/编辑距离],
  )
]
// #cbox(title: [Semirings])[
//   $(plus.o,overline(0))$: comm. monoid; $(times.o,overline(1))$: monoid
//   $times.o$ dist over $plus.o$; $overline(0) times.o a=overline(0)$
//   *Boolean*: $chevron.l{0,1},"or","and",0,1 chevron.r$ (logical)
//   *Viterbi*: $chevron.l[0,1],max,times,0,1 chevron.r$ (best deriv)
//   *Inside*: $chevron.l RR^+,+,times,0,1 chevron.r$ (prob of string)
//   *Tropical*: $chevron.l RR^+,min,+,infinity,0 chevron.r$ (shortest dist)
//   *Counting*: $chevron.l NN,+,times,0,1 chevron.r$ (\#paths)
// ]

= 6. CFG Parsing

#cbox(title: [Constituents])[
  Multi-word units as single unit
  *Tests*: Pronoun substitution, Clefting, Answer ellipsis
  Ambiguity: PP attachment, modifier scope
]

#cbox(title: [CFG Definition])[
  $G=chevron.l cal(N),cal(S),Sigma,cal(R) chevron.r$
  Non-terminals, start symbol, terminals, production rules
  *CNF*: $N_1->N_2 N_3$ or $N->a$; $O(4^N)$ trees
]

#cbox(title: [Weighted CFG])[
  *Global*: $p(t)=1/Z product_(r in t)exp("score"(r))$
  $Z=sum_(t' in cal(T))product_(r')exp("score"(r'))$ (可能$infinity$!)
  *Probabilistic*: local norm $sum_k p(alpha_k|N)=1$
]

#cbox(title: [CKY Algorithm])[
  $O(N^3|R|)$, needs CNF
  for $n=1,...,N$: for $X->s_n in cal(R)$:
  $"chart"[n,n+1,X] bold(plus.o) exp("score"(X->s_n))$
  for span$=2,...,N$: for $i=1,...,N-"span"$:
  $k<-i+"span"$; for $j=i+1,...,k-1$:
  for $X->Y Z in cal(R)$:
  $
  "chart"[i,k,X] bold(plus.o) exp{"score"} times.o "chart"[i,j,Y] times.o "chart"[j,k,Z]
  $
  *Best parse*: semiring $(max,+)$
]

= 7. Dependency Parsing

#cbox(title: [Dependency Tree])[
  Directed spanning tree, root degree 1
  *Projective*: no crossing arcs (≈constituency w/ heads)
  *Non-projective*: crossing arcs (≈discontinuous constituents)
  \# spanning trees: $O((n-1)^(n-2))$
]

#cbox(title: [Edge-Factored Model])[
  $p(t|w)=1/Z product_((i->j) in t)exp("score"(i,j,w))exp("score"(r,w))$
  *Edge factor assumption*: score factors over edges
]

#cbox(title: [Matrix-Tree Theorem])[
  $A_(i j)=exp("score"(i,j,w))$, $rho_j=exp("score"(j,w))$
  $Z=det(L)$ where $L_(i j)=cases(rho_j & i=1, sum_(i' != j)A_(i' j) & i=j, -A_(i j) & "else")$
  Computing $det$ in $O(n^3)$
]

#cbox(title: [MST Decoding])[
  $arg max_(t in cal(T))sum_((i->j) in t)"score"(i,j,w)$
  *Algo*: max incoming edge, contract cycles (update weights)
  *Root Constraint*: for each root edge, compute removal cost; remove cheapest
  Runtime: $O(n^2)$
]

= 8. Semantic Parsing

#cbox(title: [Syntax vs Semantics])[
  *Syntax*: structural org (parse tree)
  *Semantics*: underlying meaning
  *Logical form*: quantifiers, vars, boolean, predicates
]

#cbox(title: [Lambda Calculus])[
  *Abstraction*: $M$ term, $x$ var → $lambda x.M$ term
  *Application*: $M,N$ terms → $M N$ term
  *$beta$-reduction*: $(lambda x.M)N=M[x:=N]$
  *$beta$-infinity*: $F=lambda x((x x)x)$, $F F=((F F)F)=...$
]

#cbox(title: [Composition])[
  $S_"VP"->$NP VP, $S."sem"="VP"."sem"("NP"."sem")$
  *Compositionality*: meaning of whole = fn of parts
]

#cbox(title: [Combinatory Logic])[
  I: $I x=x$; K: $K x y=x$; S: $S x y z=(x z)(y z)$
  S-K calculus ≡ lambda calculus (via translator T)
  Don't need I: $(S K K)x=x$
]

= 9. WFSTs

#cbox(title: [Transducer])[
  $T=chevron.l cal(Q),Sigma,Omega,lambda,rho,delta chevron.r$
  States, input vocab, output vocab, initial/final scores, transitions
  Goal: $p(y|x)$, $x in Sigma^*$, $y in Omega^*$
]

#cbox(title: [Scoring])[
  $"score"(pi)=lambda(q_"start")+sum_n delta(q_n)+rho(q_"end")$
  $p(y|x)=1/Z sum_(pi in Pi(x,y))exp("score"(pi))$
  $Z=sum_(y' in Omega^*)sum_(pi')exp("score")$ (infinite—loops!)
]

#cbox(title: [Floyd-Warshall + Semiring])[
  $forall i,j,k: "dist"[i][j]<-"dist"[i][j] plus.o ("dist"[i][k] times.o "dist"[k][j])$
  *Matrix mult*: sum$<-overline(0)$; sum$<-$sum$plus.o(A[n][m] times.o B[m][p])$
]

#cbox(title: [Kleene Star])[
  $ W^*=plus.o_(k=0)^infinity W^k=I+W W^* arrow.l.r.double W^*=(I-W)^(-1) $
  *Warshall-Floyd-Kleene*:
  $ "dist"[i][j]<-"dist"[i][j] plus.o ("dist"[i][k] times.o "dist"[k][k]^* times.o "dist"[k][j]) $
]

= 10. Transformers & MT

#cbox(title: [Seq2Seq])[
  $z="encoder"(x)$, $y|x tilde"decoder"(z)$
  $p(y|x)=product_(t=1)^T p(y_t|x,y_1,...,y_(t-1))$
  Optimize log-likelihood
]

#cbox(title: [Attention])[
  $alpha^T V=sum_i alpha_i v_i^T$ (soft retrieval)
  $alpha_i="softmax"("score"(q,k_i))$
  $K=V=H^((e))$, $q_t=h_t^((d))$, $c=alpha^T V$
]

#cbox(title: [Transformer Components])[
  *Word Embed*: token→vector
  *Positional Embed*: encode word position (no recurrence!)
  *Residual Connections*: mitigate vanishing $nabla$s
  *Layer Norm*: normalize layer inputs
  *Self-attention*: $Q,K,V$ from same sequence
]

#cbox(title: [Decoding])[
  $y^*=arg max_(y in cal(Y))"score"(x,y)$
  W/o assumptions: $O(|Sigma|^(n_max))$ paths
  *Beam Search*: keep $k$ best at each step (greedy approx)
  *Sampling*: sample from $p(y|x)$ each step
  *Eval*: BLEU (n-gram overlap), METEOR
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

#cbox(title: [Loss & Regularization])[
  *LogLoss*: $ell(y,y')=log(1+e^(-y dot y'))$
  *Exp-Loss*: $ell(y,y')=e^(-y dot y')$
  *L1/L2*: weight penalties (Laplace/Gaussian prior)
]

#cbox(title: [Evaluation Metrics])[
  *Prec*: $P_"true"\/P_"all"$; *Recall*: $P_"true"\/(P_"true"+N_"false")$
  *Acc*: $(P_"true"+N_"true")\/N$
  *F-score*: $((1+beta^2)("prec" dot "recall"))/(beta^2"prec"+"recall")$
]

#cbox(title: [Statistical Tests])[
  $p=2 min(P(T>=t|H_0),P(T<=t|H_0))$; Rej if $p<alpha$
  *Power*: $P("reject" H_0|H_1)$
  *Multiple tests*: $P(|"FalseRej"|>0)=1-(1-alpha)^K$
  *Bonferroni*: $alpha^*=alpha\/K$
]

#cbox(title: [Permutation Test])[
  $p^*$ from original data; permute labels $k$ times
  p-value$=(|{i:p_i<=p^*}|+1)\/(k+1)$
]

#cbox(title: [McNemar's Test])[
  $chi^2=((b-c)^2)/(b+c) tilde chi_1^2$ for $b,c>=25$
  $H_0:p_b=p_c$; $b$=C1 wrong/C2 right
]

#cbox(title: [5×2cv Test])[
  $overline(p)=(p^((1))+p^((2)))\/2$
  $s^2=(p^((1))-overline(p))^2+(p^((2))-overline(p))^2$
  $t=(p_1^((1)))\/sqrt(1\/5 sum s_i^2) tilde t^5$
]

= 12. Bias & Fairness

#cbox(title: [Bias Sources])[
  *Labeling*: reproduce annotator bias
  *Sample selection*: training fits certain profile
  *Task definition*: excludes certain groups
  *Imbalanced test*: loss ignores minorities
]

#cbox(title: [Ethical Frameworks])[
  *Consequentialism*: best consequence
  *Utilitarism*: hedonistic/preference/welfare
  *Deontology*: rules must be kept
  *Social Contract*: natural equality
  *Anti-subordination*: positive discrimination for equality
]

#block(stroke: 0pt, inset: 3pt, width: 100%)[
  #set text(size: 7pt)
  = Quick Ref
  *Chain*: $(d)/(d x)[f(g(x))]=f'(g)g'(x)$; Bauer: sum over all paths
  *Softmax*: $exp(h_y)\/sum exp(h_(y'))$; $T->0$=argmax; $T->infinity$=uniform
  *Log-Linear*: $p(y|x)=exp(theta dot f)\/Z$; MLE matches expected features
  *CRF*: decompose score → DP; semiring unifies algos
  *Viterbi*: max instead of sum; decoding=$arg max "score"$
  *CKY*: $O(N^3|R|)$; needs CNF; semiring for best/count/prob
  *Dep Parse*: Matrix-Tree for $Z$ in $O(n^3)$; MST+contract cycles
  *Attention*: $alpha="softmax"(Q K^T)$; $c=alpha V$ (soft lookup)
  *WFST*: $"Kleene"^*$ for infinite sums; $(I-W)^(-1)$
  *Lambda*: $beta$-reduction $(lambda x.M)N=M[x:=N]$
  *Local vs Global*: bias vs intractability tradeoff
  *Semirings*: Boolean/Viterbi/Inside/Tropical/Counting
  *Stats*: Bonferroni $alpha\/K$; McNemar $(b-c)^2\/(b+c)$
]

#block(stroke: 0pt, inset: 3pt, width: 100%)[
  #set text(size: 6.5pt)
  = Abbrev
  *BOS/EOS*: Begin/End of Sentence; *CCG*: Combinatory Categorial Grammar; *CFG*: Context-Free Grammar; *CKY*: Cocke-Kasami-Younger; *CNF*: Chomsky Normal Form; *CRF*: Conditional Random Field; *DP*: Dynamic Programming; *LLM*: Log-Linear Model; *MLE*: Max Likelihood Est; *MST*: Min Spanning Tree; *NLP*: Natural Language Processing; *POS*: Part-of-Speech; *RNN*: Recurrent Neural Network; *WFST*: Weighted Finite State Transducer;
]