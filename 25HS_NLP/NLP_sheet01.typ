#import "../assets/tmp_sht.typ": *

#show: project.with(
  title: "25HS_NLP_Cheatsheet",
  authors: ((name: ""),),
)

// ========== Configuration ==========
#let fsize = 8pt
#let hsize = 9pt
#let pspace = 0.15em
#let plead = 0.25em

#set text(size: fsize)
#set par(spacing: pspace, leading: plead, justify: true, first-line-indent: 0em)
#show heading.where(level: 1): set text(size: hsize, weight: "bold")
#show heading: box
#show heading: set text(fill: rgb("#663399"))

#show: columns.with(3, gutter: 0.4em)

// ========== Content ==========

= Backpropagation

#cbox(title: [Linear-time DP for derivatives])[
  1. Write composite fn as labeled acyclic hypergraph
  2. Forward propagation with input
  3. Backprop: $(partial y_i)/(partial x_j) = sum_(p in P(j,i)) product_((k arrow ell) in p) (partial z_ell)/(partial z_k)$

  $sin'(x) = cos(x)$, $cos'(x) = -sin(x)$, $log'(x) = 1/x$, $exp'(x) = exp(x)$
]

= Log-linear Modelling

#cbox[
  $"score"(y,x) = bold(theta)^top bold(f)(x,y)$

  *NLL gradient = 0*: $sum_(i=1)^n bold(f)(x_i,y_i) = sum_(i=1)^n EE_(y|x_i,bold(theta))[bold(f)(x_i,y)]$

  *Hessian*: $bold(H)_bold(theta)(sum_i -log p(y_i|x_i)) = sum_i "Cov"_(y|x_i,bold(theta))[bold(f)(x_i,y)]$

  *Softmax*: $"softmax"(bold(h))_y = exp(h_y\/T) / (sum_(y') exp(h_(y')\/T))$
  $T arrow 0$: argmax. $T arrow infinity$: uniform.

  *Exponential family*: $p(x|bold(theta)) = 1/(Z(bold(theta))) h(x) exp(bold(theta)^top bold(phi)(x))$
]

= Multi-layer Perceptron

#cbox[
  *Problem*: Data must be linearly separable.
  *Solution*: Learn non-linear feature fn with MLP:

  $bold(h)_k = sigma_k (bold(W)_k^top bold(h)_(k-1))$, $bold(h)_1 = sigma_1 (bold(W)_1^top bold(e)(x))$

  Then $"softmax"(bold(theta)^top bold(h)_n)$ for prob dist.

  *Skip-Gram*: predict if 2 words in same context. Need good word repr.

  *Derivative*: $(partial ell)/(partial bold(W)_k) = (partial ell)/(partial y) (partial y)/(partial bold(h)_n) (product_(m=k+1)^n sigma'_m (...) bold(W)_m) sigma'_k (...) bold(h)_(k-1)$
]

= Structured Prediction

#cbox[
  $p(y|x) = exp("score"(y,x)) / Z(x)$, $Z(x) = sum_(y' in cal(Y)) exp("score"(y',x))$

  *Problem*: $cal(Y)$ exponentially/infinitely large.
  *Solution*: Design algorithms using structure of input/output.
]

= Language Modelling

#cbox[
  $p(bold(y)) = p("eos"|bold(y)) dot product_(i=1)^N p(y_i | bold(y)_(<i))$
  $p(y_i|bold(y)_(<i)) = 1/(Z(bold(y)_(<i))) exp("score"(bold(y)_(<i), y_i))$

  *Non-tight*: Force $p("eos"|bold(y)_(<i)) > xi > 0$

  *$n$-gram*: $p(y_i|bold(y)_(<i)) = p(y_i|y_(i-n+1),...,y_(i-1))$
  *Neural $n$-gram*: Embeddings + MLP
  *RNN*: $bold(h)_i = sigma(bold(W)_h bold(h)_(i-1) + bold(W)_x bold(e)(y_(i-1)) + bold(b))$

  *Vanishing gradient*: LSTM/GRU
]

= Semirings

#cbox(title: [Definitions])[
  *Monoid* $chevron.l bb(K), circle.tiny, bold(e) chevron.r$: assoc, identity
  *Semiring* $chevron.l bb(K), plus.o, times.o, bold(0), bold(1) chevron.r$: comm monoid, monoid, distrib, annihilator

  *Closed*: $x^* = plus.o.big_(n=0)^infinity x^(times.o n)$

  Boolean, Viterbi $chevron.l [0,1], max, times, 0, 1 chevron.r$, Inside, Real, Tropical, Log, Expectation, Counting
]

= Part-of-Speech Tagging

#cbox[
  Input: $bold(w) in Sigma^N$. Output: $bold(t) in cal(T)^N$.

  *CRF*: $"score"(bold(t),bold(w)) = sum_(n=1)^N "score"(chevron.l t_(n-1),t_n chevron.r, bold(w),n)$
  $= "trans"(t_(n-1),t_n) + "emit"(w_n,t_n)$

  *Forward*: $alpha_(n,t_n) arrow.l plus.o.big_(t_(n-1) in cal(T)) exp("score"(...)) times.o alpha_(n-1,t_(n-1))$
  Return $alpha_(N,"eot")$. Runtime: $O(N|cal(T)|^2)$

  *Dijkstra*: $O(N|cal(T)|^2 + N|cal(T)| log(N|cal(T)|))$
]

= Finite-State Automata

#cbox[
  *WFST*: $Sigma$, $Omega$, $Q$, $I subset.eq Q$, $F subset.eq Q$, $lambda: I arrow bb(K)$, $rho: F arrow bb(K)$, $delta$

  *Pathsum*: $Z(cal(T)) = plus.o.big_(i,k in Q) lambda(q_i) times.o bold(R)_(i k) times.o rho(q_k)$

  *Lehmann*: $bold(R)_(i k)^((j)) arrow.l bold(R)_(i k)^((j-1)) plus.o bold(R)_(i j)^((j-1)) times.o (bold(R)_(j j)^((j-1)))^* times.o bold(R)_(j k)^((j-1))$
  Runtime: $O(|Q|^3)$

  *Composition*: $cal(T)(x,y) = plus.o.big_(z in Omega^*) cal(T)_1(x,z) times.o cal(T)_2(z,y)$
]

= Transliteration

#cbox[
  Map $Sigma^* arrow Omega^*$. Three transducers:
  1. $cal(T)_x$: maps $x arrow x$
  2. $cal(T)_bold(theta)$: maps $Sigma^* arrow Omega^*$
  3. $cal(T)_y$: maps $y arrow y$

  Compose for $Z(x)$ and $"score"(y,x)$
]

= Constituency Parsing

#cbox[
  *CFG*: $cal(N)$, $S$, $Sigma$, $cal(R)$ (rules $N arrow bold(alpha)$)
  *PCFG*: locally normalized. *WCFG*: globally normalized.

  *CNF*: $N_1 arrow N_2 N_3$ or $N arrow a$ (no cycles)

  *CKY*: $bold(C)_(i,k,X) arrow.l plus.o.big_(X arrow Y Z) exp("score"(X arrow Y Z)) times.o bold(C)_(i,j,Y) times.o bold(C)_(j,k,Z)$
  Return $bold(C)_(1,N+1,S)$. Runtime: $O(N^3 |cal(R)|)$
]

= Dependency Parsing

#cbox[
  $(N-1)^(N-2)$ spanning trees with single-root.

  $"score"(bold(t),bold(w)) = bold(rho)_r + sum_((i arrow j) in bold(t)) bold(A)_(i j)$

  *Koo MTT*: Laplacian $bold(L)_(i j) = cases(bold(rho)_j "if" i=1, -bold(A)_(i j) "if" i eq.not j, sum_(k eq.not i) bold(A)_(k j) "otherwise")$
  $Z(bold(w)) = det(bold(L))$. Runtime: $O(N^3)$

  *Chu-Liu-Edmonds*: Greedy graph → contract cycles → swap loss → expand. $O(N^2)$
]

= Semantic Parsing

#cbox[
  *Lambda calculus*: $x,y,z$; $(lambda x. f(x))$; $(M N)$
  *$beta$-reduction*: $((lambda x. M) N) arrow M[x := N]$
  *$alpha$-conversion*: $lambda x. M[x] arrow lambda y. M[y]$

  *CCG rules*:
  $X\/Y #h(0.3em) Y arrow.r.double X$ (>), $Y #h(0.3em) X backslash Y arrow.r.double X$ (<)
  $X\/Y #h(0.3em) Y\/Z arrow.r.double X\/Z$ ($bold(B)_>$)
  $X arrow.r.double T\/(T backslash X)$ ($bold(T)_>$)

  *LIG*: CFG with stacks. Push/pop rules.
]

= Transformers

#cbox[
  *Self-attention*: Learn $bold(W)_Q, bold(W)_K, bold(W)_V in RR^(d times d)$
  $"SelfAtt"(bold(X)) = "softmax"((bold(W)_Q^top bold(X))^top (bold(W)_K^top bold(X)) / sqrt(d_q)) (bold(W)_V^top bold(X))^top$
  Runtime: $O(n d^2 + d n^2)$

  *Positional encoding*: $bold(P)_(p i) = sin(p \/ 10000^(i\/d))$ or $cos$

  *Encoder*: $plus.o bold(P) arrow "MHSA" arrow plus.o arrow "LN" arrow "MLP" arrow plus.o arrow "LN"$
  *Decoder*: + linear + softmax

  *Beam search*, *nucleus sampling*
]

= Axes of Modelling

#cbox[
  *Bias-variance*: High bias = underfit. High variance = overfit.
  *Regularization*: $ell(bold(theta)) + lambda ||bold(theta)||_2^2$

  MLE: $hat(bold(theta)) = arg min_bold(theta) -log product_((x,y) in cal(D)) p_bold(theta)(y|x)$

  *Precision* = TP/PP, *Recall* = TP/(TP+FN), *F1* = $2 dot (P dot R)/(P+R)$

  *Locally norm*: efficient, label bias
  *Globally norm*: needs normalizer
]

= Tips

#cbox[
  *Gradient*: Sum over paths, product within paths.
  *Reuse terms* in backprop for efficiency.

  *Complexities*: vec-vec $O(d)$, mat-vec $O(n m)$, mat-mat $O(n m ell)$

  *Activations*:
  - $sigma(x) = 1/(1+exp(-x))$, $sigma'(x) = sigma(x)(1-sigma(x))$
  - $"ReLU"(x) = max{0,x}$, $"ReLU"'(x) = bb(1){x > 0}$
  - $tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))$, $tanh'(x) = 1-tanh^2(x)$
]
