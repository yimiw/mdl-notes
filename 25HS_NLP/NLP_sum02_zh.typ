
#import "../assets/tmp_nt.typ": *

// Configure the document with custom settings
#show: summary_project.with(
  title: "25HS_NLP",
  authors: ((name: ""),),

  // Customize for compact printing
  base_size: 9pt, //10pt不小
  heading1_size: 1.3em,
  heading2_size: 1.2em,
  math_size: 0.95em,

  // Tight spacing for printing
  par_spacing: 0.5em,
  par_leading: 0.5em,

  // Yellow-purple theme
  primary_color: rgb("#663399"),
  secondary_color: rgb("#F59E0B"),

  // Compact margins
  margin: (x: 1.25cm, y: 1.25cm),
)

// ========== CONTENT BEGINS ==========
#pagebreak()

= Semirings <sec:semirings>

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    == Motive: one algo解决多个问题

    核心洞察：计算 normalizer $Z$ 和寻找 highest-scoring path 本质上都是 *shortest path problems*。与其为每个任务设计单独algo，不如用 semiring 参数化：

    $ sum arrow.squiggly plus.o, quad product arrow.squiggly times.o $

    若原algo计算 $Z = sum_(bold(y)) product_n exp "score"(y_n)$，则 semiringified 版本计算：
    $ plus.o.big_(bold(y)) times.o.big_n exp "score"(y_n) $

    这之所以可行，是因为我们只需 associativity、commutativity (for $plus.o$)和 distributivity。
    // 我们要 *minimize assumptions*：dynamic programming 只需要 associativity/commutativity/distributivity/identity/annihilator。extra结构（inverse、subtraction、division）不但不必要，还会限制适用范围。


    == 常用 Semirings 速查


    #figure(
      table(
        columns: 6,
        align: center,
        [*Name*], [$bb(K)$], [$plus.o$], [$times.o$], [$bold(0)$], [$bold(1)$],
        [Boolean], [${0,1}$], [$or$], [$and$], [$0$], [$1$],
        [Real #footnote[严格来说，$RR_(>=0)$ 上的 $(+, times)$ 严格来说不是 semiring（$0.6+0.6=1.2 in.not [0,1]$，不封闭），但文献中常如此称呼。]],
        [$RR_(>=0)$],
        [$+$],
        [$times$],
        [$0$],
        [$1$],

        [Tropical#footnote[因曲线形态得名。Tropical geometry 与 ReLU 网络的 decision boundary 几何相关——并非冷门领域。]],
        [$RR union {infinity}$],
        [$min$],
        [$+$],
        [$infinity$],
        [$0$],

        [Viterbi], [$RR union {-infinity}$], [$max$], [$+$], [$-infinity$], [$0$],
        [Log],
        [$RR union {pm infinity}$],
        [$"lse"$#footnote[其中$"lse"(x,y) = log(e^x + e^y)$. *Log-Sum-Exp Trick*: 若$x >= y$，则
            $log(e^x + e^y) = x + log(1 + e^(y-x))$因 $y-x <= 0$，故 $e^(y-x) <= 1$，数值稳定。Motiv是计算 $log(e^x + e^y)$ 时，直接 $exp$ 会 overflow。所有神经网络库都实现了 `logsumexp`,如`torch.logsumexp(log_probs, dim=...)`。]],
        [$+$],
        [$-infinity$],
        [$0$],
      ),
      caption: [Semiring 对照表],
    )<fig:semiring-table>
  ],
  [
    == 代数结构

    #definition(title: "Monoid")[
      三元组 $chevron.l bb(K), times.o, bold(e) chevron.r$ 满足：
      1. *Associativity:* $(x times.o y) times.o z = x times.o (y times.o z)$
      2. *Identity:* $x times.o bold(e) = bold(e) times.o x = x$

      直觉：比 group 少一个 inverse 公理，所以更简单。
    ]

    #definition(title: "Semiring")[
      五元组 $chevron.l bb(K), plus.o, times.o, bold(0), bold(1) chevron.r$ 满足：
      1. $chevron.l bb(K), plus.o, bold(0) chevron.r$ 是 *commutative monoid*
      2. $chevron.l bb(K), times.o, bold(1) chevron.r$ 是 *monoid*
      3. *Distributivity:* $(x plus.o y) times.o z = (x times.o z) plus.o (y times.o z)$（左右皆需）
      4. *Annihilation:* $bold(0) times.o x = x times.o bold(0) = bold(0)$

      命名由来：比 ring 少公理（ring 要求 $plus.o$ 构成 group，有逆元）。
    ]

    #note[
      我们在逆向工程 dynamic programming 所需的最小公理集。Distributivity 是关键——它让指数sum变成多项式形式。
    ]

    #definition(title: "Idempotent Semiring")[
      若 $forall a: a plus.o a = a$，则称 semiring 是 idempotent 的。

      典型例子：$max(a, a) = a$。注意 $plus.o$ 只是 binary operation 的记号，不必然是加法！
    ]
  ],
)



#warning[
  *考试高频题型*：判断给定结构是否是 monoid/semiring。TA 强调这类题"fast, easy to check, shows understanding"。
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Monoid 判定
    Monoid 判定练习如@fig:exs-monoid-table, Monoid必须满足：
    + *Closure*：$a times.o b in bb(K)$（operation 封闭）;
    + *Associativity*：$(a times.o b) times.o c = a times.o (b times.o c)$;
    + *Identity*：$exists bold(e): a times.o bold(e) = bold(e) times.o a = a$

    常见陷阱：
    - 减法不 associative
    - 要检查 identity 是否在集合内
    - Monoid 不要求_commutativity_——string concatenation 是典型例子

    #figure(
      table(
        columns: 3,
        inset: 4pt,
        align: (center, center, left),
        [*结构*], [*Monoid?*], [*原因*],
        [$chevron.l NN, +, 0 chevron.r$], [✅], [标准例子],
        [$chevron.l NN, -, 0 chevron.r$], [✗], [不封闭：$0 - 1 = -1 in.not NN$],
        [$chevron.l ZZ, -, 0 chevron.r$], [✗], [不 associative：$(a-b)-c != a-(b-c)$],
        [$chevron.l NN, times, 1 chevron.r$], [✅], [乘法封闭、associative],
        [$chevron.l RR_(>=0), max, 0 chevron.r$], [✅], [$max$ associative，$max(a, 0) = a$],
        [$chevron.l Sigma^*, "concat", epsilon chevron.r$], [✅], [*非交换* monoid 的例子！],
      ),
      caption: [Monoid 判定练习],
    )<fig:exs-monoid-table>
  ],
  [
    === Semiring 判定清单

    除 monoid 条件外，还需：
    1. $chevron.l bb(K), plus.circle, bold(0) chevron.r$ 是 *commutative* monoid
    2. $chevron.l bb(K), times.circle, bold(1) chevron.r$ 是 monoid
    3. *Distributivity*：$a times.circle (b plus.circle c) = (a times.circle b) plus.circle (a times.circle c)$
    4. *Annihilation*：$bold(0) times.circle a = a times.circle bold(0) = bold(0)$

    #figure(
      table(
        inset: 4pt,
        columns: 3,
        [*结构*], [*Semiring?*], [*关键点*],
        [$chevron.l NN, +, times, 0, 1 chevron.r$], [✅], [counting paths],
        [$chevron.l RR_(>=0), max, times, 0, 1 chevron.r$], [✅], [unnormalized probabilities],
        [$chevron.l RR_(>=0), max, +, 0, 0 chevron.r$], [✗], [$bold(0) = bold(1) = 0$ 矛盾！],
        [$chevron.l RR union {-infinity}, min, +, +infinity, 0 chevron.r$], [✅], [shortest path (tropical)],
        [$chevron.l RR union {+infinity}, min, +, -infinity, 0 chevron.r$], [✗], [不 distributive!],
        [$chevron.l cal(P)(Sigma^*), union, "concat", emptyset, {epsilon} chevron.r$], [✅], [语言的集合，*非交换*乘法],
      ),
      caption: [Semiring 判定练习],
    )
  ],
)




#note[
  #grid(
    columns: (2fr, 3fr),
    gutter: 1em,
    [
      *关键陷阱*：$bold(0) = bold(1)$ 时必然失败。因为：
      - $a times.circle bold(0) = bold(0)$（annihilation）
      - $a plus.circle bold(0) = a$（identity）
      - 若 $bold(0) = bold(1)$，则 $a times.circle bold(1) = bold(0)$，但应有 $a times.circle bold(1) = a$
    ],
    [
      *Distributivity 检验*：$min(1, 2) + 3 = 4$，但 $min(1+3, 2+3) = 4$？✅

      反例：$min(1, 2) + 3 != min(1, 2+3)$（错误方向的 distributivity）
    ],
  )
]

== Closed Semiring 与inftysum

#definition(title: "Closed Semiring")[
  增设 *Kleene star* 运算：$a^* = plus.o.big_(n=0)^infinity a^(times.o n)$，满足：
  $ a^* = bold(1) plus.o a times.o a^* = bold(1) plus.o a^* times.o a $
]

对 real semiring 在 $(-1,1)$ 上：$a^* = sum_(n>=0) a^n = 1/(1-a)$（geometric series）。这是 globally normalized language model 的理论基础。

== 动态规划的代数推导

#warning[*考点*：Ryan明确说这是"fundamental slide"，历年必考。]

目标：计算 $Z(bold(w)) = sum_(bold(t) in cal(T)^N) exp "score"(bold(t), bold(w))$

*Step 1*：假设 score 可加分解
$ "score"(bold(t), bold(w)) = sum_(n=1)^N "score"(chevron.l t_(n-1), t_n chevron.r, bold(w), n) $

*Step 2*：代数变换
$
  Z & = sum_(bold(t)) exp sum_n "score"_n = sum_(bold(t)) product_n exp "score"_n quad "(exp 法则)" \
    & = sum_(t_1) dots.c sum_(t_N) product_n exp "score"_n quad "(展开)" \
    & = sum_(t_1) exp "score"_1 times ( dots.c sum_(t_N) exp "score"_N ) quad "(distributivity)"
$

关键：最后一步用 distributivity 把内层sum"推进去"。complexity从 $O(|cal(T)|^N)$ 降到 $O(N |cal(T)|^2)$。

#note[
  *高频考题*："若 score 依赖连续 3 个 tags 而非 2 个？"
  答：complexity变为 $O(N |cal(T)|^3)$——多一层 for 循环，推导同理。
]

// ---------- 07_part_of_speech_tagging.typ ----------
= Part-of-Speech Tagging <sec:pos-tagging>

#cbox(title: "Figure: POS Graph")[
  $cal(T) = {"N", "V", "Det"}$ 时的示意图。Inference 找最优 $N$-path；training 对所有 $N$-paths sum。虚线为 backpointers，粗线为最优路径。
] <fig:pos-graph>

== 问题定义

给定 sentence $bold(w) in Sigma^N$，输出 tag sequence $bold(t) in cal(T)^N$。Output space 大小 $|cal(T)|^N$ 指数增长，需高效algo。

与 language modeling 的区别：这里是*有限*set上的sum，无收敛问题，只有计算complexity问题。

#note[
  *linguistic备注*：POS 范畴因语言而异。欧洲语言中 adjective/verb 分明；汉语中形容词常可直接作谓语（"我高兴"而非"我是高兴的"）。
]

== Conditional Random Fields

CRF 是 structured labeling 的 conditional model，考虑 neighboring labels 的 context（不像独立分类器）。

#definition(title: "First-Order Linear-Chain CRF")[
  假设 tag 仅依赖相邻 tag：
  $ "score"(bold(t), bold(w)) = sum_(n=1)^N ["transition"(t_(n-1), t_n) + "emission"(w_n, t_n)] $
]

#note[
  Bigram 假设针对 *tags*，不限制 word representation。用 BiRNN 时，每个位置仍"看到"全句。

  局限：无法处理 *garden-path sentences*（如 "The horse raced past the barn fell"），因为无法回溯修改早期 tagging。
]

== Forward/Backward Algorithms

#algorithm(title: [Backward Algorithm])[
  从右向左计算 semiring-sum。

  1. $forall t_N: beta[N, t_N] <- bold(1)$
  2. *For* $n = N-1, dots, 0$:
  3. #h(1em) *For* $t_n in cal(T)$:
  4. #h(2em) $beta[n, t_n] <- plus.o.big_(t_(n+1)) exp("score"_{n+1}) times.o beta[n+1, t_(n+1)]$
  5. *Return* $beta[0, "BOS"]$

  complexity $O(N |cal(T)|^2)$。Forward algorithm 方向相反，形式对称。
] <alg:backward>

结构上等价于 backpropagation（都是 DAG 上的路径sum）。

=== Forward vs Backward 实现细节

#note[
  Forward 和 backward 有微妙的不对称性，源于 BOS (beginning of sequence) 存在但 EOS 不显式处理。
]

#cbox(title: "Pseudo Code 对比")[
  *Backward Algorithm:*
  ```python
  beta[N, t] = 1  # 直接初始化为 semiring 1
  for n = N-1, ..., 0:
      for t_n in T:
          beta[n, t_n] = ⊕_{t_{n+1}} exp(score) ⊗ beta[n+1, t_{n+1}]
  return beta[0, BOS]
  ```

  *Forward Algorithm:*
  ```python
  alpha[0, t] = exp(score(BOS -> t))  # 初始化包含 BOS 转移
  for n = 1, ..., N-1:
      for t_n in T:
          alpha[n, t_n] = ⊕_{t_{n-1}} alpha[n-1, t_{n-1}] ⊗ exp(score)
  return ⊕_t alpha[N-1, t]  # 需要遍历最后一列！
  ```
]

关键差异：
1. *初始化*：Backward 直接 $bold(1)$；Forward 需计算 BOS 转移
2. *终止*：Backward 返回单值 $beta[0, "BOS"]$；Forward 需 $plus.circle$ 整个最后一列
3. *循环次数*：Forward 少一次迭代，但初始化更复杂

=== Dijkstra 的局限性

Dijkstra在哪些semiring 下失效？Dijkstra 依赖 *greedy property*：一旦 node 被 finalized，其值不再更新。

#cbox(title: "Dijkstra 失效示例")[
  在 tropical semiring $chevron.l RR, min, +, +infinity, 0 chevron.r$ 中，若允许 *负权边*：

  ```
  A --3--> B --(-9)--> C
  A --7--> C
  ```

  Dijkstra 先 finalize $A -> C$ (cost 7)，但实际最短路 $A -> B -> C$ (cost $3 + (-9) = -6$) 更优。

  问题：Dijkstra 从未考虑经过 $B$ 的路径！
]

Dijkstra 有效的条件：

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    1. 所有 edge weights 非负（tropical semiring 的标准假设）

    2. 或更一般地：semiring 满足某种 *monotonicity*（加入更多 edges 不会使 path 更优）
  ],

  [对于 `sum` semiring（计算 $Z$）：Dijkstra 正确但无加速——必须考虑所有非零路径。],
)

== Viterbi Algorithm

将 backward algorithm 中的 $sum$ 换成 $max$，并记录 backpointers：

#grid(
  columns: (2fr, 1fr),
  gutter: 1.5em,
  [
    #algorithm(title: [Viterbi])[
      1. $forall t_N: beta[N, t_N] <- 1$, $"bp"[N, t_N] <- perp$
      2. *For* $n = N-1, dots, 0$; *For* $t_n$:
      3. #h(1em) $beta[n, t_n] <- max_(t_{n+1}) exp("score"_{n+1}) times beta[n+1, t_{n+1}]$
      4. #h(1em) $"bp"[n, t_n] <- arg max (dots.c)$
      5. Backtrack 得 $bold(t)^*$
    ]
  ],

  [#note[
    *为何 search-and-replace 有效？* 因为 $(max, times)$ 与 $(+, times)$ 满足相同代数性质。

    - $min$：可以（tropical semiring）
    - $sin$ 等非线性函数：不行（violates distributivity）

    *历史*：Viterbi (1967) 因此algo成名，USC 工程学院以其命名。
  ]],
)


== Dijkstra 与 Semiring

Dijkstra 通过剪枝只探索可能最短的路径。对 $max\/min$ 有效，但对 $sum$ 无加速——sum时每条非零路径都必须计入。

== Training

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [最大化 log-likelihood：
    $ cal(L) = sum_((bold(w), bold(t)) in cal(D)) ["score"(bold(t), bold(w)) - log Z(bold(w))] $],

  [
    对 forward algorithm 做 backprop 即可求梯度。

    #note[HMM 的 forward-backward 本质上就是在算这个梯度，EM 因此类似 gradient descent。]
  ],
)




== Structured Perceptron

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [对 CRF 引入 temperature $T$：
    $ p_T(bold(t)|bold(w)) = exp("score"\/T) / Z_T $

    令 $T -> 0$，softmax 变 hard max，梯度简化为：
    $ nabla cal(L) = phi(bold(t)^"gold") - phi(hat(bold(t))), quad hat(bold(t)) = arg max "score" $],

  [其中 $arg max$ 由 Viterbi 计算。

    #note[Collins (2002) EMNLP best paper。在我们的框架下只需一行推导。]],
)





// ---------- 08_wfsa_wfst.typ ----------
= Weighted Finite-State Automata <sec:wfsa>

== 动机：Transliteration

将英文名转写为日语片假名"California" -> "カリフォルニア",日语只有一个 coda 辅音 (N)，需插入额外元音。Source-target 对齐*not one-to-one*——这正是 CRF 无法直接处理的原因。

== Formal Definitions

#definition(title: "基本概念")[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.5em,
    [*Alphabet* $Sigma$：非空有限集，元素称 letters],
    [*String*：letters 的有限序列；$epsilon$ 为 empty string],
    [*Unambiguous*：每个 string 至多一条 accepting path（$neq$ deterministic！）],
  )
]

#note[
  $union_(n=0)^infinity$ 中 $n$ 遍历自然数，$infinity in.not NN$。set无穷大，但每个元素有限长。
]

#definition(title: "FSA")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [Tuple $chevron.l Sigma, Q, I, F, delta chevron.r$：alphabet、states、initial states、final states、transitions。],
    [String $w$ 被 *accept* 当且仅当存在从 $I$ 到 $F$ 的 path 拼出 $w$。],
  )
]

#definition(title: "WFSA")[

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [在 semiring $chevron.l bb(K), plus.o, times.o, bold(0), bold(1) chevron.r$ 上的 weighted 版本，增设：],
    [
      - $lambda: Q -> bb(K)$ (initial weights)
      - $rho: Q -> bb(K)$ (final weights)
      - Transitions 带权重
    ],
  )]

#definition(title: "Path Sum")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [Path weight：沿途权重的 $times.o$-积。
      $ Z(cal(A)) = plus.o.big_(pi in Pi(cal(A))) w(pi) $],

    [有 cycle 时 path 数无穷，可能发散——类似 language model 的 tightness 问题。],
  )
]

== 收敛条件

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [考虑 self-loop 权重 $x$： $Z = 1 + x + x^2 + dots.c = 1/(1-x) quad (|x| < 1)$],
  [能做 globally normalized inference 的 language model class 本质上是 *generalized geometric distributions*。],
)

== Automata 性质

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1.5em,
  [*Accessible/Co-accessible*：从 initial 可达 / 可达 final],
  [*Trim*：所有 states 都 useful],
  [*Unambiguous*：每个 string 至多一条 accepting path（$neq$ deterministic！）],
)


#note[
  Ambiguity 在 idempotent semiring 中无影响（$1 or 1 = 1$），但在 real semiring 中需对多条 path sum。
]

== Finite-State Transducers

#definition(title: "FST")[
  Transition 形如 $(q, a, b, w, q')$，$a in Sigma$, $b in Omega$。给定 input $x$，定义 output $y$ 上的条件分布。
]

#definition(title: "Composition")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [$T_1: Sigma -> Omega$，$T_2: Omega -> Gamma$，则：
      $ (T_1 compose T_2)(x, z) = plus.o.big_(y in Omega^*) T_1(x,y) times.o T_2(y,z) $],

    [形如 matrix multiplication（带 marginalization）。可层叠构建复杂model。],
  )
]


== Transliteration model
Composition 自动处理 alignment 的sum。
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1.5em,
  [1. 定义 permissive FST：任意 symbol 可映射到任意 symbol],

  [2. 学习 transition weights],

  [3. Training：最大化 likelihood，*marginalize over latent alignments*],
)
// ```
// == 预告下节课Path Sum algo解决两个问题
// 1. *判定*：$Z(cal(A))$ 是否有限？
// 2. *计算*：若有限，如何高效求值？
// 这将统一若干经典algo：automata $->$ regex 转换、Gaussian elimination、all-pairs shortest paths。
// ```

== Acyclic WFSA 的 Backward Algorithm

#grid(
  columns: (2fr, 1fr),
  gutter: 1.5em,
  [
    #algorithm(title: [Backward Algorithm (Acyclic WFSA)])[
      按 reverse topological order 遍历 nodes，应用 distributivity。

      1. 对 nodes 按 topological order 排序：$q_1, ..., q_M$
      2. *For* $m = M, ..., 1$:
      3. #h(1em) $beta[q_m] <- rho(q_m) plus.o plus.o.big_{(q_m, a, w, q') in delta} w times.o beta[q']$
      4. *Return* $plus.o.big_{q in I} lambda(q) times.o beta[q]$

      Complexity: $O(|Q| + |delta|)$（linear time）
    ]
  ],
  [
    对于 *acyclic* WFSA，可直接应用 backward algorithm。关键性质：directed acyclic graph 可做 *topological sort*。

    #note[
      Topological sort 非唯一。对 CRF 中长度 $N$、$|cal(T)|$ 个 tags 的 lattice，topological orderings 数量为 $N times (|cal(T)|!)$——同一 time step 内的 nodes 可任意顺序更新。
    ]
  ],
)


与 CRF 版本对比：这里只需 topological order，不再显式遍历 time steps 和 tags。抽象使一切更简单, which is worth it.

== Closed Semiring 与 Kleene Star

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    处理 *cyclic* WFSA 需要 infinite sums。关键工具是 *Kleene star*。

    #definition(title: "Closed Semiring (revisited)")[
      Semiring 称为 *closed* 若存在 Kleene star 运算 $a^*$ 满足：
      $
        a^* & = bold(1) plus.o a times.o a^* \
        a^* & = bold(1) plus.o a^* times.o a
      $
    ]

    这两条公理看似抽象，实则 geometric series 天然满足：
    $sum_(n>=0) x^n & = 1 + x dot sum_(n>=0) x^n = 1 + (sum_(n>=0) x^n) dot x$

    对 $|x| < 1$，closed form 为 $x^* = 1\/(1-x)$。
  ],
  [
    #note[
      *Real semiring 本身不 closed*：若 $x = 2$，则 $sum x^n$ 发散。需扩展到 *extended reals* $RR union {infinity}$。
    ]


    * Matrix Version*: 若 $bold(M)$ 是 semiring 值矩阵，则：
    $ bold(M)^* = sum_(n>=0) bold(M)^n $

    在 real semiring 中，这收敛当且仅当 $bold(M)$ 的 *largest eigenvalue $< 1$*。此时：
    $ bold(M)^* = (bold(I) - bold(M))^(-1) $

    这给出 cubic time algorithm（matrix inversion）。但问题是：semiring 没有 minus 和 inverse！
  ],
)



== Lehmann's Algorithm

Lehmann's algorithm 是处理 cyclic WFSA 的通用 dynamic program，无需 inverse 操作。

#grid(
  columns: (3fr, 2fr),
  gutter: 1.5em,
  [
    #algorithm(title: [Lehmann's Algorithm])[
      *Input:* Weight matrix $bold(W) in bb(K)^(|Q| times |Q|)$, closed semiring

      1. $bold(R)^((0)) <- bold(W)$
      2. *For* $j = 1, ..., |Q|$:
      3. #h(1em) *For* $i, k = 1, ..., |Q|$:
      4. #h(
          2em,
        ) $bold(R)_(i k)^((j)) <- bold(R)_(i k)^((j-1)) plus.o bold(R)_(i j)^((j-1)) times.o (bold(R)_(j j)^((j-1)))^* times.o bold(R)_(j k)^((j-1))$
      5. *Return* $bold(R)^((|Q|))$

      Complexity: $O(|Q|^3)$
    ]
  ],
  [
    #definition(title: "Lehmann's Recursion")[
      令 $bold(R)_(i k)^((j))$ 为从 $q_i$ 到 $q_k$、仅经过 ${q_1, ..., q_j}$ 的所有 paths 的 semiring-sum。

      *Base case:* $bold(R)^((0)) = bold(W)$（direct edges）

      *Recursion:*
      $
        bold(R)_(i k)^((j)) = bold(R)_(i k)^((j-1)) plus.o bold(R)_(i j)^((j-1)) times.o (bold(R)_(j j)^((j-1)))^* times.o bold(R)_(j k)^((j-1))
      $]
    直观理解：经过 ${1,...,j}$ 的 paths = 不经过 $j$ 的 paths $plus.o$ 经过 $j$ 的 paths。后者分解为：$i -> j$（不经过 $j$），$j$ 上的 cycles，$j -> k$（不经过 $j$）。
  ],
)

=== Floyd-Warshall 作为特例

Floyd-Warshall 是 Lehmann's algorithm 在 tropical semiring $chevron.l RR union {infinity}, min, +, infinity, 0 chevron.r$ 下的特例。

#note[
  *关键观察*：Floyd-Warshall 中没有 Kleene star！因为在 shortest path 问题中，*走 cycle 永远使路径更长*，所以 $a^* = 0$（identity of $min$）。

  这也解释了为何 Floyd-Warshall 存在：它比 naive $|cal(V)|^2$ 次 Dijkstra 快 $O(|cal(V)|)$ 倍。
]

=== Gauss-Jordan 作为特例

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    在 real semiring 中，Lehmann's algorithm 等价于 Gauss-Jordan elimination（matrix inversion）：
    $ bold(M)^* = (bold(I) - bold(M))^(-1) $
  ],
  [
    推导（注意公理 2 的使用）：
    $
      bold(M)^* = bold(I) + bold(M)^* times.o bold(M) quad "(Axiom 2)" \
      bold(M)^* - bold(M)^* bold(M) = bold(I) ,quad
      bold(M)^* (bold(I) - bold(M)) = bold(I) . qed
    $
  ],
)

=== Kleene's Algorithm 作为特例

将 Lehmann 应用于 *regular expression semiring*（$plus.o$ = union, $times.o$ = concatenation），得到 FSA $->$ regex 转换 algorithm。这是 Kleene's theorem 的构造性证明。

== Path Sum 计算总结

给定 WFSA $cal(A)$，计算 $Z(cal(A))$：

#grid(
  columns: (1.1fr, 1fr, 1.1fr, 2fr),
  gutter: 1.5em,
  [
    1. 构造 symbol-specific transition matrices $bold(W)_a$
  ],
  [
    2. Sum 得 $bold(W) = plus.o.big_a bold(W)_a$
  ],
  [
    3. 应用 Lehmann's algorithm 得 $bold(R)^((|Q|))$
  ],
  [
    4. $Z = plus.o.big_(q_i in I, q_f in F) lambda(q_i) times.o bold(R)_(i f)^((|Q|)) times.o rho(q_f)$
  ],
)






对 transliteration：compose transducers，然后用 Lehmann 计算 $Z$。可 backprop 训练，用 Viterbi semiring 做 inference。

// ---------- 10_constituency_parsing.typ ----------

= Constituency Parsing <sec:constituency_parsing>

== Syntax 与 Hierarchical Structure

*Syntax* 是 sentence structure 的数学研究，或曰 word order 的研究。核心事实：

#note[
  *Language is structured hierarchically.* 这是 overwhelming evidence 支持的事实，非假设。
]

=== Constituency

*Constituent* 是在 hierarchical structure 中作为 single unit 运作的 word group。

#cbox(title: "Example: Constituency")[
  - John speaks *Spanish* fluently. $->$ John speaks *Chinese* fluently. ✅
  - Mary programs the homework *in the lab*. $->$ Mary programs the homework *for eternity*. ✅

  可替换的部分即为 constituent。
]

=== Syntactic Ambiguity

同一 string 可对应多个 parse trees，产生不同 meanings。

#cbox(title: "Classic Ambiguity Examples")[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    [
      "Fruit flies like..."
      - [Fruit flies] [like]... — 果蝇喜欢
      - [Fruit] [flies like]... — 水果像...飞
    ],
    [
      "...elephant in my pajamas"
      - PP attach high — 我穿睡衣
      - PP attach low — 大象穿睡衣
    ],
    [
      Modifier *scope*
      - "plastic cup holder"
      - "plastic-cup holder"
    ],
  )
]

这些 ambiguities 是*事实*（data）。我们通过 introspection 收集, which is linguistics 独特之处。

=== Constituency Tests

判定 constituent 的linguistic测试：

1. *Pronoun replacement*: "Eleanor ate [the pad thai]" $->$ "Eleanor ate *it*"
2. *Clefting*: "John loves [the red car]" $->$ "*It is [the red car]* that John loves"
3. *Pro-form substitution*: "Papa eats caviar [with a spoon]" $->$ "*How* does Papa eat caviar?"

#note[
  Clefting 可消解歧义：
  - "It is *the fruit* that flies like a green banana" — 只保留解读 2
  - 这说明 syntactic operations 作用于 tree structure，而非 string
]

=== 与 Programming Languages 对比

| | Programming Lang | Natural Lang |
|---|---|---|
| Constituents | 明确标记（brackets） | 隐式 |
| Parsing | Linear time | Cubic time |
| Ambiguity | 设计上避免 | 无处不在 |
| Grammar | 已知 | 需 reverse engineer |

== Context-Free Grammars
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #definition(title: "Context-Free Grammar (CFG)")[
      四元组 $chevron.l cal(N), S, Sigma, cal(R) chevron.r$：
      - $cal(N)$: non-terminal symbols（大写字母）
      - $S in cal(N)$: start symbol
      - $Sigma$: terminal symbols（小写字母）
      - $cal(R)$: production rules，形如 $N -> bold(alpha)$，$bold(alpha) in (cal(N) union Sigma)^*$

      String $w$ 属于 language 当且仅当存在从 $S$ 开始 yield $w$ 的 derivation。
    ]
  ],
  [ 称"context-free"是因为 rule 的应用不依赖左右 context——$N$ 无论出现在哪都可被替换。

    #note[
      *CFG 是 model，非 ground truth。*
      - 它是解释 linguistic data 的工具，非大脑中真实存在的结构
      - Tree annotations 是某人的 modelling choice
      - 切勿将 treebank 视为"ground truth"
    ]
  ],
)




=== Ambiguous Grammars

若同一 string 有多于一个 derivation tree，则 grammar 是 ambiguous 的。

#cbox(title: "Parse Tree Factorization")[
  给定 CFG，tree 的 probability/score 分解到 rules：
  $ "score"(bold(t), bold(w)) = sum_(r in bold(t)) "score"(r) $

  即 tree 只是 rules 的 *multiset*——任何两棵用相同 rules（含重数）的 tree 有相同 score。
]

== Probabilistic & Weighted CFGs

#definition(title: "PCFG")[
  五元组 $chevron.l cal(N), S, Sigma, cal(R), p chevron.r$，其中 $p: cal(R) -> [0,1]$ 是 locally normalized distribution：
  $ forall N in cal(N): sum_(N -> alpha in cal(R)) p(N -> alpha) = 1 $
]

问题：PCFG 可能 non-tight（类似 LM 的问题）；可能有 infinite trees for one string。

#definition(title: "Weighted CFG (WCFG)")[
  用任意 non-negative weights 替代 probabilities。Globally normalized：
  $
    p(bold(t) | bold(w)) = (exp "score"(bold(t), bold(w))) / (sum_(bold(t)': "yield"(bold(t)') = bold(w)) exp "score"(bold(t)', bold(w)))
  $
]

问题：$Z(bold(w))$ 可能发散！

#cbox(title: "Divergence Example")[
  Rules: $S -> S$ (weight 1), $S -> a$ (weight 1)

  String "a" 有无穷多 trees（$S -> S -> ... -> S -> a$），每棵 weight 1。$Z$ 发散。
]

== Chomsky Normal Form

#definition(title: "Chomsky Normal Form (CNF)")[
  所有 rules 形如：
  - $N_1 -> N_2 N_3$（binary branching）
  - $N -> a$（terminal emission）
  - $S -> epsilon$（仅对 start symbol，仅当 $epsilon in L(G)$）
]

#theorem(title: "CNF Theorem")[
  任何 CFG $G$ 可转换为 CNF grammar $G'$，使得 $L(G') = L(G)$（或 $L(G') = L(G) - {epsilon}$）。prob也可保持。
]

CNF 的关键后果：
1. *Decidability*：长度 $N$ 的 string 的 trees 数量有限（历史上证明 CFL membership decidable 的关键步骤）
2. *Tree size fixed*：长度 $N$ 的 string 对应的 binary tree 有 $2N - 1$ 个 nodes
3. *No cycles*：不存在 $N -> ... -> N$ 的 chain

Trees 数量虽有限但仍指数级：*Catalan number* $C_N approx O(4^N / N^(3/2))$。

== Parsing Problem

给定 sentence $bold(w)$，求 distribution over trees with yield $bold(w)$：
$
  p(bold(t) | bold(w)) = (exp "score"(bold(t), bold(w))) / Z(bold(w)), quad Z(bold(w)) = sum_(bold(t): "yield"(bold(t)) = bold(w)) exp "score"(bold(t), bold(w))
$

与 CRF 的类比：
| CRF | Parsing |
|---|---|
| 给定 $bold(w)$，distribution over $bold(t) in cal(T)^N$ | 给定 $bold(w)$，distribution over trees yielding $bold(w)$ |
| Score 分解到 bigrams | Score 分解到 rules |

关键：我们只需对*特定 string* 的 trees 求和，不需整个 WCFG 的 $Z$。

#note[
  与 WFSA 不同，WCFG 的 general $Z$ 需解 *quadratic equations*（iterative methods like Newton's method），无 closed-form algo。但 per-string $Z(bold(w))$ 可用 CKY 高效计算。
]


// ---------- 10_constituency_parsing.typ (continued) ----------
== CKY Algorithm

// === 历史与命名

// CKY = Cocke-Kasami-Younger（有时写 CYK）。三人在 1960s 独立发明——那个年代论文传播慢，同一 algorithm 被多次独立发现很常见。

// #note[
//   还有第四位作者 Schwartz（Younger 的合作者），是其中最有名的（NYU 数学教授，唯一有 Wikipedia 页面的），但名字反而不在缩写里。
// ]

此 algorithm 的意义：证明了 *CFL membership 可在 polynomial time 决定*。这在当时是开放问题。

=== 为何需要 CKY？

Programming languages 设计上保证 linear-time parsing（unambiguous, deterministic CFG）。但 natural language 天然 ambiguous，需要更通用的 algorithm。

#note[
  *与 matrix multiplication 的关系*：存在 tight reduction——更快的 matrix multiplication $arrow.r$ 更快的 parsing。Sub-cubic parsing 由 Leslie Valiant（PAC learning 发明者，Turing Award）给出。
]

Jay Earley 进一步证明：对*任意* CFG（非 CNF），可达到 $O(N^3 |G|)$ 而非 $O(N^3 |G'|)$（$G'$ 是 CNF 转换后可能变大的 grammar）。

// #note[
//   *趣闻*：Earley 后来转行做婚姻心理咨询。他的网站写满心理学资质，最后一段才提到"曾是 CS 教授，写了该领域 10 篇最高引论文中的 5 篇"。
// ]

=== Algorithm 核心思想

*Span*：sentence 的 contiguous substring，如 "like a green" 在 "fruit flies like a green banana" 中。

*Chart*：dynamic programming table，$"Chart"[i, k, X]$ 存储 non-terminal $X$ 覆盖 span $[i, k)$ 的所有 derivations 的 semiring-sum。

三层 for loops：
1. 初始化 length-1 spans（terminal rules）
2. 按 span length 递增枚举
3. 对每个 span，枚举所有 split points

#algorithm(title: [CKY Algorithm])[
  *Input:* Sentence $bold(w) = w_1 ... w_N$, CNF grammar $chevron.l cal(N), S, Sigma, cal(R) chevron.r$, scoring function

  1. $bold(C) <- bold(0)$ // Chart initialization
  2. *For* $i = 1, ..., N$: // Length-1 spans
  3. #h(1em) *For* $X -> w_i in cal(R)$:
  4. #h(2em) $bold(C)[i, i+1, X] <- bold(C)[i, i+1, X] plus.o exp("score"(X -> w_i))$
  5. *For* $ell = 2, ..., N$: // Span length
  6. #h(1em) *For* $i = 1, ..., N - ell + 1$:
  7. #h(2em) $k <- i + ell$
  8. #h(2em) *For* $j = i+1, ..., k-1$: // Split point
  9. #h(3em) *For* $X -> Y Z in cal(R)$:
  10. #h(
      4em,
    ) $bold(C)[i,k,X] <- bold(C)[i,k,X] plus.o exp("score"(X -> Y Z)) times.o bold(C)[i,j,Y] times.o bold(C)[j,k,Z]$
  11. *Return* $bold(C)[1, N+1, S]$

  Complexity: $O(N^3 |cal(R)|)$
]

#note[
  *CNF 在哪里体现？* 每个 span 由恰好 2 个 sub-spans 组成（binary branching）。若允许 $k$ 个 children，complexity 变为 $O(N^(k+1))$——这就是为何需要 CNF。
]
=== CKY 详细示例

#warning[
  *考试常见*：手工填写 CKY chart。TA 强调"做 3-4 遍就记住了"。
]

#cbox(title: "CKY Chart 索引")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      Chart 索引$bold(C)[i, k, X]$：

      - $i$: span 起点（word 之前的 position）
      - $k$: span 终点（word 之后的 position）
      - $X$: non-terminal


    ],
    [
      Position 在 *words之间*：$0$ | fruit | $1$ | flies | $2$ | like | $3$ | ...

      Span $[i, k)$ 覆盖words $w_i, ..., w_(k-1)$
      , 其长度 = $k - i$
    ],
  )
]


#algorithm(title: [CKY 填表步骤])[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.2em,
    [
      *1. 初始化* (length-1)

      对每个 $w_i$，查找 $X -> w_i$：
      $ bold(C)[i, i+1, X] = "score"(X -> w_i) $
    ],
    [
      *2. 递推* ($ell = 2..N$)

      枚举 $i, k, j$，对 $X -> Y Z$：
      $ bold(C)[i,k,X] <- bold(C)[i,k,X] plus.circle \ "score" times.circle bold(C)[i,j,Y] times.circle bold(C)[j,k,Z] $
    ],
    [
      *3. 结果*

      $ bold(C)[0, N, S] $
      完整 sentence 被 start symbol 覆盖
    ],
  )
]


#note()[
  *填表顺序*：按 span 长度递增 ($ell = 1, 2, ..., N$); 遍历顺序：*同一 length 内任意*（topological order 的自由度）。

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      - *对角线* ($ell = 1$)：词性标注
      - $bold(C)[0,1] = {N}$ ← "fruit"
      - $bold(C)[1,2] = {N, V}$ ← "flies"
    ],
    [
      - *上三角* ($ell >= 2$)：组合规则
      - $bold(C)[0,2] = {N P}$ ← $N + N$
      - $bold(C)[0,5] = {S}$ ← 最终结果 ✓
    ],
  )
]



#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center,
    //fill: (x, y) => if y == 0 {black } else if x == 0 { white.lighten(0%) } else {}, // 可选：头部高亮
    stroke: 0.5pt,

    // Header row
    [*i \\ j*],
    [*1* #text(size: 0.85em, fill: black)[fruit]],
    [*2* #text(size: 0.85em, fill: black)[flies]],
    [*3* #text(size: 0.85em, fill: black)[like]],
    [*4* #text(size: 0.85em, fill: black)[a]],
    [*5* #text(size: 0.85em, fill: black)[banana]],

    // Row 0
    [*0*], [N], [NP #text(size: 0.85em)[(N N)]], [S #text(size: 0.85em)[(NP VP)]], [], [S #text(size: 0.85em)[(NP VP)]],

    // Row 1
    [*1*], [], [N, V], [VP #text(size: 0.85em)[(V NP)]], [VP #text(size: 0.85em)[(V NP)]], [VP],

    // Row 2
    [*2*], [], [], [V, P], [PP #text(size: 0.85em)[(P NP)]], [PP, VP],

    // Row 3
    [*3*], [], [], [], [Det], [NP #text(size: 0.85em)[(Det N)]],

    // Row 4
    [*4*], [], [], [], [], [N],
  ),
  caption: [Syntax Parsing 填表例子],
  // Chart 填表可视化：
  // ```c
  // //对 "fruit flies like a banana"这句话：
  //         1    2    3    4    5
  //       fruit flies like  a  banana
  //   0  [ N   ]
  //   1       [N,V ]
  //   2            [ V  ]
  //   3                 [Det]
  //   4                      [ N  ]

  //   然后填 length-2:
  //   0-2: [NP] (N N)
  //   1-3: 无匹配
  //   ...
  //   继续直到 0-5 得到 S
  // ```
)


=== Catalan Number 与 Parse Trees 数量
#grid(
  columns: (2.5fr, 1fr),
  gutter: 1.2em,
  [
    #definition(title: "Catalan Number")[
      长度 $N$ 的 string 的 binary parse trees 数量：
      $ C_N = (1, N+1) binom(2N, N) approx (4^N, N^(3/2) sqrt(pi)) $
      , $C_1 = 1, C_2 = 2, C_3 = 5, C_4 = 14, C_5 = 42, C_10 = 16796$
    ]
  ],
  [
    这解释了为何 CKY 必要：即使只有 5 个 words，也有 42 种可能的 binary tree structures。
  ],
)

=== CNF 转换


#note[
  CKY 要求 CNF。转换步骤繁琐&机械：
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 1.2em,
    [
      1. 消除 $epsilon$-productions（除 $S -> epsilon$）
    ],
    [
      2. 消除 unit productions（$A -> B$）
    ],
    [
      3. 将长 RHS 拆成 binary（引入新 non-terminals）
    ],
    [
      4. 将 terminals 与 non-terminals 混合的 RHS 分离
    ],
  ) ]

#cbox(title: "Example")[
  $"VP" -> "V" "NP" "PP"$ 不是 CNF（3 个 children）

  转换：
  - 引入 $"VP"' -> "NP" "PP"$
  - 改为 $"VP" -> "V" "VP"'$

  现在两条 rules 都是 binary。
]

=== CRF 与 CFG 的对应

#grid(
  columns: (1fr, 2fr),
  gutter: 1em,
  [
    #warning[
      *Exercise 考点*：将 CRF 写成 CFG 形式，理解两者结构对应。
    ]
    CRF 是一种 *right-recursive CFG*：

    结果：$O(|cal(T)|^2)$ 条 transition rules，与 CRF 的 transition matrix 对应。
  ],
  [
    #cbox(title: "CRF as CFG")[
      给定 tag set $cal(T)$，构造 CFG：

      - Non-terminals: $B_t$ for each $t in cal(T)$, plus $S$
      - Rules:
        - $S -> B_t$ for each $t in cal(T)$（起始）
        - $B_t -> A_t B_{t'}$ for each $t, t' in cal(T)$（transition）
        - $A_t -> w$ for each word $w$, tag $t$（emission）

      这强制 _linear_ structure——parse tree 必须是 right-branching chain。
    ]
  ],
)







#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [
    === Topological Order 视角

    CKY 的 for loops 实际上是在遍历一个 *generalized topological order*：

    - 枚举所有 triples $(i, j, k)$ 满足 $0 < i < j < k <= N$
    - 按 span length $k - i$ 递增
    - 同一 length 内，任意顺序皆可

    这与 CRF 中同一 time step 内 tags 可任意顺序更新是同一 insight。
  ],
  [
    === Semiring 化与 Viterbi

    CKY 可用任意 semiring：

    - *Real semiring*：计算 $Z(bold(w))$（normalizer）
    - *Viterbi semiring*：找 best parse（配合 backpointers）
    - *Entropy semiring*：计算 parse distribution 的 entropy


  ],
  [
    === Training

    Scoring function 可以是任意 neural network。Training 方式与 CRF 相同：
    $ cal(L) = sum_((bold(w), bold(t)) in cal(D)) ["score"(bold(t), bold(w)) - log Z(bold(w))] $

    对 CKY forward pass 做 backprop 即可求 gradient。
  ],
)

#tip[
  *与 Assignment 2 的联系*：Given已经用 semiring 算过 entropy。同样的 semiring 直接 plug into CKY 即可算 parse trees 的 entropy。这就是 abstraction 的 power。
]




=== Weighted CKY 与 Semirings

与 CRF 相同，CKY 可用不同 semirings：

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 10pt,
    align: horizon,
    stroke: 0.75pt,
    table.header(
      [*Semiring*],
      [*操作 $(plus.circle, times.circle)$*#footnote[关键性质
          *幺元 (Identity)*：加法幺元 $bold(0)$，乘法幺元 $bold(1)$
          ；*零元 (Annihilator)*：加法零元使 $a times.circle bold(0) = bold(0)$
          ；*分配律*：$a times.circle (b plus.circle c) = (a times.circle b) plus.circle (a times.circle c)$
          ；*结合律*：两个操作都满足结合律
          ；*交换律*：加法操作通常满足交换律（乘法不一定）
        ]],
      [*计算内容 / 应用场景* #footnote[NLP应用示例：
          CRF/HMM：使用 Log 半环进行训练，Viterbi 半环进行解码
          ; PCFG：Inside-Outside 算法使用 Real 半环
          ; 神经网络：前向传播使用 Real 半环，反向传播涉及 Expectation 半环
          ; 机器翻译：束search使用 k-best 半环
          ; 依存句法分析：最大生成树使用 MaxPlus 半环
        ]],
    ),
    [Real/Probability], [$(+, times)$], [$Z(bold(w))$ (partition function/normalizer)；前向-后向算法],

    [Tropical/Viterbi], [$(max, times)$], [最优路径/解析树；Viterbi 算法；最大似然解码],

    [Log], [$(op("logsumexp"), +)$], [$log Z(bold(w))$；数值稳定的prob计算；避免下溢],

    [Boolean], [$(or, and)$], [是否存在有效路径；可达性判断；语法解析存在性],

    [Counting], [$(+, times)$], [路径/推导数量；歧义度计算；派生树计数],

    [k-best Tropical], [$(max_k, times)$], [Top-k 最优路径；k-best Viterbi；束search (beam search)],

    [Expectation], [$(+, times)$ over $RR times RR$], [特征期望；梯度计算；EM 算法 E-step],

    [MinPlus/Tropical], [$(min, +)$], [最短路径；编辑距离；CKY 最小代价解析],

    [Inside], [$(+, times)$], [Inside prob；PCFG 内向算法；子树prob],

    [Outside], [$(+, times)$], [Outside prob；PCFG 外向算法；上下文prob],

    [Entropy], [特殊组合], [Shannon 熵计算；不确定性度量；模型置信度],

    [Risk/Loss], [$(+, times)$ with loss], [期望风险；最小贝叶斯风险解码；损失感知训练],
  ),
  caption: "Semiring 及其应用",
)<fig:semiring-app-table>


= Dependency Parsing <sec:dependency_parsing>

== Dependency Grammar 简介

Dependency grammar 是 constituency grammar 的替代传统——两者都是 *models*（某人对 language structure 的 opinion），解释不同 phenomena，可互补。

核心思想：sentence 中每个 word 与其 *syntactic head* 连接，形成 directed tree。

#definition(title: "Dependency Tree")[
  - 每个 word 有唯一 parent（head）
  - 有唯一 root（通常是 main verb）
  - Edges 带 grammatical relation labels（subject, object, etc.）
]


"The boy eats Rösti"

Relations: boy $->^"subj"$ eats, Rösti $->^"obj"$ eats, The $->^"det"$ boy

=== 为何 Dependency 与 Function Application 相关？

Verb 可视为 function，arguments 是其 dependents：
$ "eats" = lambda y. lambda x. "Eats"(x, y) $

应用后：$"eats"("Rösti")("boy") = "Eats"("boy", "Rösti")$

这是 *argument structure* 的浅层建模——verb 接受哪些 arguments、如何标记（preposition, case marking 等）。

=== Dependency vs Constituency

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  align: (center, center, center),
  stroke: (x, y) => if y == 0 or y == 4 { 1pt } else { 0.5pt },
  // 表头加粗
  [#text(weight: "bold")[Aspect]], [#text(weight: "bold")[Constituency]], [#text(weight: "bold")[Dependency]],
  // data行（不变）
  [基本单位], [Phrases (constituents)], [Head-dependent pairs],
  [标注内容], [Phrase structure], [Grammatical relations],
  [信息], [Hierarchy, scope], [Head selection, valency],
  [相互关系], [可互相转换], [但 tree 结构不同],
)

从 constituency tree 提取 dependency tree：为每个 rule 指定 head child（Collins' rules），然后向上传播 head。

=== Projectivity

#definition(title: "Projective Dependency Tree")[
  若将 arcs 画在 words 上方，*没有 crossing arcs*，则称 tree 是 *projective* 的。
]

- *Projective*：与 constituency structure 紧密相关，可用 CKY 变体 parse
- *Non-projective*：有 crossing arcs，需要不同 algorithms

#cbox(title: "Non-projectivity Example")[
  "Ryan ate Rösti for lunch, which was delicious."

  "which" 修饰 "Rösti"，但 "for lunch" 介于两者之间。Arc (Rösti, which) 跨过 arc (ate, for)。

  这是英语中的 *non-contiguous constituent* 现象——CFG 假设开始 break。
]

大多数语言 *nearly projective*——non-projective structures 存在但相对稀少。

=== Universal Dependencies

*UD (Universal Dependencies)* 是跨语言的 dependency annotation 标准，覆盖数百种语言。

#note[
  *为何 dependency parsing 如此流行？* 主要是 *convenience*。UD 提供免费、统一格式的 data，便于 benchmark。相比之下，constituency treebanks 多在 paywall 后且格式各异。

  这是 NLP 的社会学现象：data 的 accessibility 极大影响 research 方向。
]

== Probability Model for Non-Projective Trees

目标：给定 sentence $bold(w)$，定义 spanning trees 上的 distribution。

问题规模：$N$ 个 nodes 的 directed spanning trees 数量是 $(N-1)^(N-2)$（Cayley's formula 的 directed 版本）——比 parse trees 的 Catalan number 还大。

=== Edge Factorization

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #definition(title: "Edge-Factored Model")[
      假设 scoring function 分解到 edges：
      $ "score"(bold(t), bold(w)) = "score"(r, bold(w)) + sum_((i -> j) in bold(t)) "score"(i, j, bold(w)) $

      其中 $r$ 是 root choice。
    ]
  ],
  [
    将 scores 组织成 matrices：
    - $bold(A)_(i j) = exp "score"(i, j, bold(w))$：weighted adjacency matrix
    - $bold(rho)_j = exp "score"(r = j, bold(w))$：root scores

    #note[
      *为何不能更强？* Edge factorization 是能保持 tractability 的 *strongest* assumption。若允许 second-order（同时看两条 edges），可 encode Hamiltonian path problem（NP-hard）。
    ]
  ],
)



== Matrix-Tree Theorem
=== Root Convention

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #note[
      Dependency tree 引入 external root node（不在 sentence 内），有一条 arc 指向 sentence 的 syntactic head（通常是 main verb）。这让 root choice 也变成普通的 edge choice。
    ]
  ],
  [两种等价写法：
    - Root as special node $0$：edges $0 -> j$ 表示 $w_j$ 被选为 root
    - Root scores vector $bold(rho)$：$bold(rho)_j = exp "score"(r = j, bold(w))$

    两者最终都落到对 Laplacian $bold(L)$ 第一行的修改（Koo et al. trick）。],
)



=== Arc Scoring: First-Order vs Higher-Order

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #definition(title: "First-Order (Arc-Factored)")[
      Score 仅依赖单条 arc：

      $"score"(bold(t), bold(w)) = "score"(r, bold(w)) + sum_((i -> j) in bold(t)) "score"(i, j, bold(w))$

      其中 $i$ = head, $j$ = dependent。Complexity: $O(N^2)$ arcs。
    ]
  ],
  [
    #cbox(title: "Second-Order: Grandparent")[
      Score 还依赖 grandparent $g$（$i$ 的 parent）：

      $ "score"(bold(t)) = sum_((g -> i -> j) in bold(t)) "score"(g, i, j, bold(w)) $

      Example: "eat a red apple"
      则Arc: apple $->$ red; Grandparent: eat（因为 eat $->$ apple）
    ]
  ],
  [
    #cbox(title: "Second-Order: Sibling")[
      Score 还依赖 sibling $s$（同一 head 下的其他 dependents）：

      $ "score"(bold(t)) = sum_((i -> j, i -> s) in bold(t)) "score"(i, j, s, bold(w)) $

      Example: "eat an apple and an orange"
      则Head: eat; Siblings: apple, orange
    ]
  ],
)







=== Complexity 分析与 Extreme Structures


#cbox(title: "Arc Scoring Templates")[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.5em,
    [
      Extreme Trees for Complexity直觉:

      一般来说First-order: *$O(N^2)$*; Second-order: *$O(N^3)$*
    ],
    [*Flat tree*：one head with $N-1$ dependents
      - First-order: $O(N)$ arcs
      - Sibling: $O(N^2)$（每条 arc 有 $N-2$ siblings）
    ],
    [
      *Chain tree*：each node has exactly one child (a path)
      - First-order: $O(N)$ arcs
      - Grandparent: $O(N)$（每条 arc 只有 1 grandparent）
    ],
  )
]



#note[
  为何 first-order 够用？Neural scoring function（BiLSTM/Transformer）已在 input representation 中编码丰富 context。Arc-factored assumption 作用于 scoring，不限制 representation。
  Edge-factored 是"还能做 exact inference"的最强可用假设。
]

=== MTT 证明结构与 Sanity Checks



#tip[Assignment 5的证明路线图
  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *Step 1*: Undirected Case (Kirchhoff)
      - 构造 adjacency matrix $bold(A)$
      - 构造 Laplacian $bold(L) = bold(D) - bold(A)$
      - 证明 $det(hat(bold(L))_i) =$ spanning trees 数量
    ],
    [
      *Step 2*: Directed Case (Tutte)
      - 修改 Laplacian 为 directed version
      - 加入 root constraint (Koo et al.)
    ],
  )
]

#definition(title: "Laplacian Matrix 构造")[
  *Undirected*：
  $
    bold(L)_(i j) = cases(
      sum_k bold(A)_(i k) & "if" i = j "(degree)",
      -bold(A)_(i j) & "otherwise"
    )
  $

  *Directed with root* (课程写法，incoming sum on diagonal)：
  $
    bold(L)_(i j) = cases(
      sum_(k != j) bold(A)_(k j) & "if" i = j,
      -bold(A)_(i j) & "if" i != j
    )
  $

  *注入 root scores*：$bold(L)_(1, j) <- bold(rho)_j$
]

*Sanity Checks*:
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [1. Undirected：列和为0 \ $det(bold(L))=0$，需 cofactor],
  [2. Directed + root：\ $det(bold(L)) != 0$ 即 $Z(bold(w))$],
  [3. 若 undirected det≠0 \ → 构造错误],
)

#v(0.5em)

#cbox(title: "MTT Workflow (Mechanical)")[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    [*1.* Build Laplacian：\ diag = col sum \ off-diag = $-bold(A)_(i j)$],
    [*2.* Inject root：\ $bold(L)_(1, j) <- bold(rho)_j$],
    [*3.* Partition function：\ $Z(bold(w)) = det(bold(L))$],
  )
]
=== Chu-Liu-Edmonds 详解

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #algorithm(title: [CLE Hand-Run Checklist])[
      *Repeat until no cycle*:
      1. Greedy step：对每个 non-root node，选 highest incoming arc
      2. Cycle detection：检查 greedy graph 是否有 cycle
      3. If no cycle：done
      4. If cycle exists：
        - Contract cycle into super-node $c$
        - Reweight entering edges
      5. Recurse on contracted graph
      6. Expand cycles using recorded choices
    ]
  ],
  [
    #cbox(title: "Reweighting 公式")[
      设 cycle $C$ 内节点 $v$ 的 best incoming arc 权重为 $w_v$。

      对外部节点 $u$ 到 $v in C$ 的 arc：
      $ "new\_weight"(u, v) = "weight"(u, v) - w_v $

      直觉：选择 $(u, v)$ 意味着放弃 $v$ 在 cycle 内的 arc。Reweight 确保 total cost 正确。
    ]

    #cbox(title: "Root Constraint 处理")[
      CLE base version 允许 root 有多条 outgoing arcs，但 dependency parsing 要求 root 只有 1 outgoing。

      *Naive*：对每条 root arc 分别运行 CLE，取最优。Complexity $O(N dot "CLE")$。

      *Clever* (Gabow et al.)：计算 swap score = next-best incoming - current incoming，删除 swap score 最小的多余 root edge。
    ]
  ],
)


=== Arc Scoring Functions (Implementation)

#grid(
  columns: (2fr, 1fr),
  gutter: 1em,
  [
    #cbox(title: "Arc Scoring Templates")[
      #grid(
        columns: (1fr, 1fr),
        gutter: 1.5em,

        [Let $bold(h)_i$ be representation of word $i$ (from BiLSTM/Transformer encoder).

          Then set $bold(A)_(i j) = exp "score"(i, j, bold(w))$],
        [Common choices:
          - Bilinear: $"score"(i,j) = bold(h)_i^top bold(W) bold(h)_j$
          - MLP: $"score"(i,j) = bold(v)^top tanh(bold(W)_h bold(h)_i + bold(W)_d bold(h)_j)$
        ],
      )
    ]
  ],
  [#note[
    If you add label types $ell$ (subj/obj/etc)：
    $ "score"(i,j) = max_ell "score"(i, j, ell) $
    或在 decoding 时 explicitly 保留 labels。
  ]],
)



// #cbox(title: "Neural Parser Pipeline (Assignment 5)")[
//   #table(
//     columns: (auto, 1fr),
//     align: (left, left),
//     stroke: 0.5pt + gray,
//     inset: 6pt,
//     [*Step*], [*Description*],
//     [0], [Data loading (Universal Dependencies format)],
//     [1], [Tokenization (subword: BPE/WordPiece)],
//     [2], [Encoding (BiLSTM / Transformer)],
//     [3], [Arc scoring (MLP: $(bold(h)_i, bold(h)_j) -> "score"(i,j)$)],
//     [4], [Decoding (CLE for best tree)],
//     [5], [Training (cross-entropy on gold arcs)],
//   )
// ]

=== 与 CRF 的结构对比

#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  stroke: 0.5pt + gray,
  inset: 6pt,
  [*Aspect*], [*CRF (Sequence)*], [*Dependency Parsing*],
  [Structure], [Linear chain], [Tree],
  [$Z$ computation], [Forward algorithm], [Determinant (MTT)],
  [Complexity], [$O(N |cal(T)|^2)$], [$O(N^3)$],
  [Inference], [Viterbi], [CLE],
  [Factorization], [Edge-factored (transitions)], [Arc-factored],
)
=== Kirchhoff's Theorem (Undirected Case)

#theorem(title: "Kirchhoff's Matrix-Tree Theorem")[
  给定 undirected graph $cal(G)$，令 Laplacian matrix：
  $
    bold(L)_(i j) = cases(
      -bold(A)_(i j) & "if" i != j,
      sum_(k != i) bold(A)_(k j) & "if" i = j
    )
  $

  则 spanning trees 数量 $= det(hat(bold(L))_i)$，其中 $hat(bold(L))_i$ 是删去第 $i$ 行列后的 matrix。
]

=== Tutte's Extension (Directed & Weighted)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    Tutte 推广到 *directed weighted* graphs：
    $ Z(bold(w)) = det(bold(L)) $

    其中 $bold(L)$ 用 directed adjacency matrix 构造。
  ],
  [
    #note[
      Undirected case 中 $bold(A)$ 是 symmetric；directed case 不对称。
    ]
  ],
)

=== Adding Root Constraint (Koo et al., 2007)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    为满足 single-root constraint，修改 Laplacian：
    $
      bold(L)_(i j) = cases(
        bold(rho)_j & "if" i = 1,
        -bold(A)_(i j) & "if" i != j,
        sum_(k != i) bold(A)_(k j) & "otherwise"
      )
    $

    结论：$Z(bold(w)) = det(bold(L))$，complexity $O(N^3)$（determinant computation）。
  ],
  [
    #note[
      *魔法公式*：整个 normalizer 就是一个 matrix determinant。这不是 dynamic program，而是 linear algebra。

      *Semiring 问题*：Determinant 需要 *subtraction*（Laplacian 定义中有负号）。若没有 subtraction，需 exponential time。这就是为何无法 semiringify。
    ]
  ],
)



== Inference: Chu-Liu-Edmonds Algorithm

Matrix-tree theorem 算 $Z$，但不给 argmax。需要另一个 algorithm 找 best tree。

=== 为何 Kruskal 不 work？

Kruskal's algorithm（greedy add edges，不成 cycle）对 *undirected* minimum spanning tree 有效。

但 directed case 失败：

#cbox(title: "Greedy 失败示例")[
  Greedy 选 highest incoming edge to each node，得到 score 7 的 tree。但 optimal tree score 是 10。

  原因：directed edges 有 constraints（每个 node 最多一条 incoming edge），locally optimal choices 可能 block globally optimal solutions。
]

=== Algorithm Overview

Chu-Liu-Edmonds (1965) / Edmonds (1967)：

#algorithm(title: [Chu-Liu-Edmonds])[
  1. *Greedy graph*：每个 non-root node 选 highest incoming edge
  2. *If* no cycle: done（greedy graph 即 optimal）
  3. *If* cycle exists:
    - *Contract* cycle 成单个 node
    - *Reweight* entering edges：新 weight = 原 weight + (被替换的 cycle edge 的 weight)
    - *Recurse* on contracted graph
  4. *Expand* contracted cycles，得到 final tree
]
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Edge Cataloging

    对 contracted node $c$：

    - *Dead edges*：cycle 内部的 edges（已处理）
    - *External edges*：不涉及 cycle 的 edges
    - *Enter edges*：进入 cycle 的 edges（需 reweight）
    - *Exit edges*：离开 cycle 的 edges

  ],
  [
    === Root Constraint Handling

    Naive：对每个可能的 root edge 分别运行 algorithm，比较结果。Complexity 增加 factor of $N$。

    Clever（Gabow et al.）：在 greedy graph 中若 root 有多条 outgoing edges，删除 *swap score* 最小的（swap score = next-best incoming edge - current incoming edge）。

  ],
)

=== Complexity
#grid(
  columns: (1fr, 1.5fr),
  gutter: 1em,
  [
    - Edmonds' original: $O(N^3)$ 或 $O(M N)$
    - Tarjan's improvement: $O(N^2)$ 或 $O(M log N)$

  ],
  [
    #tip[
      *非 Dynamic Program*：这是 assignment 中唯一一个非 DP 的 algorithm。无法 semiringify——想要不同的 computation（如 entropy）需要用 Matrix-Tree Theorem 的 gradient tricks。
    ]
  ],
)



/*
#cbox(title: "常见考题类型")[
  1. 给 simple CFG 和 sentence，画出 CKY chart 的部分填充
  2. 给 dependency tree，判断是否 projective
  3. 解释为何 edge factorization 是 tractability boundary
  4. 对比 constituency vs dependency parsing 的 pros/cons
  5. Matrix-Tree Theorem 的 Laplacian 构造
]
*/
// ---------- 12_semantic_parsing.typ ----------

= Semantic Parsing <sec:semantic_parsing>

== 什么是 Meaning？

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    Syntax 研究 sentence structure；semantics 研究 meaning, which is a philosophical question.

    #definition(title: "Truth-Conditional Semantics")[
      理解一个 expression 的 meaning，即知道它在何种条件下为 true。
      类比数学：理解 $F = F(x)$ 的 meaning，即知道哪些 $F$ 使之为 true/false。
    ]

    为何必须成立？
    我们能理解unheared, novel sentences从;这只可能因为其由可重用的parts组成; Plagiarism detection 的基础：language 太 expressive，独立产生相同句子的prob极低


  ],
  [

    #cbox(title: "Example: Quantifier Scope Ambiguity")[
      "Everybody loves somebody else" 有两个 readings：

      1. $forall p ["Person"(p) arrow exists q ["Person"(q) and p != q and "Loves"(p, q)]]$
        - 每个人都有（可能不同的）某个他们爱的人

      2. $exists q ["Person"(q) and forall p ["Person"(p) and p != q arrow "Loves"(p, q)]]$
        - 存在某个特定的人，被所有人爱

      这是 *semantic ambiguity*——非 lexical（词义歧义）、非 syntactic（结构歧义），而是 quantifier scope 的歧义。
    ]
  ],
)


=== Logical Form

*Logical form* 是 meaning 的形式化表示#footnote[
  还可以是First-order logic formulas, Lambda calculus expressions, SQL queries, Python code, Robot commands
]


关键：logical form 是 *可执行/可求值* 的——可以判断 true/false 或产生 action。

#note[
  如果学过 programming language theory：这与 PL 中的 denotational semantics 是同一思想——通过 evaluation 定义 meaning。
]

== Principle of Compositionality

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #definition(title: "Frege's Principle of Compositionality")[
      The meaning of a complex expression is a *function* of the meanings of its constituent parts.
    ]

    为何必须成立？
    我们能理解 *novel sentences*（从未听过的句子）;这只可能因为 sentence 由可重用的 parts 组成; Plagiarism detection 的基础：language 太 expressive，独立产生相同句子的prob极低
  ],
  [
    #note[
      *Idioms* 是例外（如 "kick the bucket" = die），但：
      - 可以修改 idiom："He kicked the bucket *yesterday*"
      - 打破 idiom："He kicked the *red* bucket" 失去 idiomatic reading
      - Idioms 是更大的 lexical units，compositionality 仍在更高层面成立
    ]
  ],
)

== Lambda Calculus

Lambda calculus (Church, 1932) 是 computation 的形式化model，与 Turing machine 等价（Church-Turing thesis）。

对我们而言，它主要是 *notation for anonymous functions*——Python 中的 `lambda` 即来源于此。

=== 语法定义


#definition(title: "Lambda Calculus Terms")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      Terms 归纳定义：

      *Base case:* Variables $x, y, z, ...$ 是 terms

      *Recursive rules:*
    ],
    [

      - *Abstraction:* 若 $M$ 是 term，$x$ 是 variable，则 $lambda x. M$ 是 term
      - *Application:* 若 $M, N$ 是 terms，则 $(M N)$ 是 term（$M$ applied to $N$）
    ],
  )
]


#cbox(title: "Scope Example")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      $(lambda x. x (lambda x. x) x)$

      第1、3个 $x$ → 外层 $lambda x$；第2个 $x$ → 内层 $lambda x$
    ],
    [
      类似 Python 嵌套 function 的 variable scoping——内层 scope 遮蔽外层
    ],
  )
]


=== Free vs Bound Variables


#definition(title: "Free and Bound Variables")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *Bound*：在某个 $lambda$ 的 scope 内

      *Free*：不在任何 abstraction 的 scope 内

      *归纳定义*:
    ],
    [

      - $"FV"(x) = {x}$
      - $"FV"(lambda x. M) = "FV"(M) - {x}$
      - $"FV"(M N) = "FV"(M) union "FV"(N)$
    ],
  )
]

#note[
  *为何需要 free variables？*

  虽然最终我们关心 *closed terms*（无 free variables），但定义 semantics 时必须处理 sub-expressions，而 sub-expressions 自然包含 free variables。这是 compositionality 在 formalism 中的体现。
]

=== Alpha Conversion

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #definition(title: "α-conversion")[
      重命名 bound variable 及其所有 bound occurrences：
      $ lambda x. M arrow_alpha lambda y. M[x := y] $

      条件：$y$ 不能是 $M$ 中的 free variable（否则会改变 meaning）。
    ]
  ],
  [
    #cbox(title: "需要 α-conversion 的情况")[
      $(lambda x. lambda y. x y) y$
      直接 $beta$-reduce 会得到 $lambda y. y y$——但原本外层的 $y$ 是 free 的，现在变 bound 了！

      正确做法：先 α-convert 内层 $lambda y. x y arrow_alpha lambda z. x z$，再 reduce。另外$lambda x. x arrow_alpha lambda y. y$ 这种是无所谓的✅。
    ]
  ],
)


=== Beta Reduction

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #definition(title: "β-Reduction")[
      Function application——将 argument 代入 function body：
      $ (lambda x. M) N arrow_beta M[x := N] $

      即：找到 $M$ 中所有被该 $lambda x$ 绑定的 $x$，替换为 $N$。
    ]
  ],
  [
    #cbox(title: "Example")[
      $(lambda x. lambda y. x y) z quad$
      $arrow_beta lambda y. z y$ （$x$ 被替换为 $z$）
    ]

    #warning[
      Ryan明确说会考 beta reduction。给定 Lambda expression，simplify it（反复 apply α-conversion 和 $beta$-reduction 直到无法继续）。TA也在tutorial中强调了至少是10分题。
    ]
  ],
)










=== Lambda Calculus实战

#note[
  TA建议的应试技巧：
  1. 写 Lambda calculus 时*使用不同变量名*，避免 α-conversion
  2. 每步标注*哪个 $lambda$ 正在 apply*
  3. 检查 free variables 是否会被 capture
  4. 遇到 identity function $(lambda x. x)$ 直接替换
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #cbox(title: "例题 A：无需 α-conversion")[
      化简 $(lambda z. z)((lambda y. y y)(lambda x. x a))$

      Step 1: 识别最外层 application：$(lambda z. z)$ applied to $(...)$

      Step 2: $(lambda z. z)$ 是 identity function，直接返回 argument：
      $ arrow_beta (lambda y. y y)(lambda x. x a) $

      Step 3: 继续 reduce：$y := (lambda x. x a)$
      $ arrow_beta (lambda x. x a)(lambda x. x a) $

      Step 4: 再次 reduce：外层 $x := (lambda x. x a)$
      $ arrow_beta (lambda x. x a) a $

      Step 5: 最终：$x := a$
      $ arrow_beta a a $

      结果：$a a$（两个 free variables，无法再 reduce）
    ]
  ],
  [
    #cbox(title: "例题 C：需要 α-conversion")[
      化简 $(lambda x. (lambda y. y) x)((lambda y. y) y)$

      *Trap*：若直接把 outer $y$（free）代入 inner $lambda y$，会被错误 bind！

      Step 1: 先 α-convert 内层 $lambda y$ 为 $lambda a$：
      $ (lambda x. (lambda a. a) x)((lambda y. y) y) $

      Step 2: 化简 argument：$(lambda y. y) y arrow_beta y$
      $ (lambda x. (lambda a. a) x) y $

      Step 3: 代入 $x := y$：
      $ (lambda a. a) y $

      Step 4: 最终：
      $ arrow_beta y $
    ]
  ],
)






=== First-Order Logic 翻译

// #note[
//   *TA 警告*："Be careful with negation. I lost some points because I was not very careful."
// ]

#table(
  columns: (auto, auto, auto),
  inset: 4pt,
  align: (left, center, left),
  stroke: 0.75pt,
  table.header([*English Pattern*], [*FOL Formula*], [*Example*]),

  table.cell(colspan: 3, fill: rgb(240, 240, 240))[*条件与蕴含 (Conditionals & Implications)*],

  ["If A, then B" \ "A implies B"],
  [$A arrow B$],
  ["If it rains, then the ground is wet" \ $"Rain"(x) arrow "Wet"(x)$],

  ["A if and only if B" \ "A iff B"],
  [$A arrow.l.r B$],
  ["You pass iff you score ≥60" \ $"Pass"(x) arrow.l.r "Score"(x) >= 60$],

  ["Unless A, B" \ "B unless A"],
  [$not A arrow B$ \ or $A or B$],
  ["You fail unless you study" \ $not "Study"(x) arrow "Fail"(x)$],

  table.cell(colspan: 3, fill: rgb(240, 240, 240))[*逻辑连接词 (Logical Connectives)*],

  ["A and B" \ "Both A and B"],
  [$A and B$],
  ["It's cold and raining" \ $"Cold"() and "Raining"()$],

  ["A or B" \ "Either A or B"],
  [$A or B$],
  ["Tea or coffee" \ $"Tea"(x) or "Coffee"(x)$],

  ["Not A" \ "It is not the case that A"],
  [$not A$],
  ["Not happy" \ $not "Happy"(x)$],

  ["Neither A nor B"],
  [$not A and not B$ \ or $not (A or B)$],
  ["Neither rich nor famous" \ $not "Rich"(x) and not "Famous"(x)$],

  table.cell(colspan: 3, fill: rgb(240, 240, 240))[*全称量词 (Universal Quantifiers)*],

  ["All/Every/Each A is B" \ "Everyone/Everything"],
  [$forall x [A(x) arrow B(x)]$],
  ["All dogs bark" \ $forall x ["Dog"(x) arrow "Bark"(x)]$],

  ["No A is B" \ "No one/Nothing"],
  [$not exists x [A(x) and B(x)]$ \ or $forall x [A(x) arrow not B(x)]$],
  ["No student is lazy" \ $not exists x ["Student"(x) and "Lazy"(x)]$],

  ["Only A is B"],
  [$forall x [B(x) arrow A(x)]$],
  ["Only students can register" \ $forall x ["Register"(x) arrow "Student"(x)]$],

  table.cell(colspan: 3, fill: rgb(240, 240, 240))[*存在量词 (Existential Quantifiers)*],

  ["Some/A/An A is B" \ "Someone/Something"],
  [$exists x [A(x) and B(x)]$],
  ["Some student is smart" \ $exists x ["Student"(x) and "Smart"(x)]$],

  ["There exists/is an A"],
  [$exists x [A(x)]$],
  ["There exists a solution" \ $exists x ["Solution"(x)]$],

  ["At least n A's"],
  [$exists x_1 ... x_n [and.big_(i=1)^n A(x_i) and and.big_(i<j) x_i eq.not x_j]$],
  ["At least 2 witnesses" \ $exists x_1 x_2 ["Witness"(x_1) and "Witness"(x_2) and x_1 eq.not x_2]$],

  table.cell(colspan: 3, fill: rgb(240, 240, 240))[*Complex Patterns*],

  ["Most A's are B"],
  [需要二阶逻辑或计数],
  ["Most birds fly" \ (超出 FOL 表达能力)],

  ["The A" (唯一性)],
  [$exists x [A(x) and forall y [A(y) arrow y = x]]$],
  ["The king of France" \ $exists x ["King"(x, "France") and forall y ["King"(y, "France") arrow y = x]]$],

  ["Every A has a B"],
  [$forall x [A(x) arrow exists y [B(y) and "Has"(x,y)]]$],
  ["Every person has a mother" \ $forall x ["Person"(x) arrow exists y ["Mother"(y,x)]]$],

  ["A's B" (所有格)],
  [$B(x) and "Possess"(A, x)$ \ or $"iota" x [B(x) and "Of"(x, A)]$],
  ["John's car" \ $"Car"(x) and "Possess"("John", x)$],
)

/*
*量词作用域规则：*
- *否定与量词*：$not forall x P(x) equiv exists x not P(x)$；$not exists x P(x) equiv forall x not P(x)$
- *量词顺序*：$forall x exists y P(x,y) eq.not exists y forall x P(x,y)$ （顺序很重要！）
- *限定量化*：$forall x in S : P(x) equiv forall x [S(x) arrow P(x)]$；$exists x in S : P(x) equiv exists x [S(x) and P(x)]$
- *空域问题*：当域为空时，$forall x P(x)$ 为真（空真），$exists x P(x)$ 为假

*常见陷阱：*
- "Only" 的方向性：注意蕴含箭头的方向
- 否定的作用域：区分 "not all" vs "all not"
- 存在唯一性：需要同时表达存在性和唯一性
*/

#grid(
  columns: (1fr, 5fr),
  gutter: 1em,
  [
    #note[
      多种正确答案可能存在——只要逻辑等价即可。关键是*结构正确*。
    ]
  ],
  [
    #example()[例题：嵌套量词
      "If one of Abigail's brothers makes noise, Abigail cannot sleep."

      *分析*：
      - 结构：If-then $arrow$ 使用 "$arrow$"
      - "One of Abigail's brothers" $arrow$ $exists x. "Brother"(x, "Abigail")$
      - "makes noise" $arrow$ $"MakeNoise"(x)$
      - "cannot sleep" $arrow$ $not "Sleep"("Abigail")$
      *FOL*：
      $ (exists x. "Brother"(x, "Abigail") and "MakeNoise"(x)) arrow not "Sleep"("Abigail") $
    ]
  ],
)





=== Linear Indexed Grammar 构造策略

#warning[
  *TA 强调*："In the final exam last year, there was a very similar question... the idea is the same."
]

#cbox(title: "LIG 构造思路")[
  核心问题：CFG 无法"计数"——无法保证 $a^n b^n c^n$ 中三个 $n$ 相等。

  LIG 解决方案：用 *stack* 记录计数信息。

  策略选择：
  1. 两端向中间：先生成首尾（如 $a$ 和 $d$），再生成中间（如 $b$ 和 $c$）
  2. 左向右：先生成前半部分，stack 记录信息，再生成后半部分
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #cbox(title: "例题：$a^n b^n c^n d^n$")[
      *策略*：两端向中间。先生成 $a...d$，再生成 $b...c$。

      *Rules*：
      $
          S[sigma] & arrow a S[f sigma] d quad "(push f, 生成 a 和 d)" \
          S[sigma] & arrow T[sigma] quad "(转换，开始生成 b 和 c)" \
        T[f sigma] & arrow b T[sigma] c quad "(pop f, 生成 b 和 c)" \
               T[] & arrow epsilon quad "(stack 空，结束)"
      $

      *验证*：生成 $a a b b c c d d$（$n=2$）：
      - $S[] arrow a S[f] d arrow a a S[f f] d d arrow a a T[f f] d d$
      - $arrow a a b T[f] c d d arrow a a b b T[] c c d d arrow a a b b c c d d$ ✓
    ]
  ],
  [#cbox(title: "例题：$w h(w)$ where $w in Sigma^*$")[
    *策略*：左向右。生成 $w$ 时记录每个 symbol，再用 stack 生成 $h(w)$。

    *Rules*：
    $
        S[sigma] & arrow a S[a sigma] | b S[b sigma] | ... quad "(生成 w，push symbols)" \
        S[sigma] & arrow T[sigma] quad "(转换)" \
      T[a sigma] & arrow h(a) T[sigma] quad "(pop a, 输出 h(a))" \
      T[b sigma] & arrow h(b) T[sigma] quad "(pop b, 输出 h(b))" \
             T[] & arrow epsilon quad "(stack 空)"
    $
  ]],
)



=== CCG 推导练习

#grid(
  columns: (1fr, 2fr),
  gutter: 1em,
  [#warning[
      *TA 建议*："Start from basic intuition... 'every dog' is a phrase, 'likes every dog' is a phrase." FOC要注意$not$的优先级.
    ]

    #cbox(title: "CCG 推导步骤")[
      + Lexicon assignment：为每个 word 分配 categories
      + 从语义直觉出发：哪些 words 应该先组合？
      + 应用 combinatory rules：Forward ($>$) 或 Backward ($<$)
      + 同步计算 semantics：每步 apply Lambda terms
    ]

    #note[
      *Type raising 的作用*：将 NP 提升为 $T slash (T backslash "NP")$，使得 proper noun 可以"主动"与 verb phrase 组合。在 "Alex" 的语义中用 $lambda P. P("Alex")$ 体现。
    ]
  ],
  [
    #cbox(title: "例题：'Alex likes every dog'")[
      *Lexicon*：
      - Alex : NP : $"Alex"$
      - likes : $(S backslash "NP") slash "NP"$ : $lambda P. lambda Q. Q(lambda x. P(lambda y. "Likes"(x, y)))$
      - every : $"NP" slash "N"$ : $lambda P. lambda Q. forall x. P(x) arrow Q(x)$
      - dog : N : $"Dog"$

      *Derivation*：
      ```
      every         dog
      NP/N:λP.λQ.∀x.P(x)→Q(x)    N:Dog
      ─────────────────────────────────── >
               NP:λQ.∀x.Dog(x)→Q(x)

      likes                    [every dog]
      (S\NP)/NP:...            NP:λQ.∀x.Dog(x)→Q(x)
      ──────────────────────────────────────────────── >
              S\NP:λQ.Q(λx.∀y.Dog(y)→Likes(x,y))

      Alex        [likes every dog]
      NP:Alex     S\NP:...
      ──────────────────────────────────── <
           S:∀x.Dog(x)→Likes(Alex,x)
      ```

      *最终语义*：$forall x. "Dog"(x) arrow "Likes"("Alex", x)$

      "对所有 $x$，如果 $x$ 是狗，则 Alex 喜欢 $x$"]
  ],
)



=== Termination 与 Turing Completeness

Lambda calculus 的 Turing completeness 来源于：$beta$-reduction 可能*不终止*。

#cbox(title: "Non-terminating Example")[
  令 $Omega = (lambda x. x x)(lambda x. x x) quad$
  $arrow_beta (lambda x. x x)(lambda x. x x) = Omega$

  无限循环！这是 Russell's paradox 在 Lambda calculus 中的体现。
]

#note[
  *Undecidable problem:* 判断两个 Lambda terms 是否 equivalent（即能否通过 $alpha$/$beta$ 互相到达）是不可判定的。
]

=== Extended Lambda Calculus

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #cbox(title: "NL semantics")[
      *Constants:* 表示 entities（Alex, Bob, Texas, ...）
      *Predicates:* 表示 relations（$"Likes"(dot, dot)$, $"Person"(dot)$, ...）
      *Quantifiers:* $forall, exists$
      *Logical connectives:* $and, or, not, arrow$
    ]
  ],
  [
    α-conversion 和 $beta$-reduction 规则不变。

    #cbox(title: "Semantic Composition Example")[
      Lexicon: Alex : $"Alex"quad$; Brit : $"Brit"quad$; likes : $lambda y. lambda x. "Likes"(x, y)$

      Derivation of "Alex likes Brit":
      1. $"likes"("Brit") = (lambda y. lambda x. "Likes"(x,y))("Brit") arrow_beta lambda x. "Likes"(x, "Brit")$
      2. $(lambda x. "Likes"(x, "Brit"))("Alex") arrow_beta "Likes"("Alex", "Brit")$
    ]

  ],
)
#note[
  *为何 likes 是 $lambda y. lambda x$ 而非 $lambda x. lambda y$？*

  因为英语语序是 Subject-Verb-Object。Verb 先接 object（右边），再接 subject（左边）。Lambda 的参数顺序反映了 syntactic composition 的顺序。
]


== Combinatory Logic
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    Combinatory logic (Curry, 1958) 是 Lambda calculus 的替代——不使用 abstraction，只用 *primitive combinators* 构建 functions。

    #note[
      $bold(S)$ 和 $bold(K)$ 构成 *complete basis*——任何 Lambda term 都可用 $bold(S)$, $bold(K)$ 表示。例如 $bold(I) = bold(S) bold(K) bold(K)$。
    ]
  ],
  [
    #definition(title: "Combinators")[
      *Identity:* $bold(I) x = x quad$; *Constant:* $bold(K) x y = x quad$; *Substitution:* $bold(S) x y z = x z (y z)$

      Convention: left-associative，即 $bold(K) x y = (bold(K) x) y$
    ]

    其他常用 combinators：
    *Composition:* $bold(B) x y z = x (y z)$;

    *Flip:* $bold(C) x y z = x z y$; $quad$ *Type-raising:* $bold(T) x y = y x$
  ],
)




== Combinatory Categorial Grammar (CCG)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 为何需要 CCG？

    Context-free grammars 无法优雅处理某些 phenomena：

    1. *Coordination with gapping:*
      "I like to play bridge and Sarah handball"

    2. *Cross-serial dependencies:*
      Dutch/Swiss German 的 verb-object 交叉依赖（recall Lecture 1）

    CCG 是 *mildly context-sensitive*——比 CFG 更 expressive，但仍 polynomial-time parsable。

    更重要的是：CCG 提供了 *syntax-semantics interface*——将 Lambda calculus 优雅集成到 grammar 中。
  ],
  [
    === Linear Indexed Grammars（热身）

    #definition(title: "Linear Indexed Grammar")[
      类似 CFG，但 non-terminals 可带 *stack*，且 stack 只能传给*one* child：
      $
          N[sigma] & arrow alpha M[sigma] beta \
          N[sigma] & arrow alpha M[f sigma] beta quad "(Push)" \
        N[f sigma] & arrow alpha M[sigma] beta quad "(Pop)"
      $
    ]

    LIG 可生成 ${a^n b^n c^n | n in NN}$——CFG 无法做到。

    直觉：CFG 等价于 pushdown automata（无限 states via stack）。LIG 进一步扩展了这种"controlled infinity"。
  ],
)

=== CCG 形式定义

#definition(title: "Combinatory Categorial Grammar")[
  五元组 $chevron.l V_T, V_N, S, f, R chevron.r$：
  - $V_T$: terminals（词汇）
  - $V_N$: atomic categories（基本范畴，如 S, NP, N）
  - $S in V_N$: start category
  - $f: V_T union {epsilon} -> cal(P)(C(V_N))$: lexicon（词 $->$ categories 集合）
  - $R$: combinatory rules

  $C(V_N)$ 是 *categories* 的无限集：
  - $V_N subset C(V_N)$
  - 若 $c_1, c_2 in C(V_N)$，则 $c_1 \/ c_2, c_1 backslash c_2 in C(V_N)$
]

=== Categories 与 Type-Theoretic View

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    Category 编码 *argument structure*：
    - $S backslash "NP"$: 左边要 NP → S (intransitive)
    - $(S backslash "NP") \/ "NP"$: 右边要 NP → $S backslash "NP"$ (transitive)
  ],
  [
    #cbox(title: "Example Lexicon")[
      Mary, John : NP; walks : $S backslash "NP"$ ;

      likes : $(S backslash "NP") \/ "NP"$

    ]
  ],
)

=== Combinatory Rules

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [
    *Application*
    $
             X \/ Y quad Y & arrow.double X #h(0.5em) (>) \
      Y quad X backslash Y & arrow.double X #h(0.5em) (<)
    $
  ],
  [
    *Composition*
    $
                    X \/ Y quad Y \/ Z & arrow.double X \/ Z #h(0.3em) (bold(B)_>) \
      Y backslash Z quad X backslash Y & arrow.double X backslash Z #h(0.3em) (bold(B)_<)
    $
  ],
  [
    *Type-raising*
    $
      X & arrow.double T \/ (T backslash X) #h(0.3em) (bold(T)_>) \
      X & arrow.double T backslash (T \/ X) #h(0.3em) (bold(T)_<)
    $
  ],
)

#note[
  Rules 是 *schematic*——适用于所有 matching categories。这是 CCG 的设计哲学：rules 是 universal，language-specific 信息全在 lexicon。
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Syntax-Semantics Integration

    CCG 的优雅之处：category 的 argument structure 直接对应 Lambda term 的 type！

    #cbox(title: "Derivation with Semantics")[
      Lexicon:
      - Mary : NP : $"Mary"$
      - likes : $(S backslash "NP") \/ "NP"$ : $lambda y. lambda x. "Likes"(x, y)$
      - John : NP : $"John"$

      Parse "Mary likes John":
      ```
      Mary          likes                        John
      NP:Mary   (S\NP)/NP:λy.λx.Likes(x,y)      NP:John
                 ───────────────────────────────────── >
                            S\NP:λx.Likes(x,John)
      ──────────────────────────────────────────────── <
                        S:Likes(Mary,John)
      ```
    ]
  ],
  [
    === Practical Application: Semantic Parsing to SQL

    给定 question "What states border Texas?"，CCG parse 可得：
    $ lambda x. "State"(x) and "Borders"(x, "Texas") $


    ```sql
    SELECT x FROM states WHERE borders(x, 'Texas') #一步之遥即为 SQL
    #同样适用于 robot commands、database queries、code generation 等。
    ```

    #note[
      *与 LLM 的关系*：现代 LLM 可以直接生成 SQL/code，但无法 *guarantee* syntactic validity。CCG 等 grammar-based methods 提供 formal guarantees——在 safety-critical 应用中仍有价值。
    ]

    [Bonus]:CCG parsing 可在 $O(N^6)$ 完成（类似 CKY，但因 composition/type-raising 导致更高 complexity）。

    #tip[
      *不考*：CCG parsing algorithm 细节不在考试范围。但理解 CCG 如何集成 syntax 和 semantics 是重要的 conceptual point。
    ]
  ],
)



= Transformer

== Machine Translation: 问题定义

*Task*：给定 source language 句子 $bold(x) = (x_1, ..., x_M)$，生成 target language 句子 $bold(y) = (y_1, ..., y_N)$。

#definition(title: "MT as Optimization")[
  $ bold(y)^* = argmax_(bold(y) in cal(Y)) "score"(bold(y) | bold(x)) $

  目标：学习 $p(bold(y) | bold(x))$，从 input space 映射到 output space。特殊 token：#bos 和 #eos 标记序列边界。
]

#note[
  MT 不是 one-to-one mapping！同一句子可有多种合法翻译。例如 "The cat is black" → Spanish 需决定 grammatical gender（el gato / la gata），而英语不携带此信息——model需hallucinate额外信息。
]

*历史演进*：
- *Rule-based*（1950s）：手工词典 + 语法规则
- *Statistical MT*（1990s）：多阶段 pipeline（morphology → alignment → generation），每步独立建模
- *Neural MT*（2014+）：*end-to-end* 单model，直接输出 $p(bold(y)|bold(x))$ ← 今日主题

== Sequence-to-Sequence Models

#definition(title: "Encoder-Decoder Architecture")[
  1. *Encoder*：$bold(z) = "Encode"(bold(x))$，将输入压缩为 representation
  2. *Decoder*：$p(bold(y)|bold(x)) = product_(n=1)^N p(y_n | y_1, ..., y_(n-1), bold(z))$

  训练：MLE + backpropagation。
]

#note[
  *Information Bottleneck*：无论输入长度如何，encoder 输出 *fixed-length* vector $bold(z)$。如3词句子与100词句子被压缩到同维向量——信息损失不可避免。
]

== Attention Mechanism



=== 动机与直觉

Attention 解决 fixed-length bottleneck：允许 decoder 在每一步 *动态关注* 输入的不同部分，不仅从单一压缩向量 $bold(z)$ 解码，而是每步动态查询 encoder 全部hidden states.
核心隐喻：*Soft Hash Table*。

=== 从 Hard 到 Soft
#cbox(title: "Hash Table → Attention 的演进")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      *Step 1*: Hard Hash Table"
      $ V = "lookup"(K, "query") = cases(v_i & "if" k_i = "query", "null" & "otherwise") $
      问题：discrete lookup，不可微。仍是 hard retrieval。


      *Step 2*: Algebraic View
      用 one-hot vector $bold(alpha) in {0,1}^n$（仅一个位置为 1）检索：
      $ bold(c) = bold(alpha)^top bold(V) = sum_i alpha_i bold(v)_i $


    ],

    [
      *Step 3*: Soft Attention
      用prob分布$bold(alpha) in Delta^(|K|)$替代one-hot:
      $
        alpha_i &= (exp score(bold(q), bold(k)_i)) / (sum_j exp score(bold(q), bold(k)_j)) = softmax(score(bold(q), bold(K)))_i \
        bold(c) &= sum_(i=1)^m alpha_i bold(v)_i = bold(alpha)^top bold(V) quad "(context vector, weighted average)"
      $

      现在 $bold(alpha)$ 表示"关注度分布", continuous, differentiable; $alpha_i >= 0, sum_i alpha_i = 1, bold(c) = sum_i alpha_i bold(v)_i$.
    ],
  )
]



=== Math Formulation
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,

  [

    #definition(title: "Attention Mechanism")[
      给定：
      - *Query* $bold(q) in RR^(d_q)$：当前 decoder 状态（"我在找什么"）
      - *Keys* $bold(K) in RR^(n times d_k)$：候选位置的表示（"有什么可选"）
      - *Values* $bold(V) in RR^(n times d_v)$：实际要检索的信息
      $
        alpha_i = "softmax"_i ("score"(bold(q), bold(k)_i)) = (exp "score"(bold(q), bold(k)_i))/(sum_j exp "score"(bold(q), bold(k)_j))
      $
      $ bold(c) = sum_i alpha_i bold(v)_i = bold(alpha)^top bold(V) quad "(context vector)" $
    ]

    *Scoring Functions*（实践中常用）：
    - Dot-product: $bold(q)^top bold(k)$，最常用，高效
    - Scaled dot-product: $bold(q)^top bold(k) \/ sqrt(d_k)$，Transformer 默认#footnote[
        Scaled dot-product 中除以 $sqrt(d_k)$：防止 $d_k$ 很大时 dot product 值过大，导致 softmax 梯度消失（进入饱和区）。Dot product 的 variance 与 dimension 成正比。若不 normalize，当 $d_k$ 很大时，softmax 输入值过大，梯度saturation趋近于0
      ]

    - Additive: $bold(w)^top tanh(bold(W)_q bold(q) + bold(W)_k bold(k))$，Bahdanau 2015

    #algorithm(title: "Encoder-Decoder Attention Flow")[
      ```python
      # Encoder 阶段
      K, V = Encoder(x)  # shape: (m, d_model)

      # Decoder 逐步生成
      for t in 1..n:
          q = decoder_hidden[t]     # query: (d_model,)
          scores = score(q, K)      # (m,)
          α = softmax(scores)       # attention weights
          c = weighted_sum(α, V)    # context: (d_model,)

          p_t = softmax(FFN([c; q])) # 融合context生成prob
          y_t ~ p_t
      ```
    ]],
  [
    === Self-Attention Complexity
    Self-Attention Complexity & Comparison with N-gram:
    #figure(table(
      columns: (auto, auto, auto),
      column-gutter: 0em,
      align: (center, center, center),
      stroke: 0.2pt,

      // 表头分组线
      [Operation / Aspect], [Self-Attention], [N-gram],
      // 分隔线

      // Complexity 部分
      [*Complexity Analysis*], [], [],
      [Sequence length], [$N$], [$N$],
      [Hidden dimension], [$H$], [---],
      [Compute $bold(Q), bold(K), bold(V)$], [$O(N H^2)$], [---],
      [$bold(Q) bold(K)^top$], [$O(N^2 H)$], [---],
      [Softmax], [$O(N^2)$], [---],
      [Multiply $bold(V)$], [$O(N^2 H)$], [---],
      [Total],
      [*$O(N^2 H + N H^2)$*
        #footnote[不难发现当 $N >> H$ 时，$O(N^2)$ 是主要瓶颈。]],
      [---],

      [Bottleneck ($N >> H$)], [$O(N^2)$], [---],
    ))
    #figure(table(
      columns: (auto, auto, auto),
      column-gutter: 0em,
      align: (center, center, center),
      stroke: 0.2pt,
      // N-gram 对比部分
      [*Qualitative对比*], [Self-Attention], [N-gram],
      [Context], [Full sequence], [Fixed window $k$],
      [Parameters], [Fixed], [Depends on vocab size],
      [Small dataset], [Prone to overfit], [Works with smoothing],
      [Large dataset], [Stronger], [Limited by sparsity],
      [Runtime], [*$O(N^2)$*], [*$O(N)$*],
    ))
  ],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,

  [
    === Multi-Head Attention

    #definition(title: "Multi-Head Attention")[
      并行运行 $h$ 个 attention（各有独立参数），拼接后：

      $
        "MultiHead"(bold(Q), bold(K), bold(V)) = "Concat"("head"_1, ..., "head"_h) bold(W)^O
        \
        "head"_i = "Attention"(bold(Q) bold(W)_i^Q, bold(K) bold(W)_i^K, bold(V) bold(W)_i^V)
      $
    ]


    #note[
      不同 head 可能关注句法/语义/位置等不同模式; 增加model容量，类似 CNN 多通道; 经验上 $h = 6$ 或 $8$ 效果好。
    ]

    #tip[
      Assignment 6: 证明multi-head self-attention可以表示任意conv层。#footnote[
        Theorem: 若 multi-head self-attention 有 $K^2$ 个 heads（$K$ 是 kernel size），且使用特定 Gaussian positional encoding，则可精确表示任意 $K times K$ 卷积。这解释了ViT的成功——Transformer 至少和 CNN 一样 expressive，且更 general。
      ]
    ]
  ],
  [
    === Encoder-Decoder Attention

    #example(title: "MT 中的 Attention")[
      - $bold(K) = bold(V)$：encoder 各位置的 hidden states $bold(h)_1^("enc"), ..., bold(h)_M^("enc")$
      - $bold(Q)$：decoder 当前 hidden state $bold(h)_n^("dec")$

      语义：decoder 在生成第 $n$ 个词时，询问"源句中哪些词与当前生成最相关？"
    ]

    === Self-Attention

    #definition(title: "Self-Attention")[
      $bold(Q), bold(K), bold(V)$ 均来自 *同一序列* 的不同 linear projections：
      $ bold(Q) = bold(X) bold(W)^Q, quad bold(K) = bold(X) bold(W)^K, quad bold(V) = bold(X) bold(W)^V $

      每个位置的表示由 *整个序列* 加权得到，捕获 long-range dependencies。
    ]
  ],
)



== Trsf Architecture

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,

  [
    Self-attention的输出为contextual embeddings —— 每个 token 的表示包含了序列中其他 tokens 的信息。
    Self-attention 的意义：
    - *Encoder-only*（如 BERT）：双向 self-attention
    - *Decoder-only*（如 GPT）：causal self-attention（只看过去）
    - 不再需要 encoder-decoder 架构！
  ],
  [
    #note[Transformer 的核心贡献，关键不在精度提升，而是：*parallelization*。RNN 必须顺序处理，Transformer 可并行处理整个序列——这是 scaling 的基础。

      *训练并行 vs. 推理串行*：训练时 teacher forcing 可并行计算全序列，但生成时仍需逐词sampling（autoregressive bottleneck）]
  ],
  [
    #figure(
      table(
        columns: (auto, auto),
        align: (center, center),
        stroke: .65pt,
        table.hline(stroke: 0.5pt),
        table.header([Component], [Complexity]),
        [Encoder ($L$ layers)], [$O(L N D^2)$],
        [Decoder ($1$ layer, no attention)], [$O(M D^2)$],
        [Cross-attention], [$O(M N D)$],

        [Total with attention], [$O(L N D^2 + M N D)$],
      ),
      caption: [设encoder 长度 $N$，decoder 长度 $M$，hidden dim $D$，encoder 层数 $L$],
    )
  ],
)




=== 整体结构
Transformer Encoder Block 图略

=== 关键组件

*1. Positional Encoding*

Self-attention 是 permutation-invariant
#footnote[
  Permutation Equivariance:  若 $f$ 是 permutation equivariant，则对任意 permutation $pi$, $ f(pi(X)) = pi(f(X)) $, 即打乱输入顺序，输出以相同方式打乱。

  设 $bold(P)$ 是 permutation matrix, 具体证明利用了 softmax 对 row-wise 操作的性质和 $bold(P)^top bold(P) = bold(I)$。 则：
  $
    "Attention"(bold(P X)) &= "softmax"( 1/ sqrt(d) (bold(P X) bold(W)_Q)(bold(P X) bold(W)_K)^top) (bold(P X) bold(W)_V) \
    &= "softmax"(bold(P) bold(Q) bold(K)^top bold(P)^top/ sqrt(d)) bold(P) bold(V) = bold(P) "softmax"(bold(Q) bold(K)^top / sqrt(d)) bold(V) = bold(P) "Attention"(bold(X))
  $

  若$bold(Q)$ fixed（如常数矩阵），则 attention 变成permutation invariant——输出完全不依赖输入顺序。
]
——丢失位置信息hence需positional embedding。

#definition(title: "Sinusoidal Positional Encoding")[
  $ "PE"_("pos", 2i) = sin("pos" \/ 10000^(2i\/d)) $
  $ "PE"_("pos", 2i+1) = cos("pos" \/ 10000^(2i\/d)) $

  性质：bounded in $[-1, 1]$，远距离衰减，相对位置可通过线性变换表示。
]

这是 engineering hack——没有理论证明为何 sine/cosine 最优。
//类比：1950s 航空业已能造飞机，但不完全理解空气动力学。我们 deploy 不完全理解的系统。

*2. Residual Connection*:

每个 sub-layer 输出：$bold(x)_(l+1) = bold(x)_l + "SubLayer"(bold(x)_l)$
作用：缓解 vanishing gradient，允许更深网络。信息可 bypass 中间层直接传递(允许梯度直接回传)。

*3. Layer Normalization*:

$ "LayerNorm"(bold(x)) = gamma dot.circle (bold(x) - mu) / sigma + beta $
其中 $mu, sigma$ 是单个样本内 #footnote[
  注意各种norm的区别：
  Layer Norm: 在特征维度上归一化（更适合序列data）; Batch Norm: 在 batch 维度上归一化
]
hidden states 的均值/标准差。作用：稳定训练(缓解梯度消失)，加速收敛。



=== Encoder vs Decoder vs Encoder-Decoder

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,

  [
    #algorithm(title: "Transformer Encoder Layer")[
      ```python
          #
          def encoder_layer(x):
          # 1. Multi-head self-attention
          attn_out = MultiHeadAttention(Q=x, K=x, V=x)
          x = LayerNorm(x + attn_out)  # residual + norm

          # 2. Feed-forward network
          ffn_out = FFN(x)  # 2-layer MLP: ReLU(xW₁)W₂
          x = LayerNorm(x + ffn_out)

          return x
      ```
    ]堆叠N=6层（原论文）。Decoder 类似，但增加 encoder-decoder attention。
  ],
  [
    #cbox(title: "完整 Transformer 架构")[
      *Encoder*:
      1. Input Embedding + Positional Encoding
      2. $times N$ layers:
        - Multi-Head Self-Attention
        - Add & Norm
        - Feed-Forward Network (FFN)
        - Add & Norm

      *Decoder*:
      1. Output Embedding + Positional Encoding
      2. $times N$ layers:
        - Masked Multi-Head Self-Attention（causal）
        - Add & Norm
        - Encoder-Decoder Attention（$bold(Q)$ from decoder，$bold(K),bold(V)$ from encoder）
        - Add & Norm
        - FFN + Add & Norm
      3. Linear + Softmax → $P(y_t | y_(< t), bold(x))$
    ]
  ],
)



// #table(
//   columns: (1fr, 2fr, 2fr),
//   align: (left, left, left),
//   stroke: 0.5pt + gray,
//   inset: 8pt,
//   [*架构*], [*Attention 类型*], [*代表model*],
//   [Encoder-only], [Bidirectional self-attention], [BERT],
//   [Decoder-only], [Causal self-attention（masked）], [GPT, LLaMA],
//   [Encoder-Decoder], [Encoder self-attn + cross-attn + decoder self-attn], [T5, 原版 Transformer],
// )

== Decoding: 如何生成文本

=== 问题：指数爆炸

设 vocabulary size $|cal(V)| = 30000$，最大长度 $N = 20$：$|cal(Y)| = |cal(V)|^N = 30000^20 > "宇宙粒子数"$无法穷举！Dynamic programming 也不适用（无 Markovian structure, Viterbi $O(n|cal(V)|^k)$不适用）。

Transformer 的 scoring function 考虑 *entire context*有global依赖性，不满足local分解假设 → search空间/search图是 *tree* 而非 DAG，状态不合并 → 指数复杂度 → 考虑heuristic剪枝

=== Decoding 策略

#cbox(title: "Decoding Strategies 对照")[
  #table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    stroke: 0.5pt + gray,
    inset: 6pt,
    [*类型*], [*方法*], [*特点*],
    [Deterministic], [Greedy search], [每步取 $argmax$，快但 suboptimal],
    [Deterministic], [Beam search], [保留 top-$K$ candidates，trade-off],
    [Stochastic], [Top-$k$ sampling], [从前 $k$ 高prob词中随机sampling],
    [Stochastic], [Nucleus (top-$p$)], [从累积prob达 $p$ 的词中sampling],
  )
]

常见问题：greedy 导致重复，高温sampling产生乱码。

*随机sampling族*（stochastic decoding）：

e.g. Nucleus Sampling (Top-p)
从累积prob $>= p$ 的最小词集sampling：
$ V^((p)) = "argmin" {V' subset V : sum_(w in V') P(w | bold(y)_(< t), bold(x)) >= p} $

动态调整候选集大小。常用 $p=0.9$。

*Beam Search*：Pruned BFS，每步保留 $K$ 个最高分 partial sequences。
- $K = 1$ 退化为 greedt; $K$ 越大越接近 exact search，但计算量 $O(K dot |cal(V)|)$
- 实践中 $K = 4 tilde.op 10$ 常用。无 formal guarantee 但 works well。

#algorithm(title: "Beam Search")[
  维护 top-$K$ 候选路径（beam width = $K$）：
  ```python
  beams = [(score=0, seq=[BOS])]

  for t in 1..max_len:
      all_candidates = []
      for (s, seq) in beams:
          probs = model.predict(seq, x)  # P(y_t | seq, x)
          for w in top_K(probs):
              new_score = s + log(probs[w])
              all_candidates.append((new_score, seq + [w]))

      # 剪枝：只保留global top-K
      beams = sorted(all_candidates)[:K]

      # 终止条件
      if all EOS in beams: break

  return max(beams, key=score)
  ```

  复杂度：$O(n K |cal(V)|)$。权衡：$K=1$ 退化为 greedy，$K=infinity$ 穷举。
]

#note[
  *Parallelization 的限制*：training 时可并行（所有 tokens 已知），但 *inference 仍是 sequential*（autoregressive generation）——当前 LLM 推理速度瓶颈。
]



== Evaluation: BLEU Score

人工评估不可扩展 → 需自动化指标。核心困难：一句多译皆合理。

#definition(title: "BLEU Score")[
  基于 $n$-gram 精确率 + 长度惩罚：
  $
    "BLEU" & = "BP" times exp(sum_(n=1)^N w_n log p_n) \
       p_n & = (sum_("ngram") "count"_"clip"("ngram")) / (sum_("ngram") "count"("ngram")) quad "(modified precision)" \
      "BP" & = cases(
               1 & "if" c > r,
               e^(1 - r\/c) & "otherwise"
             ) quad "(brevity penalty)"
  $

  其中：$c$ = 候选长度，$r$ = 参考长度，$w_n = 1\/N$（通常 $N=4$）;\
  $p_n$：$n$-gram precision（预测中出现在 reference 的比例）; $"BP"$：brevity penalty，惩罚过短翻译


]

*Clipped Count*：防止重复词刷分
$ "count"_"clip"("the") = min("count"_"pred"("the"), max_"ref" "count"_"ref"("the")) $

#note[
  *BLEU 家族*：ROUGE（摘要）：Recall-oriented; METEOR：考虑同义词、词干; chrF：字符级 $n$-gram.
  现代趋势：NN评估（BERTScore）+ 人类评估（仍是金标准）
]

局限性：BLEU 只是 proxy metric。MT evaluation 仍是 open problem: 只看词重叠，忽略语义（"not good" vs. "bad"）; 对改写、同义替换不友好; 需要高质量参考译文


= Axes of Modelling

== 问题定义：从data到任务

核心问题：给定data $bold(X)$，学习映射 $f: bold(X) -> bold(Y)$。
*Task Characterization*（决定后续所有选择）：

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    - Classification：$bold(Y)$ 是离散 label 集合
    - Structured Prediction：$bold(Y)$ 是 exponentially large 的结构化输出（如序列、树）
  ],
  [
    - Representation Learning：学习 $bold(X)$ 的 dense embedding
    - Density Estimation：建模 $p(bold(X))$
  ],
)



#note[
  data质量 >> model复杂度。High-quality data 是 NLP 的 "magic element"
]

== model分类体系

#cbox(title: "Model Taxonomy")[
  ```
  Models
  ├── Probabilistic（建模 p(Y|X) 或 p(X,Y)）
  │   ├── Discriminative：直接建模 p(Y|X)
  │   │   └── Logistic Regression, CRF, Neural Classifiers
  │   └── Generative：建模 joint p(X,Y) = p(Y)p(X|Y)
  │       └── N-gram LM, Naive Bayes, HMM
  └── Non-Probabilistic
      ├── Learned：SVM, MLP, Perceptron
      └── Handcrafted：CFG, Rule-based systems
  ```
]

=== Probabilistic vs Non-Probabilistic

#table(
  columns: (1fr, 1fr, 1fr),
  align: (left, left, left),
  stroke: 0.5pt + gray,
  inset: 6pt,
  [*Aspect*], [*Probabilistic*], [*Non-Probabilistic*],
  [输出], [prob分布 $p(y|x)$], [决策边界 / 得分],
  [优势], [Uncertainty quantification, 理论框架成熟], [Interpretable, geometric intuition],
  [劣势], [Independence假设可能不成立], [难以估计 uncertainty],
  [典型], [Naive Bayes, LM], [SVM, rule-based],
)

=== Generative vs Discriminative
Trade-off：Generative 需更多假设但可处理 missing data；Discriminative 通常分类性能更优。
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [*Generative*：建模 $p(bold(x), y) = p(y) dot p(bold(x)|y)$，可生成新样本],
  [*Discriminative*：直接建模 $p(y|bold(x))$，专注分类边界],
)


== Structured Prediction

当 output space $|cal(Y)|$ 指数级大（如所有可能句子）时，无法为每个 $y$ 单独建模。

=== Local vs Global Normalization

#cbox(title: "Normalization 对比")[
  #table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    stroke: 0.5pt + gray,
    inset: 6pt,
    [*Type*], [*Locally Normalized*], [*Globally Normalized*],
    [定义],
    [$p(bold(y)|bold(x)) = product_n p(y_n | y_(< n), bold(x))$],
    [$p(bold(y)|bold(x)) = exp("score"(bold(y))) \/ Z(bold(x))$],

    [归一化], [每步独立归一化], [整个序列归一化],
    [代表], [N-gram, RNN LM, GPT], [CRF, EBM],
    [优点], [训练高效，teacher forcing], [无 label bias],
    [缺点], [*Label bias*：早期错误无法恢复], [计算 $Z$ 代价高],
  )
]

#definition(title: "Label Bias Problem")[
  Locally normalized model 中，若某状态转移prob集中于少数 successor，则该路径被"锁定"——即使后续 observation 强烈反对，也难以修正。
]

=== Independence Assumptions 的 Trade-off

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [有假设（如 Markov）
    - 可用 DP 进行 *exact* decoding
    - model参数少，不易 overfit
    - 可能 underfit（假设过强）
  ],
  [无假设（如 Transformer）
    - 建模能力强，捕获 long-range dependency
    - 只能 *approximate* decoding（beam search 等）
    - 易 overfit，需大量data
    - 可解释性差
  ],
)


== Loss Functions

=== 定义与性质

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [#definition(title: "Loss/Objective/Cost Function")[
      将model参数 $theta in Theta$ 映射到实数，量化model在训练data上的拟合程度。
      $ cal(L): Theta -> RR $
    ]
  ],
  [Desirable Properties：
    *Convexity*：保证收敛到 global minimum; *Differentiability*：可用 gradient-based optimization; *Computational efficiency*：快速计算; *Robustness to noise*：对异常值不敏感; *Statistical guarantees*：如 consistency, efficiency
  ],
)



=== Maximum Likelihood → Cross-Entropy Loss
#grid(
  columns: (1fr, 2fr),
  gutter: 1em,
  [MLE：$hat(theta) = argmax_theta product_i p(y_i | x_i; theta)$
    取负对数，转化为 loss：
    $ cal(L)_("CE")(theta) = -sum_i log p(y_i | x_i; theta) $],

  [#note[
    *MLE 优点*：Consistent estimator（data $arrow infinity$ 时收敛到真实参数）; Asymptotically efficient（达到 Cramér-Rao lower bound）; 低 KL divergence from true distribution
    *MLE 局限*：仅适用于 probabilistic models; 易 overfit; 要求 $p(y|x) > 0$（否则 $log 0$ 爆炸）
  ]],
)



=== 其他 Loss Functions
常用 Loss 对比:
#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, center),
  stroke: 0.5pt + gray,
  inset: 6pt,
  [*Loss*], [*Formula*], [*Convex?*],
  [0-1 Loss], [$bb(1)[hat(y) != y]$], [No],
  [Hinge (Max-margin)#footnote[
      *Hinge Loss*: 不仅要分类正确，还要 margin $>= 1$。Convex 但在 0 点不可微（可用 subgradient）。
    ]],
  [$max(0, 1 - y dot f(x))$],
  [Yes],

  [Logistic], [$log(1 + e^(-y dot f(x)))$], [Yes],
  [Exponential], [$e^(-y dot f(x))$], [Yes],
  [Cross-Entropy], [$-sum_c y_c log hat(y)_c$], [Yes],
)


== Regularization
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [目标：防止 overfitting，提升 generalization（在 unseen data 上的表现）。
    #definition(title: "Regularized Loss")[
      $ cal(L)_("reg")(theta) = cal(L)(theta) + lambda dot R(theta) $

      其中 $R(theta)$ 是 penalty term，$lambda > 0$ 控制正则化强度。
    ]

    其他正则化技术：
    - *Dropout*：训练时随机置零部分神经元
    - *Early stopping*：validation loss 不再下降时停止
    - *Weight decay*：等价于 L2 in certain optimizers
  ],
  [
    #table(
      columns: (auto, 1fr, 1fr),
      align: (left, left, left),
      stroke: 0.5pt + gray,
      inset: 6pt,
      [*Aspect*], [*L1 (Lasso)*], [*L2 (Ridge)*],
      [Penalty], [$lambda sum_j |theta_j|$], [$lambda sum_j theta_j^2$],
      [效果], [Sparse：多数 $theta_j = 0$], [Shrinkage：$theta_j$ 趋近 0 但非零],
      [Bayesian 解释], [Laplace prior], [Gaussian prior],
      [适用], [Feature selection], [防止 collinearity],
    )
  ],
)

=== L1 vs L2 Regularization

== Evaluation Metrics

#tip[
  *Loss ≠ Evaluation Metric*. Loss 用于训练优化；Evaluation metric 用于评估 trained model 在 held-out set 上的表现。两者可以相同（如 perplexity），但通常不同。
]

=== Classification Metrics

#definition(title: "Confusion Matrix 衍生指标")[
  #figure(
    table(
      columns: 4,
      align: center,
      stroke: 0.5pt + gray,
      inset: 6pt,
      [], [Predicted +], [Predicted −], [],
      [Actual +], [TP], [FN], [$P = "TP" + "FN"$],
      [Actual −], [FP], [TN], [$N = "FP" + "TN"$],
    ),
  )

  $
    "Precision" = "TP" / ("TP" + "FP"), quad "Recall" = "TP" / ("TP" + "FN") quad, "F"_1 = 2 dot ("Precision" dot "Recall") / ("Precision" + "Recall") quad "(harmonic mean)"#footnote[F1 是 NLP 标准：平衡 precision 和 recall。]
  $
]

#tip[
  *为何不直接用 Accuracy?*
  - Class imbalance：99% negative 时，"always predict negative" 得 99% accuracy
  - 不同错误代价不同：cancer 漏诊 (FN) 比误诊 (FP) 严重得多
]

=== Intrinsic vs Extrinsic Evaluation

#grid(
  columns: (2fr, 1fr),
  gutter: 1em,
  [#table(
      columns: (auto, 1fr, 1fr),
      align: (left, left, left),
      stroke: 0.5pt + gray,
      inset: 6pt,
      [*Type*], [*Intrinsic*], [*Extrinsic*],
      [定义], [Task-neutral，评估model本身], [评估model在下游任务的表现],
      [LM 例子], [Perplexity, likelihood], [MT BLEU, QA accuracy],
      [Embedding 例子], [Word analogy, similarity], [下游分类性能],
      [优点], [快速、可复现], [更贴近实际应用],
      [缺点], [可能与实际性能脱节], [昂贵、任务依赖],
    )
  ],
  [ NLP 评估的挑战

    主观性：语言风格因人而异（表情符号、句末标点的"攻击性"）; 多正确答案：同一问题可有多个合理回答; Human evaluation 昂贵：且 annotator diversity 难保证; Alignment 问题：model价值观与人类期望的对齐; Data contamination：benchmark 泄露到 training data
  ],
)



== Model Selection

=== 为何需要 Model Selection

*Hyperparameters*（如 learning rate, hidden size, regularization $lambda$）显著影响model性能，但不能通过 training 优化——需单独选择。

Two目标常冲突：
+ *Inference*：选择最能 explain data的model（interpretability）#footnote[
    例：银行信贷必须使用 interpretable model（如 logistic regression），即使 neural network 准确率更高——因法规要求解释拒贷原因。
  ]
+ *Prediction*：选择predict性能最佳的model

=== Cross-Validation

#definition(title: "K-Fold Cross-Validation")[
  1. 将数据随机分为 $K$ 份（folds）。
  2. *Iteration*: 循环 $K$ 次，每次取第 $k$ 份作为 Test/Validation Set，其余 $K-1$ 份作为 Training Set。
  3. *Aggregation*: 报告 $K$ 次结果的 mean $pm$ std。
]

#tip[exm21b中：test set size： $N / K quad$; training set size: $N times (K-1)/K quad$; total model created: $K$(必须完整跑完每一折)]

用途区分：
- Model Assessment: 用于评估模型泛化能力（此时留出份为 Test Set）。
- Model Selection: 用于超参数调优（此时留出份为 Validation Set）。

Nested CV (嵌套交叉验证)：用于同时进行调参和无偏评估。
- Inner Loop: Model Selection。在训练集中再次做 CV 来寻找最佳超参数。
- Outer Loop: Model Assessment。使用 Inner Loop 选出的最佳参数，在外层 Test fold 上评估泛化误差。
- 稳定性检测: 若外层不同 fold 选出的最佳参数差异巨大，说明模型不稳定或数据过少。

=== Statistical Significance Testing

问题：Model A 比 B 好 2%，是真实差异还是随机噪声？
- Multiple Testing Problem：若比较 20 个model，$alpha = 0.05$ 时期望有 1 个 false positive。
- Bonferroni Correction：比较 $m$ 个model时，使用 $alpha' = alpha / m$。

常用*检验*：
- Parametric：paired t-test（假设正态分布）
- Non-parametric：McNemar test, permutation test（无分布假设）


=== McNemar's Test
标准 t-test 假设数据服从 normal/t 分布——textual data 通常不满足。因此 NLP 常用 *non-parametric tests*：无需指定 parametric family。
#definition(title: "McNemar's Test")[
  用于比较 *两个 classifiers* 在同一数据集上的表现。

  构造 contingency table：
  #figure(
    table(
      columns: 3,
      align: center,
      stroke: 0.5pt + gray,
      inset: 6pt,
      [], [Model B Correct], [Model B Wrong],
      [Model A Correct], [$n_(00)$], [$n_(01)$],
      [Model A Wrong], [$n_(10)$], [$n_(11)$],
    ),
  )

  *Insight*：只关注 *disagreement cells* $n_(01), n_(10)$（两模型都对/都错的样本不提供区分信息）。

  Test statistic：
  $ chi^2 = ((|n_(01) - n_(10)| - 1)^2) / (n_(01) + n_(10)) $

  $H_0$：两 classifier 性能相同。要求 $n_(01) + n_(10) >= 25$。
]



=== Permutation Test

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #definition(title: "Permutation Test")[
      检验 classifier 是否学到了有意义的 pattern（vs random chance）。

      *Algorithm*：
      1. 在原始数据上训练模型，记录 performance $P_0$
      2. Repeat $B$ 次（$B >= 1000$）：
        - 随机 permute labels（打乱 $y$ 与 $x$ 的对应）
        - 重新训练，记录 performance $P_b$
      3. p-value $approx$ tion of $P_b >= P_0$

      *直觉*：若 labels 包含信息，原始模型应显著优于 permuted versions。
    ]
  ],
  [
    #definition(title: "5×2 CV t-Test")[
      解决问题：标准 CV 中各 fold 的样本 *相互依赖*，违反 t-test 独立性假设。

      *Procedure*：
      1. 将数据 50-50 split 为 train/test
      2. 训练两个 classifiers，计算 performance difference $d$
      3. 交换 train/test，再次计算 $d'$
      4. 重复 5 次（共 10 个 difference values）
      5. 计算特殊 t-statistic（考虑 variance）

      优点：保持样本独立性，结果 *conservative*（不易 false positive）。
    ]
  ],
)

#note[
  *实践建议*：小样本时务必检查 test 的 statistical power; 多重比较必须做 *Bonferroni correction*：$alpha' = alpha / m$; 不要过度依赖 p-value，关注 *effect size*
]

=== Statistical Power 与 Type I Error

#cbox(title: "常用检验的 Type I Error 对比")[
  #table(
    columns: (1fr, auto, auto),
    align: (left, center, left),
    stroke: 0.5pt + gray,
    inset: 6pt,
    [*Test*], [*Type I Error*], [*Note*],
    [Resampled t-test], [High], [不推荐用于 NLP],
    [McNemar's test], [Low], [适合比较两个 classifiers],
    [Permutation test], [Low], [最常用，无分布假设],
    [5×2 CV t-test], [Low], [适合 CV 场景],
  )
]

== Occam's Razor in NLP

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #cbox(title: "模型选择原则")[
      *Prefer simpler models*：
      - 两个model的Loss 相近 → 选系数/参数更小的
      - 简单模型更 stable，泛化更好
      - L1/L2 regularization $=>$ enforce simplicity
      - Parameter sharing #footnote[如 word embeddings 跨位置共享] $=>$ _implicit_ regularization
    ]
  ],
  [
    #note[
      即使 NN 看似 over-parameterized，good practice：实验多种复杂度，选满足性能的最简模型
    ]
  ],
)


== Domain Adaptation (Unsupv: density ratio; Supv: feature augmentation)

Train/test 分布不同（*covariate shift*）→ 性能下降

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    *Setup*：
    - $P_("old")(x,y)$：training分布 (有 labels)
    - $P_("new")(x,y)$：test分布 (可能无 labels)

    *Goal*：学习在 $P_("new")$ 上表现好的 classifier
  ],
  [
    #example(title: "Domain Shift")[
      "Very small" 对 USB drive 是 positive，对 hotel room 是 negative——同一 feature 在不同 domain 语义相反。
    ]
  ],
)

#definition(title: "Importance Sampling")[
  当 $P_("new")$ 无 label 时，用 density ratio 重新加权训练样本：

  $ cal(L)_("new") = EE_(P_("new"))[ell(x, y)] = EE_(P_("old"))[ (P_("new")(x)) / (P_("old")(x)) dot ell(x, y)] $

  *直觉*：给与 $P_("new")$ 更相似的训练样本更高权重。
]


#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #definition(title: "Unsupervised: Importance Sampling")[
      当 $P_("new")$ 无 label 时，用 density ratio 重新加权训练样本,
      用 density ratio 重新加权：
      $ cal(L)_("new") = EE_(P_("old"))[ (P_("new")(x)) / (P_("old")(x)) dot ell(x, y)] $

      *实现*：训练 binary classifier 区分 old/new, use prob ratio as weight:
      $ w(x) = P(s=0|x) / P(s=1|x) $

      其中 $s = 1$ 表示来自 old distribution;
      _若 classifier 准确率高 → shift 严重 → 更难 adapt_
    ]
  ],
  [
    #definition(title: "Supervised: Feature Augmentation")[
      若 $P_("new")$ 有少量 labels，可共享 feature：
      $ phi_("aug")(x) = [phi_("shared"); phi_("old"); phi_("new")] $

      - Shared features: 两 domains 都激活（通用sentiment words）
      - Domain-specific features: 仅对应 domain 激活
    ]
  ],
)

== Bias and Fairness in NLP

#tip[
  *Non-exam content*，但对 responsible AI practice 至关重要。
]

=== Bias 来源

#example()[Bias 进入 NLP 系统的途径:
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    [
      *1. Labeling bias*
      annotators 偏见

      *2. Sample selection*
      数据偏向 Western, male
    ],
    [
      *3. Task design*
      如 binary gender 假设

      *4. Feature omission*
      缺信息 → 模型"猜测"
    ],
    [
      *5. Majority pattern*
      优化 avg loss，忽视 minority

      *6. Feedback loops*
      部署后放大偏差
    ],
  )
]

=== Train-Test Mismatch 视角

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [训练数据主要来自 group A], [测试时遇到 group B (_distribution shift_)], [→ 性能在 group B 显著下降],
)

=== 伦理框架

#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  stroke: 0.5pt + gray,
  inset: 6pt,
  [*Framework*], [*Core Idea*], [*NLP 应用*],
  [Consequentialism], [行为的道德性由后果决定], [评估 model deployment 的社会影响],
  [Utilitarianism], [最大化总体 utility], [权衡不同群体的 performance],
  [Deontology], [遵循规则，无论后果], [Hard constraints（如禁止生成特定内容）],
  [Social Contract], [相互尊重的隐含契约], [Fairness 作为社会规范],
)

== Debiasing Word Embeddings

=== Bolukbasi et al. (2016)：线性 Bias Subspace

#definition(title: "Gender Bias as Linear Subspace")[
  核心假设：Gender bias 在 embedding space 中是 *linear subspace*。

  Vocabulary Partition：
  - Neutral words：无固有 gender（programmer, homemaker）
  - Gendered word pairs：$(w_m, w_f)$（he/she, king/queen）

  Identify Bias Subspace：
  1. 构造 difference matrix：$bold(D) = [bold(w)_("she") - bold(w)_("he"); bold(w)_("her") - bold(w)_("him"); ...]$
  2. 对 $bold(D)$ 做 PCA，前 $k$ 个 principal components 定义 bias subspace $bold(B)$
]




#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    Debiasing Steps：

    1. Neutralize：对 neutral words，投影去除 bias 分量
      $ bold(w)'_n = bold(w)_n - "proj"_bold(B)(bold(w)_n) $
  ],
  [
    2. Equalize：确保 gendered pairs 与 neutralized words 等距
      - 将 neutral word 置于 gendered pair 的"中点"
      - 调整 gendered words s.t.到 neutral words 距离相等
  ],
)


#note[
  *Equalization 的几何*：设 neutral word 为 $bold(w)_n$，gendered pair 为 $(bold(w)_m, bold(w)_f)$。
  目标：$||bold(w)'_n - bold(w)'_m|| = ||bold(w)'_n - bold(w)'_f||$。
  通过 Pythagorean theorem 解析求解 $lambda$ 参数。
]

=== Kernel PCA Debiasing (Cotterell et al.)

*Motivation*：为何 bias 必须是 linear？

*Idea*：用 kernel trick #footnote[幽默的是，结果显示Non-linear debiasing没有显著优于linear 方法——linearity assumption 对 gender bias 足够。]将 embeddings 映射到高维 feature space $phi: RR^d -> cal(H)$，在 $cal(H)$ 中做 linear debiasing（等价于原空间的 non-linear debiasing）。

#definition(title: "Kernel Trick")[
  无需显式计算 $phi(bold(w))$，只需定义 kernel function：
  $ K(bold(w), bold(w)') = chevron.l phi(bold(w)), phi(bold(w)') chevron.r $

  常用 kernels：polynomial, RBF/Gaussian。
]


=== Word Embedding Association Test (WEAT)

#definition(title: "WEAT")[
  量化 embedding 中的 implicit association。

  给定 target sets $X, Y$（如 male/female names）和 attribute sets $A, B$（如 career/family words）：
  $ s(w, A, B) = 1/(|A|) sum_(a in A) cos(bold(w), bold(a)) - 1/(|B|) sum_(b in B) cos(bold(w), bold(b)) $
  $ "WEAT" = sum_(x in X) s(x, A, B) - sum_(y in Y) s(y, A, B) $

  Effect size 类似 Cohen's $d$。
]

=== Debiasing 的局限

#note[
  _Residual bias_：即使移除 gender subspace，gender prediction 仍可达 70-75% accuracy（vs 原始 97%）——信息仍 encoded elsewhere。

  _Quality preservation_：验证 debiased embeddings 在 SimLex-999 等 benchmark 上与 human similarity judgments 的相关性未下降。
]
