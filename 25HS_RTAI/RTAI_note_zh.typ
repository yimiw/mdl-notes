#import "../assets/tmp_nt.typ": *

#show: summary_project.with(
  title: "25HS_RTAI_Note",
  authors: ((name: ""),),
  base_size: 9pt,
  heading1_size: 1.3em,
  heading2_size: 1.2em,
  math_size: 0.95em,
  par_spacing: 0.5em,
  par_leading: 0.5em,
  primary_color: rgb("#997933"),
  secondary_color: rgb("#2E7D5A"),
  margin: (x: 1.25cm, y: 1.25cm),
)

#pagebreak()

= Part 1: Introduction <sec:intro>

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    == 课程 Verticals

    课程围绕三个研究方向展开：

    #figure(
      table(
        columns: 3,
        align: center,
        fill: (x, y) => if y == 0 { c1 },
        [*Attacks & Guarantees*], [*Privacy*], [*Provenance*],
        [Convex Relaxation], [Membership Inference], [Watermarking],
        [Certified Training], [Differential Privacy], [Benchmark Eval],
        [Randomized Smoothing], [Federated Learning], [Contamination],
      ),
      caption: [RTAI 三大方向],
    )
  ],
  [
    == 核心问题

    所有方法都围绕一个核心问题展开：

    $ forall delta in cal(B)(x): quad f(x + delta) = f(x) ? $

    其中 $cal(B)(x) = {x' mid(|) norm(x' - x)_p lt.eq epsilon}$ 是扰动集合。

    #note[
      关键张力在于 *存在量词 $exists$* vs *全称量词 $forall$*：
      - Attack：证明 $exists delta$（找反例）
      - Verification：证明 $forall delta$（性质成立）
      - Defense：构造 $f_theta$ 使 Verification 成功
    ]
  ],
)

== Robustness Verification

=== 问题定义

#definition(title: "Robustness Verification")[
  给定网络 $f$ 和输入规格#footnote[Input Specification，定义了允许的扰动范围] $Phi(x)$，验证：
  $ forall x' in Phi(x): f(x') = f(x) $

  其中 $Phi(x)$ 通常是 $ell_p$ 球：$Phi(x) = {x' mid(|) norm(x' - x)_p lt.eq epsilon}$
]

#grid(
  columns: (1.5fr, 1fr),
  gutter: 1em,
  [
    为什么难？考虑 MNIST：
    - 输入dim：784
    - 可能的扰动：$2^(784)$ 种（穷举不可行）
    - 即使是 $ell_infinity$ 球，内部点数也趋近无穷

    解决思路：不枚举每个点，而是*把整个凸形状推过网络*。
  ],
  [
    ```
    Input Space    →    Output Space
    ┌──────┐
    │ ●●●● │  ──f──→   Decision Region
    │ ●●●● │
    └──────┘
      ε-ball           全在同一类？
    ```
  ],
)

=== Certification 方法对比

两类主要方法提供不同类型的保证：

#figure(
  table(
    columns: 3,
    align: left,
    [], [*Convex Methods*], [*Randomized Smoothing*],
    [原理], [传播凸集合通过网络], [采样 + 统计保证],
    [是否需要特殊训练], [需要 Certified Training], [推理时即可使用],
    [可验证性质], [多种（robustness, fairness 等）], [有限],
    [可扩展性], [小到中型网络], [大model（包括 LLM）],
    [保证类型], [确定性], [prob性],
  ),
  caption: [两类 Certification 方法对比],
)

#note[
  选择哪种方法取决于应用场景：
  - 需要确定性保证：Convex Methods
  - 需要扩展到大model：Randomized Smoothing
  - 需要训练时优化：Certified Training
]

== Min-Max 优化框架

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Standard Training

    $ min_theta bb(E)_((x,y) tilde cal(D)) [cal(L)(f_theta (x), y)] $

    只关心单点 $x$ 的准确率。
  ],
  [
    === Robust Training

    $ min_theta bb(E)_((x,y) tilde cal(D)) [max_(x' in Phi(x)) cal(L)(f_theta (x'), y)] $

    关心整个 $Phi(x)$ 内的 worst-case。
  ],
)

#definition(title: "Min-Max 双层优化")[
  Robust Training 是嵌套优化问题：
  - *Inner max*：在 $Phi(x)$ 内找 worst-case 扰动（攻击者视角）
  - *Outer min*：优化model参数 $theta$ 抵御最坏情况（防御者视角）

  两者形成对抗博弈。
]

这个框架统一了三节课的内容：

#figure(
  table(
    columns: 4,
    align: left,
    [*任务*], [*Inner Max*], [*Outer Min*], [*方法*],
    [Attack], [找 worst-case $delta^*$], [—（$theta$ 固定）], [FGSM/PGD/C&W],
    [Adversarial Training], [PGD 生成对抗sample], [SGD 更新权重], [PGD-AT],
    [Certified Training], [Convex Relaxation 推区域], [优化 certified loss], [IBP/CROWN],
  ),
  caption: [Min-Max 框架的三种实例化],
)

== Robustness ≈ Individual Fairness

#theorem(title: "技术等价性")[
  Robustness 和 Individual Fairness 在数学上等价，区别仅在距离度量 $d$ 的定义：

  *Robustness*：$forall x': d(x, x') lt.eq epsilon arrow.r.double f(x') = f(x)$

  *Individual Fairness*：$forall x': d_("sensitive")(x, x') lt.eq epsilon arrow.r.double f(x') = f(x)$
]

#note[
  实际意义：同一套 Convex Relaxation 技术可以用于：
  - Robustness：$d$ 是像素级 $ell_p$ 距离
  - Fairness：$d$ 只看敏感属性（如种族、性别）的差异
  - Quantization：$d$ 是量化误差范围
]

== Differential Privacy

#definition(title: "$(epsilon, delta)$-Differential Privacy")[
  算法 $M$ 满足 $(epsilon, delta)$-DP，若对所有相差一条记录的数据库 $D, D'$：
  $ P[M(D) in S] lt.eq e^epsilon dot P[M(D') in S] + delta $

  直觉：加入/移除一个人，输出分布变化很小 → 无法推断个体是否在数据中。
]

#figure(
  table(
    columns: 3,
    align: left,
    [*符号*], [*含义*], [*备注*],
    [$M$], [随机化算法], [如 DP-SGD 训练过程],
    [$D, D'$], [相差一条记录的数据库], ["邻居"dataset],
    [$epsilon$], [隐私预算], [越小越隐私],
    [$delta$], [失败prob], [通常取 $lt.double 1/n$],
  ),
  caption: [DP 符号说明],
)



= Part 2: Verification <sec:verification>

== Verification 问题形式化

#definition(title: "Formal Verification Problem")[
  $ forall i in I: phi(i) arrow.r.double.long N(i) tack.double C $

  对所有输入 $i$，若满足前条件 $phi(i)$，则网络输出 $N(i)$ 满足后条件 $C$。
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #definition(title: "Sound")[
      若方法说"成立"，则性质*确实成立*。
      $ "Proved" arrow.r.double "True" $

      宁可多说"不知道"（保守），也不能误报"安全"。
    ]
  ],
  [
    #definition(title: "Complete")[
      若性质*确实成立*，则方法*能够证明*。
      $ "True" arrow.r.double "Provable" $

      不会遗漏可证明的性质。
    ]
  ],
)

#tip[
  大多数实用方法是 *Sound but Incomplete*：
  - 说"安全"时可信
  - 说"不知道"时不代表不安全，可能只是方法能力有限
]

```
                    Property Actually Holds?
                         YES        NO
                    ┌──────────┬──────────┐
    Method says     │  ✓ OK    │ ✗ UNSOUND│
       "HOLDS"      │          │  (危险!) │
                    ├──────────┼──────────┤
    Method says     │INCOMPLETE│  ✓ OK    │
     "UNKNOWN"      │  (保守)   │          │
                    └──────────┴──────────┘
```

== Box Relaxation (IBP)

#definition(title: "Interval Bound Propagation")[
  用区间 $[l, u]$ 表示神经元可能的取值范围。区间形成 hyper-rectangle（Box）。

  $ "Box": quad [l_1, u_1] times [l_2, u_2] times dots.c times [l_n, u_n] $
]

=== Abstract Transformers

对每种运算定义在区间上的操作：

#figure(
  table(
    columns: 2,
    align: left,
    [*操作*], [*符号定义*],
    [加法], [$[a,b] plus.o [c,d] = [a+c, b+d]$],
    [取负], [$-[a,b] = [-b, -a]$],
    [标量乘], [$lambda [a,b] = cases([lambda a\, lambda b] & lambda gt.eq 0, [lambda b\, lambda a] & lambda < 0)$],
    [ReLU], [$"ReLU"([l, u]) = [max(0, l), max(0, u)]$],
  ),
  caption: [Box Abstract Transformers],
)

=== Affine 层的传播

对于 $bold(z) = W bold(x) + bold(b)$，精确计算区间：

$ [bold(l)', bold(u)'] = [W^+ bold(l) + W^- bold(u) + bold(b), quad W^+ bold(u) + W^- bold(l) + bold(b)] $

其中 $W^+ = max(W, 0)$，$W^- = min(W, 0)$。

=== Crossing ReLU

#definition(title: "Crossing ReLU")[
  若 ReLU 输入的 bounds 满足 $l < 0 < u$，则称该 ReLU 处于 *crossing* 状态。

  非 crossing 情况更简单：
  - $l gt.eq 0$：恒正，$y = x$（直接传递）
  - $u lt.eq 0$：恒负，$y = 0$（输出恒为 0）
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    ```
    Crossing ReLU (l < 0 < u):
            Y
            │      /
            │     / ← 真实 ReLU
            │    /
            │   /████ ← Over-approx 区域
            │  /█████
            │ /██████
       ─────┼─────────── X
          L │        U
    ```
  ],
  [
    Crossing ReLU 引入 *over-approximation*：
    - 真实可达集（黄色碎片）
    - 近似包络（紫色 Box）
    - 紫色 $supset.eq$ 黄色（*Sound*）
    - 额外区域称为 "garbage points"
  ],
)

#tip[
  Over-approximation 误差随深度*累积*：
  $ "Error"_k = f("Error"_(k-1), "Layer"_k) $

  深度是 bound propagation 的敌人。
]

== MILP 编码

Mixed Integer Linear Programming 通过引入二进制变量实现 *Sound & Complete* 验证。

#definition(title: "Crossing ReLU 的 MILP 编码")[
  对于 crossing ReLU（$l < 0 < u$），引入二进制变量 $a in {0, 1}$：

  $ y gt.eq x $
  $ y lt.eq x - l(1-a) $
  $ y lt.eq u dot a $
  $ y gt.eq 0 $

  - 当 $a = 1$：约束简化为 $y = x$（active branch）
  - 当 $a = 0$：约束简化为 $y = 0$（inactive branch）
]

#theorem(title: "MILP 复杂度")[
  复杂度为 $O(2^k)$，其中 $k$ 是 *Crossing ReLU 数量*，而非总神经元数。

  $ "Complexity" prop 2^(|{"Crossing ReLUs"}|) $
]

#note[
  这意味着：
  - 更 tight 的 bounds → 更少 crossing → MILP 更快
  - Box/DeepPoly 预计算 bounds 可以大幅减少 MILP 分支数
  - 论文声称"验证百万神经元"时，要检查 crossing 数量和网络准确率
]

=== 具体例子

给定：$x_1 in [0, 0.5], quad x_2 in [0.2, 0.7]$

Affine 层：$x_3 = x_1 + x_2, quad x_4 = x_1 - x_2$

Box 传播：
$ x_3 in [0 + 0.2, 0.5 + 0.7] = [0.2, 1.2] quad "（非 crossing，" l gt.eq 0 "）" $
$ x_4 in [0 - 0.7, 0.5 - 0.2] = [-0.7, 0.3] quad "（Crossing！" l < 0 < u "）" $

结论：$x_3$ 无需分支（$y = x$），$x_4$ 需要 MILP 二进制变量。

== DeepPoly Relaxation

#grid(
  columns: (1fr,1fr),
  [
    DeepPoly 是介于 Box 和 MILP 之间的方法：比 Box 精确，比 MILP 快。

#definition(title: "Linear Symbolic Bounds")[
  每个神经元 $X_j$ 维护four *constraints*：

  $ X_j gt.eq sum_i a_i^L X_i + b^L quad "(下界线性约束)" $
  $ X_j lt.eq sum_i a_i^U X_i + b^U quad "(上界线性约束)" $
  $ X_j gt.eq L_j quad "(具体下界)" $
  $ X_j lt.eq U_j quad "(具体上界)" $
]
  #note[
  为什么需要具体 bounds $L_j, U_j$？
  - 判断 ReLU 是否 crossing
  - 随时停止 back-substitution（不必回溯到输入层）
  - 计算效率：$O(1)$ 的 ReLU transformer
]
  ],
  [
    === Affine 层

对于 $bold(z) = W bold(x) + bold(b)$，DeepPoly 是 *Exact*（无损）：
$ bold(z) lt.eq W bold(x) + bold(b) lt.eq bold(z) quad "(upper = lower)" $

=== ReLU 层

对于 crossing ReLU $Y = "ReLU"(X)$，其中 $X in [l, u]$ 且 $l < 0 < u$：

*Upper bound（固定）*：
$ Y lt.eq frac(u, u - l)(X - l) = lambda X - lambda l, quad "where" lambda = frac(u, u - l) $

*Lower bound（可选/可优化）*：
$ Y gt.eq alpha X, quad "where" alpha in [0, 1] $

#tip[
  这个 $alpha$ 就是 $alpha$-CROWN 中的 $alpha$！它是可优化参数。
  - $alpha = 0$：$Y gt.eq 0$
  - $alpha = 1$：$Y gt.eq X$
  - 中间值：精度与速度的 trade-off
]
  ],
)





=== Back-Substitution

核心算法：递归展开线性约束直到输入层，获得更 tight 的 bounds。

#algorithm(title: "Back-Substitution")[
  计算神经元 $X_j$ 的具体 bounds $[L_j, U_j]$：

  1. 获取 $X_j$ 的线性上界：$X_j lt.eq sum_i c_i X_i + d$
  2. 对每个 $X_i$，用其自身的线性约束替换
  3. 递归直到到达输入层
  4. 用输入 bounds 计算最终数值
]

#tip[
  *符号反转陷阱*：计算 upper bound 时，若系数为*负*，需要用变量的 *lower* bound：

  $ "If" quad X_j lt.eq -X_i + dots, quad "then substitute" quad X_i gt.eq dots "(lower bound)" $

  原因：$-X_i$ 要最大化，需要 $X_i$ 最小化。
]



#grid(
  columns: (1fr,1fr),
  [
  === Single-Neuron vs Multi-Neuron
  #figure(
  table(
    columns: 4,
    inset: 3pt,
    align: left,
    [*类型*], [*依赖关系*], [*并行性*], [*精度*],
    [Single-Neuron], [仅依赖前一层], [完全并行（GPU 友好）], [较低],
    [Multi-Neuron], [可用同层神经元约束], [串行], [较高],
  ),
  caption: [Relaxation 类型对比],
)
DeepPoly 采用 Single-Neuron，牺牲精度换取并行性。
],
  [
    

=== Triangle Relaxation 为什么不 Scale

```
Triangle (3 个约束):          DeepPoly (2 个约束):
        Y                              Y
        │    /│                        │    /
        │   / │                        │   /███
        │  /  │                        │  /████
        │ /   │ ← Y ≥ X               │ /█████
        │/    │                        │/██████
   ─────┼─────┼─── X              ─────┼─────── X
      L │     U                      L │     U

   约束数随层数指数增长           约束数固定为 2
```
  ],
)



== $alpha$-$beta$-CROWN

#grid(
  columns: (1fr, 1fr),[
    === Lagrange Multiplier 方法

问题：朴素 split 会丢失关系信息（DeepPoly 只能维护一对约束）。

对于 split 约束 $X gt.eq 0$（即 $-X lt.eq 0$）：

$ max_X f(X) quad "s.t." quad X gt.eq 0 $

转化为 Lagrangian：

$ max_X min_(beta gt.eq 0) [f(X) + beta dot X] $

由 *Weak Duality*：

$ max_X min_beta [dots] lt.eq min_beta max_X [dots] $

右侧更好处理：
- $max_X$ 可通过 back-substitution 计算
- $min_beta$ 可用gradient下降优化
  ],
  [#definition(title: "$alpha$-$beta$-CROWN 组成")[
  - $alpha$：ReLU 下界斜率参数，$in [0, 1]$，可gradient优化
  - $beta$：Lagrange 乘子，$gt.eq 0$，用于编码 split 约束
  - CROWN：DeepPoly 框架
]

#note[
  关键性质：$alpha$ 和 $beta$ 只影响 *tightness*，不影响 *soundness*。
  - 任意 $alpha in [0, 1]$ 都是 sound 的 ReLU relaxation
  - 任意 $beta gt.eq 0$ 都给出 sound 的 upper bound
]]
)


== Floating-Point Soundness

#tip[
  很多"Sound" verifier 在浮点运算下实际是 *Unsound* 的！

  - 理论：MILP 在实数 $RR$ 上是 Sound & Complete
  - 现实：硬件用 IEEE-754 浮点数，有 rounding error

  $ "Sound"_("theory") eq.not "Sound"_("hardware") $
]

== Branch & Bound 算法

#definition(title: "Branch & Bound for NN Verification")[
  *核心思想*：结合 Relaxation（快但 incomplete）和 Splitting（慢但 complete）

  *算法流程*：
  1. 用 DeepPoly/CROWN 计算 bounds（*Bound* 阶段）
  2. 若证明成功 → 返回 SAFE
  3. 若 bounds 不够紧 → 选择一个 unstable ReLU split（*Branch* 阶段）
  4. 递归处理两个子问题（$X gt.eq 0$ 和 $X lt 0$）
]

#grid(
  columns: (1fr, 1fr),
  [=== 算法伪代码

    #algorithm(title: "Branch & Bound")[
      ```python
      def verify(spec, model, bounds):
          # 1. Bound: 尝试用 relaxation 证明
          lb, ub = compute_bounds(model, bounds)  # DeepPoly/CROWN

          if lb > 0:  # 所有输出 > 0
              return SAFE
          if ub < 0:  # 存在必然违反
              return UNSAFE (with counterexample)

          # 2. Branch: 选择最不稳定的 ReLU
          neuron = select_unstable_relu(bounds)  # 启发式选择

          # 3. 递归
          result_pos = verify(spec, model, bounds ∪ {neuron ≥ 0})
          if result_pos == UNSAFE:
              return UNSAFE

          result_neg = verify(spec, model, bounds ∪ {neuron < 0})
          return result_neg
      ```
    ]],
  [
    === Branching 启发式

    #figure(
      table(
        columns: 3,
        align: left,
        [*启发式*], [*选择标准*], [*直觉*],
        [Largest Interval], [$max(u - l)$], [区间最大的最不确定],
        [Closest to Zero], [$min(|l|, |u|)$], [最接近 0 的最关键],
        [Gradient-based], [$max |nabla_x "objective"|$], [对目标影响最大],
        [Learning-based], [神经网络预测], [从历史学习],
      ),
    )

    === 复杂度分析

    #theorem(title: "Branch & Bound 复杂度")[
      *最坏情况*：$O(2^k)$，其中 $k$ 是 unstable ReLU 数量

      *实际表现*：取决于
      - Bounds 的 tightness（越紧需要 branch 越少）
      - Branching 启发式的质量
      - 问题本身的结构

      *关键优化*：用 α-β-CROWN 在 runtime 优化 bounds，减少 branch 次数
    ]
  ],
)

== VNN-COMP 竞赛批判分析

#tip[
  *读论文时必须警惕的指标陷阱！*

  论文声称 "验证了 68,000,000 参数网络" 时，立即检查：
]

#grid(
  columns: (1fr, 1fr, 1.2fr),
  gutter: 1em,
  [
    === 需要检查的指标

    1. *Crossing ReLU 有多少？*
      - 若只有 10 个 → $2^10 = 1024$
      - 复杂度取决于 crossing 数而非总参数

    2. *网络准确率是多少？*
      - 过度正则化的网络容易验证
      - 但实际 accuracy 可能很低

    3. *用了什么 specification？*
      - 小 $epsilon$ → 更少 crossing
      - $epsilon = 0.001$ 比 $epsilon = 0.3$ 验证简单得多
  ],
  [
    === Critical Thinking

    #note[
      *Sound but Impractical*：
      - 论文可能只验证了特殊网络
      - 在 standard benchmark 上可能失败

      *Complete but Slow*：
      - MILP 理论上 complete
      - 但 timeout = 3600s 后也算"证明失败"
    ]

    $ "Verified" eq.not "Practically Robust" $
  ],
  [=== VNN-COMP 常见问题

    #figure(
      table(
        columns: 2,
        inset: 3pt,
        align: horizon,
        stroke: 0.75pt,
        [*问题*], [*警示*],
        [Network 太小], [只在 tiny network 上验证，无法泛化],
        [Epsilon 太小], [$epsilon = 2/255$ 对 ImageNet 几乎没意义],
        [Timeout 太长], [3600s 证明一个sample不实用],
        [Certified Accuracy 低], [验证成功但只有 30% certified],
      ),
    )],
)



== Part 2 易错点补充

*α 在 α-β-CROWN 中的作用*：控制 ReLU 下界斜率，$alpha in [0, 1]$，可gradient优化

*β 的物理意义*：Lagrange 乘子，编码分支约束 $X gt.eq 0$

*Weak Duality*：$max min lt.eq min max$（总是成立）

*为什么 tighter relaxation 不一定更好?*：更紧 = 更难优化 = 训练可能失败

*Certified Accuracy 的局限*：只衡量"能证明安全"的比例，不是"实际安全"的比例

*Branch & Bound 的 bottleneck*：不是神经元总数，而是 *unstable ReLU 数量*



== 三种攻击方法对比

#figure(
  table(
    columns: 5,
    align: center,
    fill: (x, y) => if y == 0 { c1 },
    [*方法*], [*步数*], [*范数约束*], [*优化目标*], [*典型用途*],
    [FGSM], [1], [$ell_infinity$（固定）], [快速启发式], [快速评估脆弱性],
    [C&W], [多步优化], [$ell_2$（最小化）], [最小扰动], [精确攻击],
    [PGD], [10-20], [$ell_infinity$（投影）], [最大化 loss], [攻击 + Adversarial Training],
  ),
  caption: [三种攻击方法对比],
)

#note[
  核心关系：$"PGD" = "FGSM" times K "迭代" + "投影"$

  C&W 代表另一种哲学：最小化扰动大小，而非固定扰动预算。
]

== Targeted vs Untargeted Attack

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Targeted Attack

    目标：使model输出*特定*错误类别 $t eq.not y$

    $ eta^* = arg min_eta norm(eta)_p quad "s.t." quad f(x + eta) = t $

    优化方向：*靠近*目标类（gradient下降）
  ],
  [
    === Untargeted Attack

    目标：使model输出*任意*错误类别

    $ eta^* = arg min_eta norm(eta)_p quad "s.t." quad f(x + eta) eq.not y $

    优化方向：*远离*正确类（gradient上升）
  ],
)

== FGSM

#definition(title: "Fast Gradient Sign Method")[
  #grid(
    columns: (1fr, 1fr),
    [*Targeted*：
      $ x' = x - epsilon dot "sign"(nabla_x cal(L)(f(x), t)) $],
    [
      *Untargeted*：
      $ x' = x + epsilon dot "sign"(nabla_x cal(L)(f(x), y)) $
    ],
  )

]

=== 为什么用 Sign 函数？

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    1. *归一化效应*：gradient各dim量级差异巨大，sign 统一为 ${-1, 0, +1}$

    2. *最大步长*：在 $ell_infinity$ 约束下，每个像素都走到 box 边界

    3. *单步最优*：一阶 Taylor 展开下，这是 $ell_infinity$ 约束的最优单步移动
  ],
  [
    ```
    gradient空间              Sign 空间
    [100, 0.01, -50]  →  [1, 1, -1]
    不同尺度          →  统一步长 ε
    连续实数向量      →  离散方向集

    几何意义：跳到 ℓ∞ 球的顶点
    ```
  ],
)

== C&W Attack

#definition(title: "Carlini & Wagner")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [*原问题*（难优化）：
      $ min_eta norm(eta)_p quad "s.t." quad f(x + eta) = t $],
    [
      *松弛后*（连续可优化）：
      $ min_eta norm(eta)_p^2 + c dot "OPS"(x + eta, t) $

      其中 $"OPS"(x', t) = max(0, max_(i eq.not t) Z(x')_i - Z(x')_t + kappa)$
    ],
  )
]


#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === OPS 函数的契约

    $ "OPS"(x', t) lt.eq 0 quad arrow.r.long quad f(x') = t $

    这是*单向蕴含*（Sound but Incomplete）：
    - OPS $lt.eq 0$ → 攻击必然成功
    - OPS $> 0$ → 不一定失败
  ],
  [
    #note[
      参数 $kappa$（margin）控制 confidence：
      - $kappa = 0$：只要分类正确即可
      - $kappa > 0$：目标类 logit 至少比其他类大 $kappa$
    ]
  ],
)



== PGD

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #algorithm(title: "Projected Gradient Descent")[
      ```python
      初始化: x₀ = x + random_in_ε_box
      for k = 1 to K:
          # 1. FGSM step
          g_k = ∇_x L(f(x_{k-1}), y)
          x'_k = x_{k-1} + α · sign(g_k)

          # 2. Projection (关键!)
          x_k = Π_{B_ε(x)}(x'_k)
      return x_K
      ```
    ]
  ],
  [
    对于 $ell_infinity$ 范数，投影操作非常简单：
    $ Pi_(cal(B)_epsilon (x))(z) = "clip"(z, x - epsilon, x + epsilon) $

    逐dim裁剪：超出范围则拉回边界。

    #tip[
      投影复杂度是关键：
      - $ell_infinity$ 球：$O(n)$（逐dim clip）
      - $ell_2$ 球：$O(n)$（归一化）
      - 复杂凸多面体：可能需要 QP solver

      这是 Certified Training 的计算瓶颈之一。
    ]

  ],
)


== Adversarial Training

#grid(
  columns: (1fr, 1fr),
  [#definition(title: "Adversarial Training (PGD-AT)")[
    $ min_theta bb(E)_((x, y) tilde cal(D)) [max_(delta in cal(B)_epsilon (0)) cal(L)(f_theta (x + delta), y)] $

    - *Inner max*：用 PGD 找 worst-case 对抗sample
    - *Outer min*：在对抗sample上做标准 SGD
  ]],
  [=== 伪代码

    ```python
    for (x, y) in train_loader:
        # Inner Max: 找最难的对抗sample
        x_adv = PGD_attack(x, model, epsilon, steps=10)

        # Outer Min: 在对抗sample上训练
        loss = CrossEntropy(model(x_adv), y)
        loss.backward()  # 对 θ 求gradient
        optimizer.step()
    ```],
)





=== Contrastive Learning 视角

#note[
  Adversarial Training 可理解为对抗性对比学习：
  - *Anchor*：原始输入 $x$
  - *Positive Bag*：$cal(B)_epsilon (x)$ 内所有点（语义应保持不变）
  - *Hard Negative*：PGD 在 $cal(B)_epsilon (x)$ 内找到的最大 loss 点

  传统对比学习采样*有限个*负sample；Adversarial Training 对*整个 $epsilon$-球*都鲁棒。
]

== TRADES：精度-鲁棒性权衡

#definition(title: "TRADES Loss")[
  $
    cal(L)_("TRADES") = underbrace(cal(L)(f(x), y), "Natural Accuracy") + lambda underbrace(max_(x' in cal(B)_epsilon) "KL"(f(x) || f(x')), "Robustness")
  $

  *核心思想*：自然准确率和鲁棒性*分开*优化，用 $lambda$ 权衡。
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === PGD-AT vs TRADES

    #figure(
      table(
        columns: 3,
        align: left,
        [], [*PGD-AT*], [*TRADES*],
        [Loss], [$max cal(L)(f(x'), y)$], [$cal(L)(f(x), y) + lambda "KL"$],
        [目标], [最小化最坏情况], [平衡精度与鲁棒性],
        [Clean Acc], [较低], [较高],
        [Robust Acc], [较高], [中等],
      ),
    )
  ],
  [
    === 选择指南

    - *需要最高鲁棒性*：使用 PGD-AT
    - *需要平衡*：使用 TRADES + 调节 $lambda$
    - $lambda$ 越大 → 越关注鲁棒性
    - 典型值：$lambda in [1, 6]$

    #note[
      实践中 TRADES 通常在 clean-robust 权衡曲线上表现更好。
    ]
  ],
)

#grid(
  columns: (1fr, 1fr),
  [== AutoAttack：可靠的攻击评估

    #definition(title: "AutoAttack Ensemble")[
      *组成*#footnote[设计思想：组合多种攻击以避免假阳性（误以为model鲁棒）]：
      1. *APGD-CE*：Auto-PGD with CE loss
      2. *APGD-DLR*：Auto-PGD with Difference of Logits Ratio loss
      3. *FAB*：Fast Adaptive Boundary attack
      4. *Square Attack*：Black-box query-based attack
    ]],
  [#tip[
    *为什么需要 AutoAttack？*
    - 单一攻击可能被"过拟合防御"绕过
    - 论文可能选择性报告弱攻击结果
    - AutoAttack 提供*标准化评估*

    *使用规则*：
    - 报告 Robust Accuracy 时必须用 AutoAttack
    - 自定义攻击结果只能作为*补充*
  ]],
)



= Part 4: Certified Training <sec:certified>

#grid(
  columns: (1fr, 1fr),
  [
    == PGD Training vs Certified Training

    #figure(
      table(
        columns: 3,
        align: left,
        [], [*PGD Training*], [*Certified Training*],
        [优化空间], [输入空间 $S(x)$], [输出空间 $gamma(f^sharp (S(x)))$],
        [Inner max], [$max_(x' in S(x)) cal(L)(f(x'), y)$], [$max_(z in gamma(f^sharp (S(x)))) cal(L)(z, y)$],
        [使用的点], [具体对抗sample], [符号区域（含 garbage points）],
        [保证类型], [Heuristic（可能 miss attacks）], [Sound（可证明保证）],
        [计算方式], [具体前向传播], [符号前向传播],
      ),
      caption: [训练范式对比],
    )

    ```
    PGD Training:                     Certified Training:

       S(x)                              S(x)
        ●                                 ●
       ╱ ╲                               ╱ ╲
      ╱   ╲     攻击空间                ╱   ╲
     ●─────●                           ●─────●
        ↓                                  ↓
        ↓  找 worst-case 输入             ↓  Convex Propagation
        ↓                                  ↓
       ●──●                            ████████  ← 输出 region
      具体输出                           (含 garbage points)
                                            ↓
                                       在这里找 worst-case
    ```
  ],
  [
    == Certified Training Paradox

    实验发现：*更 tight 的 relaxation 反而导致更差的训练结果！*

    #figure(
      table(
        columns: 3,
        align: center,
        [*Relaxation*], [*Tightness*], [*Certified Accuracy*],
        [Box (IBP)], [Low], [86%],
        [Zonotope], [Medium], [73%],
        [DeepPoly], [High], [70%],
      ),
      caption: [Tightness vs Training Performance（反直觉！）],
    )

    === 原因分析

    #grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      [
        *Sensitivity（敏感性）*：
        - DeepPoly 有 discrete switching（选 $alpha$ 时）
        - 权重小变化 → bounds 剧变
        - gradient不稳定
      ],
      [
        *Discontinuity（不连续性）*：
        - 复杂 relaxation 引入更多不连续点
        - 优化 landscape 更难 navigate
      ],
    )

    ```
    Box 的优化 landscape:          DeepPoly 的优化 landscape:

        ╲    ╱                        ╱╲  ╱╲  ╱╲
         ╲  ╱                        ╱  ╲╱  ╲╱  ╲
          ╲╱                        ╱           ╲
        smooth!                     discontinuous!
    ```

    #note[
      反直觉但重要：*Tightness $eq.not$ Optimizability*

      Box 虽然松（精度低），但gradient平滑，反而更好优化。
    ]
  ],
)



== SABR: Layer-wise Training

核心思想：在中间层做 PGD，而非在输出空间优化。

#algorithm(title: "SABR Method")[
  对于每层 $k$：
  1. 用 convex relaxation 将输入 spec 传播到第 $k$ 层
  2. 冻结第 $k$ 层之前的参数
  3. 在 intermediate shape 上做 PGD（只训练后面的层）
  4. 更新第 $k$ 层及之后的权重
]

```
Input     Layer 1    Layer 2    Layer 3    Output
  ●────────H₁────────H₂────────H₃────────●
  │                  │                    │
  │         ①        │         ②          │
  │    Propagate     │     PGD here!      │
  │    (frozen)      │    (train H₂,H₃)   │
  ↓                  ↓                    ↓
 S(x) ──convex──→ Shape ──PGD──→ worst points
```

#tip[
  投影问题：PGD 需要投影到 $S(x)$，但中间层 shape 可能不是 $ell_infinity$ 球！

  - $ell_infinity$ ball 投影：简单 clip
  - DeepPoly shape 投影：需要解 QP

  解决方案：用 Zonotope（可高效投影）。
]

== Logic → Loss Translation

#grid(
  columns: (2fr, 1fr),
  [#definition(title: "逻辑约束到loss函数")[
      任意逻辑公式 $phi$ 可翻译为loss $L_phi$，满足：
      $ L_phi (x) = 0 quad arrow.l.r.double quad x tack.double phi $
    ]
    #note[
      这提供了处理任意 safety specs 的统一框架：
      - Adversarial attack：$exists delta: norm(delta)_infinity lt.eq epsilon and arg max f(x + delta) eq.not y$
      - Robustness verification：$forall delta: norm(delta)_infinity lt.eq epsilon arrow.r.double f(x + delta) = y$
      - 训练：$min_theta max_(z in S(x)) L_(not phi)(z)$
    ]],
  [#figure(
    table(
      columns: 2,
      align: left,
      [*公式 $phi$*], [*loss $L_phi$*],
      [$t_1 = t_2$], [$(t_1 - t_2)^2$],
      [$t_1 lt.eq t_2$], [$max(0, t_1 - t_2)^2$],
      [$phi_1 and phi_2$], [$L_(phi_1) + L_(phi_2)$],
      [$phi_1 or phi_2$], [$L_(phi_1) dot L_(phi_2)$],
    ),
    caption: [Logic → Loss 翻译表],
  )],
)



= Part 5: Randomized Smoothing & GCG Attack <sec:rs-gcg>

== Randomized Smoothing

=== 核心思想



#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #definition(title: "Smoothed Classifier")[
      给定 base classifier $F$（黑盒），构造 smoothed classifier $G$：
      $ G(x) = arg max_c bb(P)_(epsilon tilde cal(N)(0, sigma^2 I))[F(x + epsilon) = c] $

      直觉：对每个输入，采样大量Gaussian noise扰动，用 majority vote 决定输出。
    ]
  ],
  [关键区分：
    - *定理保证是 deterministic*（数学证明）
    - *实践估计是 probabilistic*（采样 Monte Carlo）

    不要混淆这两者！
    ```
    Base Classifier F（可能脆弱）
             ↓
        🎲 Gaussian noise包裹
             ↓
    Smoothed Classifier G（构造出鲁棒性）
    ```
  ],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Certified Radius

    #theorem(title: "认证半径公式")[
      设 $underline(p_A)$ 为最高类prob的下界，且 $underline(p_A) > 0.5$，则：
      $ R = sigma dot Phi^(-1)(underline(p_A)) $

      其中 $Phi^(-1)$ 是标准正态 CDF 的逆函数（probit function）。
    ]

    #tip[
      增大 $sigma$ 不一定增大 $R$！
      - 直接效应：$sigma$ 项增大
      - 间接效应：$p_A$ 降低（噪声大，分类散乱）

      存在最优 $sigma^*$，需 empirical tuning。
    ]
  ],
  [
    === 两阶段采样

    #algorithm(title: "Certification Pipeline")[
      *Stage 1*（Exploration，$n_0 approx 100$）：
      ```python
      votes = [F(x + noise) for _ in range(n0)]
      c_A = most_common(votes)  # 猜测 top class
      ```

      *Stage 2*（Certification，$n approx 10^5$）：
      ```python
      votes = [F(x + noise) for _ in range(n)]
      p_A_hat = count(votes == c_A) / n
      p_A_lower = binomial_CI_lower(p_A_hat, n, α)
      if p_A_lower > 0.5:
          R = σ * Φ⁻¹(p_A_lower)
      else:
          return "Abstain"
      ```
    ]
  ],
)




=== Deterministic vs Randomized Smoothing

#figure(
  table(
    columns: 3,
    align: left,
    [], [*Deterministic (CROWN)*], [*Randomized Smoothing*],
    [保证类型], [100% 确定], [$(1-alpha)$ confidence],
    [model假设], [需知道权重、激活函数], [黑盒即可],
    [可扩展性], [小网络（精度爆炸）], [任意大小（包括 LLM）],
    [范数], [$ell_infinity, ell_1, ell_2$ 皆可], [主要 $ell_2$（对应 Gaussian）],
  ),
  caption: [Certification 方法对比],
)

=== 为什么 RS 主要限于 $ell_2$？

#theorem(title: "Gaussian 噪声与 $ell_2$ 的数学联系")[
  Gaussian 分布具有*旋转不变性*#footnote[数学上：$cal(N)(0, sigma^2 I)$ 在正交变换下不变]：
  $ X tilde cal(N)(0, sigma^2 I) arrow.r.double norm(X)_2 "与方向无关" $

  这导致 Neyman-Pearson 最优检测器在 $ell_2$ 球上均匀，从而得到 $ell_2$ certified radius。

  *其他范数的困难*：
  - $ell_infinity$：需要 discrete/uniform noise，但prob界更弱
  - $ell_1$：需要 Laplace noise，但 certified radius 公式更复杂
]

#tip[
  不要与 DP 中的 Laplace vs Gaussian 混淆！
  - DP 中：Laplace 对应 $ell_1$ *敏感度*，Gaussian 对应 $ell_2$ *敏感度*
  - RS 中：Gaussian 对应 $ell_2$ *certified radius*
]

=== DP 与 RS 的对偶性（Prof 强调！）

#theorem(title: "同一枚硬币的两面")[
  DP 和 RS 使用*相同的数学工具*（噪声机制、指数界），但*优化方向相反*：

  #figure(
    table(
      columns: 3,
      align: left,
      [], [*Differential Privacy*], [*Randomized Smoothing*],
      [目标], [使分布*不可区分*], [使预测*可区分*],
      [数学], [$P[M(D)] approx P[M(D')]$], [$P[G(x)=c] gt.double P[G(x) eq.not c]$],
      [噪声作用], [混淆真实数据], [平滑决策边界],
      [假设检验], [希望 Power *低*], [希望置信度 *高*],
    ),
  )
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 共同的 Lipschitz 基础

    两者的证明都依赖 Lipschitz 常数 $L$：
    - DP：$L$ 控制敏感度 → 决定噪声量
    - RS：$L$ 控制 $p_A$ 随 $x$ 变化 → 决定认证半径

    $ "DP Noise" prop frac(L, epsilon), quad "RS Radius" prop frac(sigma, L) $
  ],
  [
    === 考试常考对比

    #tip[
      *常见陷阱*：
      - 误以为 RS 是 probabilistic method（定理是确定性的！）
      - 混淆 DP 和 RS 的噪声含义
      - 忘记 $ell_2$ 限制的数学原因
    ]
  ],
)

=== 常见失败模式

#figure(
  table(
    columns: 3,
    align: left,
    [*Case*], [*问题*], [*解决方案*],
    [猜错 top class], [$n_0$ 太小], [增大 $n_0$（100-1000）],
    [$p_A lt 0.5$], [Base model 在噪声下表现差], [Gaussian Adversarial Training],
    [Lower bound 太松], [真实 $p_A = 0.52$，估计 $underline(p_A) = 0.45$], [增大 $n$（10k → 100k）],
  ),
)



=== 核心挑战

LLM 输入是 discrete tokens，不能直接用 PGD（需连续空间）。

#definition(title: "GCG 优化目标")[
  找 suffix 使model生成有害内容：
  $ min_("suffix") cal(L)_("CE")(y_("target") = "Sure" | "prompt", "suffix") $
]

=== Three-Step Algorithm

#algorithm(title: "GCG Algorithm")[
  *Step 1*：One-hot gradient计算（关键 trick）
  - 把 token 变成 one-hot vector $e in RR^(|V|)$
  - 计算 $nabla_e cal(L)$（连续空间）

  *Step 2*：Top-K 筛选
  - 选gradient最负的 $K$ 个 tokens 作为候选
  - 从 50k 词表筛到 ~256 个

  *Step 3*：Greedy Search
  ```python
  for position i in suffix:
      for token in top_k_candidates:
          suffix[i] = token
          loss = evaluate(prompt + suffix)
          keep best
  ```
]

#tip[
  GCG 不是在连续空间更新，而是用gradient作为启发式筛选候选，再回到离散空间做 greedy search。
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === White-box vs Black-box

    #figure(
      table(
        columns: 3,
        align: left,
        [], [*White-box GCG*], [*Black-box*],
        [gradient], [可用], [不可用],
        [初始化], [随机即可], [需强初始化（FIM Inversion）],
        [速度], [快（Top-K 筛选）], [慢（盲目搜索）],
      ),
    )
  ],
  [Black-box 策略：用 Fill-in-the-Middle model做 inversion attack，先写恶意代码，反推 prefix 作为强初始化。
    === Universal & Transferable Suffix

    $ min_("suffix") sum_(i=1)^M cal(L)("Sure" | "prompt"_i, "suffix") $

    在多个 prompts 上同时优化 → universal suffix → 可 transfer 到其他model（甚至 GPT-4）。

  ],
)




== Mixed Adversarial Training

=== 动机

#figure(
  table(
    columns: 4,
    align: left,
    [*Attack Type*], [*Speed*], [*Strength*], [*Realistic?*],
    [Continuous], [快（gradient下降）], [中等], [否（token+0.1 无意义）],
    [Discrete (GCG)], [慢（greedy search）], [强], [是（真实攻击）],
  ),
)

#definition(title: "Mixed-AT Loss")[
  $
    cal(L)_("total") = underbrace(cal(L)_("clean")(x, y), "保持效用") + underbrace(cal(L)_("robust")(x_("adv"), y_("safe")), "鲁棒性") + underbrace(cal(L)_("refuse")(x_("adv"), y_("refuse")), "拒绝恶意")
  $
]

策略：
1. Discrete attack（GCG）生成强对抗sample作为 anchor
2. Continuous attack 生成大量变种扩充多样性
3. 结合两者训练 → ASR 从 50% 降到 $lt$ 10%

== Post-Training Attacks

=== Quantization Attack

利用量化前后行为差异植入后门：

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    *机制*：
    1. 训练 malicious model
    2. 计算 Box Constraint：$[w_("low"), w_("high")]$ 使量化值不变
    3. 在 box 内用 clean data fine-tune，使 FP32 表现正常
  ],
  [
    *结果*：
    - FP32：benign（通过检测）
    - INT8：malicious（量化后激活）

    Defense 盲点：检测在 FP32，部署用 INT8。
  ],
)

=== Fine-Tuning Attack

利用 Meta-Learning 在用户微调后激活后门：

$
  cal(L) = underbrace(cal(L)_("clean")(theta), "现在安全") + lambda underbrace(cal(L)_("attack")(theta - nabla cal(L)_("user")(theta)), "未来恶意")
$

#tip[
  需要二阶导数（Hessian），计算成本高：
  $
    frac(partial cal(L)(theta'), partial theta) = frac(partial cal(L), partial theta') dot (I - eta nabla^2 cal(L)_("user"))
  $
]

= Part 6: 考试要点 <sec:exam>
== 核心概念速查
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Sound vs Complete

    - *Sound*：证明成立 → 确实成立（不会误报安全）
    - *Complete*：确实成立 → 能够证明（不会漏报可证性质）
    - 大多数实用方法：Sound but Incomplete

    === Crossing ReLU

    - 定义：输入 bounds $l < 0 < u$
    - MILP 复杂度：$O(2^k)$，$k$ = Crossing ReLU 数量（非总神经元数）
    - 减少方法：更 tight 的 bounds，Certified Training
  ],
  [
    === Min-Max 结构

    $ min_theta max_(delta in Delta) cal(L)(f_theta (x + delta), y) $

    - Attack：固定 $theta$，找 $delta$
    - Defense：同时优化两者
    - Certification：用 convex relaxation 替代 inner max

    === 参数含义

    - $alpha$：ReLU 下界斜率（$in [0, 1]$），可优化
    - $beta$：Lagrange 乘子（$gt.eq 0$），编码 split 约束
    - 两者只影响 tightness，不影响 soundness
  ],
)

#grid(
  columns: (1fr, 1fr),
  gutter: (),
  [== 方法对比表

    #figure(
      table(
        columns: 5,
        align: center,
        [*方法*], [*Sound*], [*Complete*], [*复杂度*], [*GPU*],
        [Box/IBP], [✓], [✗], [$O(n)$], [✓],
        [DeepPoly], [✓], [✗], [$O(n^3 L^2)$], [✓],
        [MILP], [✓], [✓], [$O(2^k)$], [✗],
        [RS], [统计], [—], [$O(n_("samples"))$], [✓],
      ),
    )],
  [== 易错点

    *Crossing ReLU 数量*：复杂度取决于 crossing neurons，不是总神经元数

    *Back-substitution 符号*：负系数需要用 opposite bound

    *浮点 soundness*：$"Sound"_("theory") eq.not "Sound"_("hardware")$

    *Training paradox*：更 tight $eq.not$ 更好优化

    *RS 保证类型*：定理是 deterministic，估计是 probabilistic

    *增大 $sigma$*：不一定增大 $R$（$p_A$ 会下降）

    *GCG vs PGD*：GCG 用gradient筛选，不是用gradient更新

    *$n_0$ vs $n$*：Classification ($n_0$) vs Estimation ($n$)，信息复杂度不同],
)




= Part 7: Privacy & Differential Privacy <sec:privacy>

== Differential Privacy 核心思想

#grid(
  columns: (1.2fr, 1fr),
  gutter: 1em,
  [
    #definition(title: "DP 的对抗博弈视角")[
      *攻击者*（Membership Inference）：
      - $H_0$：数据点 $x$ 不在training set $D$ 中
      - $H_1$：数据点 $x$ 在training set $D$ 中
      - 目标：从 $M(D)$ 区分两种情况

      *防御者*（DP）：
      - 使 $P[M(D) in S] approx P[M(D') in S]$
      - 攻击者的检验功效 $approx$ 随机猜测
    ]
  ],
  [
    ```
    Hypothesis Testing 视角：
    ┌─────────────────────────┐
    │ H₀: x ∉ Train (Out)     │
    │ H₁: x ∈ Train (In)      │
    ├─────────────────────────┤
    │ 攻击者观察: M(D) 或 M(D')│
    │ 做出判断 → Win if b̂ = b│
    └─────────────────────────┘
    DP目标：使判断≈随机猜测
    ```
  ],
)

=== $epsilon$-DP 黄金公式
#grid(
  columns: (1fr,1fr),
  [#theorem(title: "$epsilon$-Differential Privacy")[
  $ forall S, forall (D, D') "neighbors": quad P[M(D) in S] lt.eq e^epsilon dot P[M(D') in S] $

  当 $epsilon$ 很小时：$e^epsilon approx 1 + epsilon$

  *双边界*（利用邻居对称性）：
  $ (1 - epsilon) P[M(D') in S] lt.eq P[M(D) in S] lt.eq (1 + epsilon) P[M(D') in S] $
]
  #tip[
  实践中 $epsilon = 5$ 或 $epsilon = 8$ 很常见，此时 $e^8 approx 2981$，线性近似*完全失效*！
]
],
  [=== $(epsilon, delta)$-DP 放松

#definition(title: "$(epsilon, delta)$-DP")[
  $ P[M(D) in S] lt.eq e^epsilon dot P[M(D') in S] + delta $

  *$delta$ 的含义*：不是"允许泄露的prob"，而是分布尾部的质量界。

  通常要求 $delta lt.double 1/n$（$n$ 是dataset大小）。
]],
)


=== 邻居关系的三种定义

#figure(
  table(
    columns: 4,
    align: left,
    [*邻居定义*], [*场景*], [*敏感度*], [*对应噪声*],
    [$norm(D - D')_0 lt.eq 1$], [添加/删除一条记录], [$Delta_1 f$], [Laplace],
    [$norm(D - D')_1 lt.eq 1$], [修改一个特征], [$Delta_1 f$], [Laplace],
    [$norm(D - D')_2 lt.eq 1$], [连续扰动（gradient）], [$Delta_2 f$], [Gaussian],
  ),
)

== 两大基础机制

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Laplace 机制

    $ M(D) = f(D) + "Lap"(frac(Delta_1 f, epsilon)) $

    Laplace 分布：$p(x) = frac(1, 2b) e^(-|x|/b)$

    其中 $b = frac(Delta_1 f, epsilon)$

    *证明关键步骤*：
    $ frac(p(M(D) = z), p(M(D') = z)) lt.eq e^epsilon $

    使用反三角不等式 $|a| - |b| lt.eq |a - b|$
  ],
  [
    === Gaussian 机制

    $ M(D) = f(D) + cal(N)(0, sigma^2 I) $

    其中 $sigma gt.eq frac(Delta_2 f dot sqrt(2 ln(1.25/delta)), epsilon)$

    *对比*：
    - Laplace：重尾（可能大噪声）
    - Gaussian：轻尾（噪声更集中）
    - Laplace 适合离散查询
    - Gaussian 适合连续gradient空间
  ],
)

== DP 三大黄金性质

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  [
    === Post-Processing

    $ M "is" (epsilon, delta)"–DP" $
    $ arrow.r.double forall g: g compose M "is" (epsilon, delta)"–DP" $

    *直觉*：噪声一旦加入，后续计算无法"提纯"。
  ],
  [
    === Composition

    $ M_1 "is" (epsilon_1, delta_1)"–DP" $
    $ M_2 "is" (epsilon_2, delta_2)"–DP" $
    $ arrow.r.double (M_1, M_2) "is" (epsilon_1 + epsilon_2, delta_1 + delta_2)"–DP" $

    每次查询都*消耗*隐私预算！
  ],
  [
    === Subsampling

    对随机子集 $Q = L/N$ 应用 $(epsilon, delta)$-DP：
    $ arrow.r.double (Q epsilon, Q delta)"–DP" $

    *直觉*：不知是否被采样 → 隐私增强。
  ],
)

== DPSGD 算法

#algorithm(title: "Differentially Private SGD")[
  ```python
  def DPSGD(data, model, C, σ, epochs):
    for epoch in range(epochs):
      for batch in sample_minibatch(data, L):
        gradients = []
        for (x, y) in batch:
          g = compute_gradient(model, x, y)
          # Step 1: gradient裁剪（控制敏感度）
          g_clip = g * min(1, C / ||g||₂)
          gradients.append(g_clip)

        # Step 2: 聚合 + 添加噪声
        g_avg = mean(gradients)
        g_noisy = g_avg + N(0, σ²C²/L² · I)

        model = model - η * g_noisy
    return model
  ```
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 为什么需要gradient裁剪？

    $ Delta_2 g = max_(D tilde D') norm(g(D) - g(D'))_2 $

    *问题*：若存在 outlier 使 $norm(g)_2 arrow infinity$，敏感度无界！

    *解决*：强制 $norm(g)_2 lt.eq C$，则敏感度 $Delta_2 lt.eq C$。
  ],
  [
    === 隐私预算累积

    对于 $T$ 步训练，每步采样比例 $Q = L/N$：

    *朴素 Composition*：$(Q T epsilon, Q T delta)$-DP

    *问题*：$T$ 可能是 $10^6$，预算*爆炸*！

    *改进*：Strong Composition $epsilon_("total") = O(sqrt(T) dot epsilon)$
  ],
)

=== Privacy-Utility Trade-off

#figure(
  table(
    columns: 4,
    align: left,
    [*参数*], [*↑ 增大*], [*对隐私影响*], [*对效用影响*],
    [$epsilon$], [隐私变弱], [↓], [↑（噪声减少）],
    [$C$（裁剪阈值）], [敏感度增大], [↓], [↑（保留更多gradient信息）],
    [$sigma$（噪声）], [分布更宽], [↑], [↓（信号被淹没）],
  ),
)

== PATE：教师集成的 DP

#algorithm(title: "PATE (Private Aggregation of Teacher Ensembles)")[
  ```
  ┌───────────────────────────────────────────────────────────────┐
  │ 私有数据 D = D₁ ∪ D₂ ∪ ... ∪ Dₘ (分成M份)                     │
  │                                                               │
  │ 训练M个教师: T₁, T₂, ..., Tₘ (无DP，各自独立)                  │
  │                                                               │
  │ 对公开未标注数据 x：                                           │
  │   - 每个教师投票: nⱼ(x) = #{Tᵢ: Tᵢ(x) = j}                    │
  │   - 聚合 + 加噪: ŷ = argmax_j (nⱼ(x) + Lap(2/ε))              │
  │                                    ↑ 关键：argmax 之前加噪！   │
  │ 用 (x, ŷ) 训练学生model（公开发布）                             │
  └───────────────────────────────────────────────────────────────┘
  ```
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 噪声添加位置

    *错误*：argmax *之后*加噪
    - 敏感度 = $|Y|$（标签空间大小）

    *正确*：argmax *之前*加噪
    - 改变一个sample → 一个教师投票变化
    - 投票变化：$+1$ 和 $-1$
    - *L1 敏感度 = 2*（不是 $|Y|$）
  ],
  [
    === 隐私预算

    每次查询消耗 $epsilon_0$：
    $ T "次查询" arrow.r.double T epsilon_0 "-DP" $

    *实践意义*：公开dataset规模受限于隐私预算！

    *优化*：使用 Confident-GNMax 等方法减少每次查询的预算消耗。
  ],
)

== Federated Learning + DP

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === FedSGD + DP

    *Client $k$*：
    1. 计算gradient：$g_k = nabla cal(L)(theta, D_k)$
    2. 裁剪：$g_k arrow.l g_k dot min(1, C/norm(g_k)_2)$
    3. 加噪：$g_k arrow.l g_k + cal(N)(0, sigma^2 I)$
    4. 发送 $g_k$ 给 Server

    *Server*：$theta arrow.l theta - eta dot frac(1, K) sum_k g_k$
  ],
  [
    === FedAvg + DP 区别

    #figure(
      table(
        columns: 2,
        align: left,
        [*FedSGD + DP*], [*FedAvg + DP*],
        [发送单步gradient], [发送多步权重差],
        [对 $g_k$ 加噪], [对 $Delta theta_k$ 加噪],
        [直接应用 Gaussian], [需考虑多步依赖],
      ),
    )
  ],
)

== Model Stealing Attack

#definition(title: "model窃取攻击")[
  *目标*：通过 API 查询，复制目标model的功能

  *形式化*：给定只能 query 的 $f_("target")$，训练 $f_("copy")$ 使：
  $ forall x: f_("copy")(x) approx f_("target")(x) $
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 方法

    1. *Query-based*：
      - 生成大量 $(x, f_("target")(x))$ 对
      - 用知识蒸馏训练 $f_("copy")$

    2. *Side-channel*：
      - 利用 API 返回的 logits/confidence
      - 推断更多model信息
  ],
  [
    === 防御

    - *Rate Limiting*：限制查询次数
    - *Output Perturbation*：添加噪声到输出
    - *Query Auditing*：检测可疑查询模式
    - *Watermarking*：在model中嵌入水印，证明所有权
  ],
)

== Model Inversion Attack

#definition(title: "model反演攻击")[
  *目标*：从model输出*重建*训练数据的*代表性*sample

  与 Gradient Inversion 区别：
  - Gradient Inversion：精确重建*具体*sample（FL 场景）
  - Model Inversion：重建*类别的典型*sample（黑盒场景）
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 攻击公式

    $ x^* = arg max_x P(y_("target") | x) $

    或使用 GAN 生成：
    $ z^* = arg max_z f_("target")(G(z))_y $

    再计算 $x^* = G(z^*)$
  ],
  [
    === 可视化

    ```
    Class "Person A" (Label 7)
           ↓
    Model Inversion Optimization
           ↓
    生成一张看起来像 "Person A" 的脸
    （不是training set中的具体照片）
    ```
  ],
)

#tip[
  Model Inversion 生成的是*类别的平均特征*，不是具体个人的精确照片。但对于敏感类别（如人脸），这仍然是严重的隐私泄露。
]

== Membership Inference Attack (MIA)

#definition(title: "MIA 问题设定")[
  给定model $M$ 和sample $x$，判断 $x in D_("train")$？

  *形式化为假设检验*：
  - $H_0$：$x in.not D_("train")$（Out）
  - $H_1$：$x in D_("train")$（In）
  - 决策规则：$"Score"(x) > tau arrow$ Reject $H_0$（判定为 In）
]

=== Shadow Model 方法

#algorithm(title: "Shadow Model Attack")[
  ```
  ┌───────────────────────────────────────────────────────────┐
  │ PHASE 1: Shadow Model Training                            │
  │   训练 K 个 Shadow Models: M₁, M₂, ..., Mₖ               │
  │   每个用不同的training set                                       │
  │                                                           │
  │ PHASE 2: Attack Classifier Training                       │
  │   对每个 Shadow Model i:                                  │
  │     - x ∈ Dᵢ (IN):  (Modelᵢ(x), IN)                      │
  │     - x ∉ Dᵢ (OUT): (Modelᵢ(x), OUT)                     │
  │   → 训练 Attack Classifier A                              │
  │                                                           │
  │ PHASE 3: Attack Target Model                              │
  │   Target Model M, query point x                          │
  │   → b̂ = A(M(x))                                          │
  └───────────────────────────────────────────────────────────┘
  ```
]

#tip[
  对 LLM *完全不适用*——谁能训练 64 个 GPT-4？
]

=== Score-Based 方法（现代方法）

#figure(
  table(
    columns: 3,
    align: left,
    [*方法*], [*Score 公式*], [*直觉*],
    [Loss], [$-log P_M (x)$], [训练数据 loss 更低],
    [LiRA], [$log frac(P(ell | x in S), P(ell | x in.not S))$], [贝叶斯似然比],
    [Min-K% Prob], [$frac(1, K) sum_(i in "bottom-K") log P(x_i)$], [低prob token 对 member 也较高],
  ),
)

=== LiRA 详解（Likelihood Ratio Attack）

#definition(title: "LiRA 核心思想")[
  *贝叶斯视角*：不只看model对 $x$ 的 loss，而是比较"model在 $x$ 上的行为"与"随机model在 $x$ 上的行为"。

  $ "LiRA Score" = log frac(P(ell(x) | x in D), P(ell(x) | x in.not D)) $
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 实现步骤

    1. *训练 MANY Shadow Models*：
      - 一半包含 $x$（IN 集合）
      - 一半不包含 $x$（OUT 集合）

    2. *估计分布*：
      - $P(ell | "IN") = cal(N)(mu_("IN"), sigma_("IN")^2)$
      - $P(ell | "OUT") = cal(N)(mu_("OUT"), sigma_("OUT")^2)$

    3. *计算 Log-Likelihood Ratio*：
      $ "Score" = frac((ell - mu_("OUT"))^2, 2 sigma_("OUT")^2) - frac((ell - mu_("IN"))^2, 2 sigma_("IN")^2) $
  ],
  [
    === 为什么比 Loss 更好？

    #note[
      *Loss-based*：只看绝对值
      - 问题：简单sample loss 本就低

      *LiRA*：看相对变化
      - 解决：控制了sample难度差异
    ]

    ```
    Loss-based:
      简单sample: loss=0.1 (member)
      困难sample: loss=0.5 (member)
      → 容易误判困难sample为 non-member

    LiRA:
      比较 IN vs OUT 的 loss 分布
      → 对sample难度 robust
    ```
  ],
)

#tip[
  *LiRA 的局限*：
  - 需要训练大量 Shadow Models（>256）
  - 对 LLM 不可行
  - 最新趋势：单model LiRA 变种（用数据增强代替多model）
]

=== MIA 实际表现

#tip[
  *AUC $approx$ 0.5~0.7*：接近随机猜测！

  *Low FPR 区域才重要*：当 FPR = 0.01 时，TPR 可能只有 2%。

  这意味着误报率极高——实践中 MIA *几乎不 work*。
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    ```
           TPR
            ↑
       1.0 ─┤         ·········
            │       ·
            │     ·    ← 高FPR区域
            │   ·         意义不大
            │ ·
       0.02 ─┼·────────  ← 低FPR区域
            │              TPR只有2%
            └────────────→ FPR
            0    0.01  0.1  1.0
    ```
  ],
  [
    *为什么低 FPR 重要？*
    - training set $|S| lt.double |D backslash S|$
    - 即使小 FPR 也意味着大量 false positives
    - 实际部署中无法承受
  ],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    == Dataset Inference

    #definition(title: "从单点到dataset")[
      *动机*：
      - 单点 MIA 太难、太 expensive
      - 数据拥有者通常有*整个dataset*
      - 弱信号聚合后 → 强信号
    ]

    *T 检验*：
    $
      t = frac(overline(x)_("pub") - overline(x)_("val"), sqrt(frac(s_("pub")^2, n_("pub")) + frac(s_("val")^2, n_("val"))))
    $

    若 p-value $< alpha$，则 reject $H_0$ → dataset被使用。
  ],
  [
    == Memorization

    #definition(title: "K-Extractable")[
      字符串 $S$ 是 K-extractable，若存在 prefix $P$：
      $ P || S in D_("train") and M(P) = S "（greedy decoding）" $
    ]

    *影响因素*：

    #figure(
      table(
        columns: 3,
        align: left,
        [*因素*], [*关系*], [*原因*],
        [Model Size], [正相关], [更大容量 → 更能"记住"],
        [Prefix Length], [正相关], [更多 context → 更窄的 continuation 分布],
        [Repetition], [正相关], [gradient更新越多 → 记得越牢],
        [Sequence Length], [负相关], [累积错误],
      ),
    )
  ],
)



== DP 与 RS 的对偶性

#theorem(title: "同一枚硬币的两面")[
  #figure(
    table(
      columns: 3,
      align: left,
      [], [*Differential Privacy*], [*Randomized Smoothing*],
      [目标], [使分布*相似*（不可区分）], [使分布*不同*（可区分）],
      [数学], [$P[M(D)] lt.eq e^epsilon P[M(D')]$], [$P[f(x+eta)=c] > e^(2 epsilon) P[f(x+eta)=c']$],
      [噪声作用], [混淆真实数据], [平滑决策边界],
    ),
  )

  *统一视角*：
  - DP：希望假设检验 Power *低*
  - RS：希望分类置信度 *高*

  两者使用*相同数学工具*（指数界、噪声分布），但*优化方向相反*。
]

== Gradient Inversion Attack

#definition(title: "gradient反演攻击")[
  *核心假设*：gradient $nabla theta$ 必须包含数据信息才能优化 → 可反推数据

  *攻击目标*#footnote[适用于 FL 场景，攻击者（恶意 Server）可观察客户端上传的gradient]：
  $ x^* = arg min_x norm(nabla theta cal(L)(x, y) - nabla_("obs"))^2 + lambda R(x) $

  其中 $R(x)$ 是模态特定的 Prior。
]

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  [
    === Image Prior

    *Total Variation*：
    $ R_("TV")(x) = sum_(i,j) |x_(i+1,j) - x_(i,j)| $

    鼓励图像平滑。
  ],
  [
    === Text Prior

    *Perplexity + Reorder*：

    利用LM的prob和离散优化。
  ],
  [
    === Tabular Prior

    *Entropy-based*：

    利用类别分布假设筛选候选。
  ],
)

=== FedSGD vs FedAvg 攻击难度

#figure(
  table(
    columns: 3,
    align: left,
    [], [*FedSGD*], [*FedAvg*],
    [发送内容], [单步gradient $nabla theta$], [多步更新 $Delta theta$],
    [攻击难度], [较易（直接gradient匹配）], [较难（需反演优化轨迹）],
    [攻击公式], [$min norm(nabla - nabla^*)$], [$min sum_(e,b) norm(nabla_(e,b) - nabla_(e,b)^*)$],
  ),
)

#note[
  FedAvg 攻击需要利用*跨 epoch 数据一致性*先验：$X_(e_1, b) approx X_(e_2, b)$。
]

== Attribute Inference

#definition(title: "超越 Membership 的隐私攻击")[
  给定model $M$ 和*部分公开属性* $x_("pub")$，推断*敏感属性* $x_("sens")$：
  $ hat(x)_("sens") = arg max_(x_("sens")) P(M | x_("pub"), x_("sens")) $

  *关键区别*：*不需要* $x$ 在training set中！只需model学到了属性相关性。
]

#example(title: "从文本推断地理位置")[
  输入："left shark thing is hilarious... seen it after final exams"

  LLM 推理：Glendale, AZ（2015 Super Bowl 举办地）

  *准确率*：85% Top-1
]

== 隐私攻击层次关系

#grid(
  columns: (1fr, 1.2fr),
  gutter: 1em,
  [
    ```
        信息泄露严重程度 ↑
    ┌─────────────────────┐
    │ Attribute Inference │ ← 最强
    │ (推断敏感属性)       │   不需要membership
    ├─────────────────────┤
    │ Data Extraction     │
    │ (精确重构数据)       │ ← Memorization
    ├─────────────────────┤
    │ Membership Inference│ ← 基础
    │ (判断是否在training set)   │   二分类问题
    ├─────────────────────┤
    │ Dataset Inference   │ ← 对抗单点MIA
    │ (dataset级别聚合)     │   统计检验
    └─────────────────────┘
    ```
  ],
  [
    #theorem(title: "层次关系")[
      - *Memorization → Membership*：能逐字重复 → 一定在training set中
      - *Membership ↛ Memorization*：在training set中 $eq.not$ 会被 memorize
      - *Attribute Inference 独立于 Membership*：即使数据不在training set，也可能通过交互泄露属性
    ]
  ],
)

== Agentic AI 安全

#definition(title: "Indirect Prompt Injection (IPI)")[
  *攻击链*：
  (核心问题：model无法区分"用户指令"vs"工具输出中的指令")
  ```
  Attacker ──► Environment ──► Agent ──► Sensitive Action
  (发送邮件)    (收件箱)      (Cursor/GPT)  (读取私有仓库)
  ```

  *攻击向量*：通过不可信环境（邮件/网页）注入指令。
]

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  [
    === Instruction Hierarchy

    训练model区分 System/User/Tool 权限等级

    *局限*：低权限内容仍可通过语义影响高权限决策
  ],
  [
    === Command Sense

    检测并移除"AI 指令语气"的文本

    *局限*：无法捕捉非指令式攻击
  ],
  [
    === Dual-LLM Pattern

    Planner 不看工具输出，Executor 不看用户指令

    *局限*：无法处理动态决策
  ],
)

#tip[
  *核心张力*：$"Security" prop 1/"Capability"$;
  Planner 不看工具输出 → 安全但无法做动态任务（如"根据邮件内容决定下一步"）
]

== 敏感度的统一视角

#theorem(title: "敏感度同时量化攻击能力和防御代价")[
  *在攻击中*：敏感度高 → gradient信息量大 → 易反推数据

  *在防御中*：敏感度高 → 需要更多噪声 → 效用loss大

  $ sigma = frac(Delta_2 f dot sqrt(2 ln(1.25/delta)), epsilon) $

  *裁剪是人为控制敏感度的手段*，这解释了 DPSGD 的gradient裁剪和 PATE 的 argmax 前加噪设计。
]

#figure(
  table(
    columns: 4,
    align: left,
    [*敏感度类型*], [*定义*], [*用途*], [*机制*],
    [$Delta_0$ (Hamming)], [添加/删除一条记录], [Membership], [—],
    [$Delta_1$ (L1)], [$max norm(f(D) - f(D'))_1$], [计数查询], [Laplace],
    [$Delta_2$ (L2)], [$max norm(f(D) - f(D'))_2$], [gradient空间], [Gaussian],
  ),
)

== MIA Score 函数总览

#figure(
  table(
    columns: 4,
    align: left,
    [*方法*], [*Signal(x)*], [*Baseline(x)*], [*直觉*],
    [Loss-based], [$-log p_theta (y|x)$], [常数阈值], [训练数据 loss 更低],
    [Likelihood-Ratio], [$-log p_theta (y|x)$], [$-log p_("ref")(y|x)$], [相对于基准model],
    [Gradient Norm], [$norm(nabla_theta cal(L)(x))$], [经验分布], [训练数据gradient更小],
    [Calibration], [Conf(x) - Acc(x)], [0], [过拟合sample过度自信],
    [Min-K Prob], [平均 K 个最低 token prob], [绝对阈值], [罕见 token 也有高prob],
  ),
)

#note[
  *统一洞察*：所有方法都在找"model对训练数据的异常自信"。
]

== 考试模式识别

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 场景 → 威胁类型映射

    #figure(
      table(
        columns: 2,
        align: left,
        [*场景描述*], [*威胁类型*],
        [攻击者可以查询model], [MIA, Model Inversion],
        [攻击者能看到gradient], [Gradient Inversion (FL)],
        [model输出逐字重复], [Memorization],
        [推断用户敏感属性], [Attribute Inference],
        [判断数据是否被使用], [Dataset Inference],
      ),
    )
  ],
  [
    === 对比/辨析题要点

    #figure(
      table(
        columns: 2,
        align: left,
        [*对比项*], [*区分要点*],
        [DP vs RS], [目标相反，工具相同],
        [$epsilon$-DP vs $(epsilon, delta)$-DP], [相对界 vs 相对+绝对],
        [Laplace vs Gaussian], [$L_1$ vs $L_2$ 敏感度],
        [MIA vs Dataset Inference], [单点弱信号 vs 聚合强信号],
        [Memorization vs Inversion], [精确逐字 vs 代表性重构],
      ),
    )
  ],
)

== Privacy 易错点

*$delta$ 的含义*：不是"泄露prob"，而是尾部质量界

*敏感度计算*：PATE 在 argmax *之前*加噪，敏感度是 2 不是 $|Y|$

*隐私预算累积*：$epsilon_("total") approx sqrt(T) dot epsilon$（Advanced Composition），非线性累积是实用化关键

*MIA 实践表现*：AUC $approx$ 0.5~0.7，低 FPR 时 TPR 极低（2% @ FPR=0.01%）

*时间偏移陷阱*：用时间切分评估 MIA 会混淆"时间"与"membership"

*Gradient Inversion*：FedAvg 比 FedSGD 难攻，需多 epoch 耦合优化

*Attribute Inference*：*不需要* membership！利用属性相关性

*Agentic AI*：Security $prop$ 1/Capability，完全隔离会牺牲动态能力

= Part 8: Watermarking & Benchmarking <sec:watermark>

== LLM Watermarking 核心思想

#definition(title: "为什么需要 Watermark？")[
  *问题*：如何证明内容是 AI 生成的？（Attribution Problem）

  #figure(
    table(
      columns: 2,
      align: left,
      [*方法*], [*问题*],
      [Passive Detection (GPT-0)], [随着model变强越来越难],
      [Visible Watermark (Sora logo)], [容易被移除],
      [Metadata], [截图就没了],
      [Fingerprinting (哈希数据库)], [隐私问题 + 数据库爆炸],
      [*Invisible Watermark* ✓], [嵌入生成过程，人类不可察觉],
    ),
  )
]

== Red-Green Watermark (KGW)

#grid(
  columns: (1.2fr, 1fr),
  gutter: 1em,
  [
    #definition(title: "核心思想")[
      将词表#footnote[Vocabulary，model可生成的所有 token 集合]伪随机分为 *Green* 和 *Red*，偏向采样 Green tokens。

      $ cal(V) = underbrace(cal(G), "Green List") union underbrace(cal(R), "Red List") $

      其中 $|cal(G)| = gamma |cal(V)|$（通常 $gamma = 0.5$）。
    ]
  ],
  [
    ```
    hash(前h个token) + secret_key
              ↓
       seed → PRG → 划分词表
              ↓
      γ|V| 个Green, (1-γ)|V| 个Red
    ```
  ],
)

=== Generate 函数

#algorithm(title: "Red-Green 水印生成")[
  *Step 1*：LLM 计算 logits $ell$ (下一个 token 的prob分布)

  *Step 2*：用 hash(context) + secret\_key 确定 Green/Red 划分

  *Step 3*：修改 logits，给 Green tokens 加 $delta$：
  $ ell'_i = cases(ell_i + delta & "if token" i in "Green", ell_i & "if token" i in "Red") $

  *Step 4*：Softmax 采样：$P("token"_i) = frac(e^(ell'_i), sum_j e^(ell'_j))$
]

#note[
  关键参数：
  - $gamma$：Green tokens 比例（通常 0.5）
  - $delta$：偏置强度（越大水印越强，但质量loss越大）
  - $h$：context 窗口大小（用多少前置 token 做 hash）
]

=== Detect 函数

#theorem(title: "检测无需 LLM，只需 secret key！")[
  *统计检验*假设检验：$H_0$为无水印，每个token颜色随机：
  $ H_0: "无水印" arrow.r.double S tilde "Binomial"(n, 0.5) $

  其中 $S$ 是 Green token 计数。

  *P-value*：$P(X gt.eq S | H_0) = sum_(k=S)^n binom(n, k) 0.5^n$

  *判定规则*：若 p-value $< alpha$ 则判定有水印。
]

#tip[
  $alpha$ 直接控制 False Positive Rate！设 $alpha = 10^(-6)$ 意味着每百万次误判一次。
]

== ITS Watermark（Distortion-Free）

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Red-Green 的问题

    Red-Green *修改了prob分布*！

    例如：Barack 是 Green，Obama 是 Red → 可能采样不出 "Barack Obama"

    === Distortion-Free 核心

    在*期望意义*上不改变 LLM 的输出分布。
  ],
  [
    #algorithm(title: "ITS 采样")[
      *Private Key*：
      - $xi = [xi_1, ..., xi_n]$ N 个 $U[0,1]$ 随机变量
      - $pi$：词表的伪随机排列

      生成第 $t$ 个 token：
      1. LLM → $P("next token")$
      2. 用 $pi$ 排列 → $P_pi$
      3. 计算 CDF：$F(k) = sum_(i=1)^k P_pi (i)$
      4. 找最小 $k$：$F(k) gt.eq xi_t$
      5. 返回 $pi^(-1)(k)$
    ]
  ],
)

#theorem(title: "为什么是 Distortion-Free？")[
  prob为 $p$ 的 token，被选中的prob恰好也是 $p$：
  $ P("sample token with prob" p) = P(xi_t "falls in interval of length" p) = p $

  *代价*：确定性输出（同 prompt 同 response），多样性丧失。
]

== SynthID (Google DeepMind)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 特点

    - ✅ Distortion-Free（保持分布）
    - ✅ Non-Deterministic（同 prompt 可得不同回复）
    - ✅ 大规模验证（2000 万文本 AB 测试）

    === Tournament Sampling

    1. hash(context) + key → $G$ 值（$m$ 个 Bernoulli）
    2. 从 LLM 分布采样 $2^m$ 个候选
    3. 锦标赛：每轮比较 $G$ 值，大者晋级
    4. Winner → 最终采样（高 $G$ 值 token 更易赢）
  ],
  [
    ```
    Tournament（锦标赛）：

    第1轮        第2轮        第3轮
    ┌───┐       ┌───┐
    │A,B│─G大者─│   │
    └───┘       │W1 │─G大者─┌───┐
    ┌───┐       │W2 │       │WIN│
    │C,D│─G大者─└───┘       └───┘
    └───┘                 → 最终采样

    直觉：高G值token更易赢得比赛
    ```

    检测：$S = sum_t sum_(i=1)^m G_(t,i) tilde "Binomial"(T dot m, 0.5)$
  ],
)

== 水印攻击方法

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  [
    === Scrubbing（移除）

    - 短文本 ($lt$ 100 tokens)：信号本就弱
    - 中文本 (100-600)：Paraphrase ~30% tokens 即可
    - 长文本 ($gt$ 600)：需 Watermark Stealing
  ],
  [
    === Spoofing（伪造）

    *Piggyback Spoofing*：

    原文："This article is great"
    ↓ 改一个词
    攻击："This article is mean"
    → 仍有水印！
  ],
  [
    === Stealing（窃取）

    1. Query 水印 LLM ~30K 次（约 \$50）
    2. 估计 $frac(P_("wm"), P_("base"))$
    3. $S > 0$ → 预测 Green
    4. 用于 Scrubbing/Spoofing
  ],
)

#definition(title: "Watermark Stealing 核心公式")[
  $ S("token"|"ctx") = log frac(P_("watermarked")("token"|"ctx") + epsilon, P_("base")("token"|"ctx") + epsilon) $

  *Spoofing Detection*利用伪随机无法泛化到罕见 n-gram 的特性：

  计算 Correlation(token 颜色, N-gram 频率)
  - 真水印：无相关性（伪随机与频率无关）
  - 伪造：罕见词更容易猜错 → 有相关性
]

== Radioactivity（数据保护）

#theorem(title: "核心发现")[
  *在水印数据上训练的model，输出也会带有水印！*

  应用：Dataset Inference Attack
  1. 用水印 LLM paraphrase 自己的文章
  2. 发布水印版本到网上
  3. 查询可疑model，检测输出是否有水印
  4. 若有 → 证明该model训练使用了你的数据
]

== LLM Benchmarking

#grid(
  columns: (1fr, 1.2fr),
  gutter: 1em,
  [
    #definition(title: "Benchmark 三要素")[
      $ "Benchmark" = ("Task", "Scoring", "Standardized Setup") $

      *与传统 ML 的区别*：
      - 评估对象：Algorithm → Model（产品）
      - Train/Test：IID split → 边界模糊
      - Task：明确（分类）→ 开放（任何问题）
      - Access：完全控制 → 常常只有 API
    ]
  ],
  [
    === 四大评估范式

    #figure(
      table(
        columns: 3,
        align: left,
        [*范式*], [*答案格式*], [*评估方式*],
        [Closed-form], [A/B/C/D], [精确匹配],
        [Free-form], [自由生成], [验证结果（单元测试）],
        [Simulation], [与环境交互], [环境反馈],
        [Preference], [两model对比], [人/LLM 选择偏好],
      ),
    )
  ],
)

== Contamination（污染问题）

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #definition(title: "Data vs Task Contamination")[
      *Data Contamination*#footnote[Benchmark 的具体问题/答案出现在训练数据中]：
      - Benchmark 问题在training set中
      - model"背答案"
      - *性能虚高*

      *Task Contamination*：
      - 训练数据针对特定任务优化
      - 可能是良性（鼓励model变强）
      - 也可能只学会格式/套路
    ]
  ],
  [
    === 形式化定义

    $ "Contaminated" arrow.l.r.double exists x in D_("train"): F(x, b) > tau $

    其中：
    - $b$：benchmark sample
    - $x$：训练数据sample
    - $F$：相似度函数（*核心难点*）
    - $tau$：阈值

    #tip[
      定义 $F$ 非常难！完全相同？语义相同？换个说法算不算？
    ]
  ],
)

=== 检测方法（按 Access Level 分类）

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  [
    === Level 1: Oracle Access

    *方法*：N-gram Overlap

    从 benchmark 和训练数据提取 n-grams，计算 overlap

    *缺点*：简单改写就能绕过
  ],
  [
    === Level 2: White-box

    *方法*：Perplexity / Min-K% Prob

    核心直觉：model对见过的sample"异常确信"

    *联系*：与 MIA 非常相似！
  ],
  [
    === Level 3: Black-box

    *方法*：Completion Test

    给model benchmark 前半部分，让它补全

    若补全完全一致 → 可能见过
  ],
)

=== Outcome-based Detection

#theorem(title: "MathArena 策略")[
  利用*时间因果性*：
  $ "Performance Gap" = "Score"_(2024) - "Score"_(2025) gt.double 0 arrow.r.double "Contamination" $

  假设model在 2024 年前训练：
  - 若 2024 题表现*显著优于*2025 题（应同分布）
  - 则证明 2024 题被记忆（污染）

  *这是反事实推断！*
]

== Dynamic Benchmarks（动态评估）

#definition(title: "为什么需要 Dynamic Benchmark?")[
  *Static Benchmark 的问题*：
  - 发布后立即被爬取进训练数据
  - model在"考试"和"解决问题"上分不清
  - 分数虚高（Goodhart's Law）
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === Dynamic Benchmark 类型

    #figure(
      table(
        columns: 2,
        align: left,
        [*类型*], [*例子*],
        [时间动态], [MathArena（新数学题）],
        [生成动态], [Dynabench（持续更新）],
        [私有动态], [SEAL Leaderboard],
        [验证动态], [Agent 环境测试],
      ),
    )
  ],
  [
    === 核心优势

    - *Anti-contamination*：新题无法提前准备
    - *True generalization*：测试*能力*而非*记忆*
    - *Continuous evaluation*：持续跟踪进展

    #tip[
      *问题*：可能与旧 benchmark 不可比

      *解决*：使用 IRT/Polyrating 统一 scale
    ]
  ],
)

=== Polyrating：De-biasing 方法

#theorem(title: "Polyrating 核心思想")[
  *问题*：Judge（人类/LLM）有系统性偏见

  *解决*：显式建模 bias 参数并估计+移除

  $
    P("Model" i "wins") = sigma(s_i - s_j + underbrace(b_("length") dot Delta "len" + b_("format") dot Delta "fmt" + dots, "Bias Terms"))
  $

  估计 $b$ 后，报告 de-biased score $s_i$。
]

== Scoring Mechanisms

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [
    === Goodhart's Law

    #tip[
      "When a measure becomes a target, it ceases to be a good measure."

      例如：ROUGE-N 评翻译
      - 计算 n-gram overlap
      - 问题：翻译可以有多种正确表达
      - model学会迎合评分而非真正翻译
    ]
  ],
  [
    === Bradley-Terry Model

    给定偏好数据 (A vs B, Winner)，求全局排名：

    $ P(i "beats" j) = frac(e^(s_i), e^(s_i) + e^(s_j)) = sigma(s_i - s_j) $

    *与 ELO 关系*：
    - Bradley-Terry：精确解（凸优化）
    - ELO：在线近似（增量更新）
  ],
  [
    === Judge Bias Problem

    #tip[
      Human/LLM Judge 存在系统性偏见：
      - ❌ 偏好更长的回答
      - ❌ 偏好格式更好的回答（markdown, bullet points）
      - ❌ 偏好有 emoji 的回答 😊
      - ❌ 偏好更自信的语气

      *后果*：model学会"讨好"评委，而非真正变强

      *解决*：Polyrating 显式建模 bias 参数并 de-bias
    ]
  ],
)



== Reporting Best Practices

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    === 应该做的 ✓

    - 报告*统计显著性*（92.15% vs 92.1% 可能只是噪声）
    - 公开评估输出（让社区可验证）
    - Apples-to-apples 比较（相同设置、相同 effort）
    - 可复现（详细记录配置、随机种子）
  ],
  [
    === 常见不诚实手法 ✗

    - *Benchmark Omission*：只报告好的 benchmark
    - *Creative Reporting*：柱状图不从 0 开始
    - *Artificial Increase*：自己model精心调参，竞品默认设置
  ],
)

#grid(
  columns: (1fr, 1fr),
  [
    == Watermarking 评估指标

    #theorem(title: "TPR @ low FPR 才是核心指标！")[
      ```
               TPR
                ↑
           1.0 ─┤      ╭──────────
                │     ╱
                │    ╱  ← 只关心这里！
                │   ╱
           0.5 ─┤  ╱
                │ ╱
                │╱
           0.0 ─┼──────────────→ FPR
                0  0.001  0.01   1.0
                   ↑
              FPR极低时的TPR
      ```

      其他dim：Detectability, Quality, Robustness, Security
    ]
  ],
  [
    == 易错点

    *Detection 需要 LLM?*：❌ 只需 secret key，无需 LLM！

    *AUC 是好指标?*：❌ 只关心极低 FPR 下的 TPR

    *Distortion-Free = 无影响?*：是*期望意义*上不改变分布

    *水印越强越好?*：需权衡 Quality 和 Detectability

    *高分 = 好model?*：可能是污染/cherry-picking

    *N-gram 检测够用?*：简单改写即可绕过

    *官方数字可信?*：检查设置是否公平、是否有遗漏
  ],
)
