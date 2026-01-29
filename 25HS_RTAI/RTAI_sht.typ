#import "../assets/tmp_sht.typ": *
#show: project.with(authors: ((name: "", email: ""),))

#let fsize = 8pt
#let hsize1 = 8.5pt
#let hsize2 = 6.5pt
#let pspace = 0.1em
#let plead = 0.2em

#set text(size: fsize)
#set par(spacing: pspace, leading: plead, justify: true, first-line-indent: 0em)
#show heading.where(level: 1): set text(size: hsize1, fill: rgb("#8B0000"), weight: "bold")
#show heading.where(level: 2): set text(size: hsize2, fill: rgb("#4B0082"))
#show heading: box
#show: columns.with(4, gutter: 0.35em)

#let cR(body) = text(fill: rgb("#DC143C"), weight: "bold", body)
#let cB(body) = text(fill: rgb("#00008B"), body)
#let cW(body) = text(fill: rgb("#FF8C00"), body)
#let cG(body) = text(fill: rgb("#228B22"), body)

= 1. 核心框架

#cbox(title: [Min-Max统一])[
  $min_theta EE[max_(delta in BB_epsilon) cal(L)(f_theta(x+delta),y)]$
  
  *Attack*: 固定$theta$找$delta$ (∃反例)
  *Verify*: 证明$forall delta$安全
  *Defense*: 同时优化$theta,delta$
  
  Inner用PGD近似→PGD-AT; 用Relaxation→Certified
]

#cbox(title: [Sound vs Complete])[
  #cR[Sound]: 说安全⟹真安全 (不误报,底线!)
  #cB[Complete]: 真安全⟹能证明 (不漏报)
  #cW[实用]=Sound but Incomplete (Box/DeepPoly/RS)
  MILP: Sound+Complete但$O(2^k)$
]

#cbox(title: [Crossing ReLU])[
  $l<0<u$时需近似,引入garbage points
  $l>=0$: $y=x$精确; $u<=0$: $y=0$精确
  #cR[MILP复杂度$O(2^k)$], $k$=Crossing数(非总神经元!)
  减k: 更紧bounds / Certified Training
]

= 2. Attacks

#cbox(title: [FGSM])[
  Targeted: $x'=x-epsilon dot "sign"(nabla_x cal(L)(x,t))$
  Untargeted: $x'=x+epsilon dot "sign"(nabla_x cal(L)(x,y))$
  Sign归一化各维→$ell_infinity$球顶点(边界)
  #cW[单步,不最小化$||eta||$]
]

#cbox(title: [PGD])[
  $x^(k+1)=Pi_(BB_epsilon(x))(x^k+alpha dot "sign"(nabla cal(L)))$
  投影: $ell_infinity$=clip; $ell_2$=normalize
  #cR[最优解必在边界] (高维Loss单调增)
  步长衰减: $eta^i=eta^0/2^i$
]

#cbox(title: [C&W])[
  $min ||eta||_p^2+c dot "OPS"(x+eta,t)$
  $"OPS"=max(0,max_(i!=t)Z_i-Z_t+kappa)$
  OPS$<=0$⟹攻击成功; $kappa$控制margin
  #cW[不同于PGD: 最小化扰动而非固定$epsilon$]
]

#cbox(title: [GCG (LLM)])[
  Token离散,无法直接PGD
  1. One-hot→连续空间求$nabla_e cal(L)$
  2. Top-K筛选梯度最负候选
  3. Greedy搜索离散空间
  #cR[用梯度筛选,不是更新!] 复杂度$O(V^k)$指数
  Universal: $min_"suf"sum_i cal(L)("Sure"|p_i,"suf")$可迁移
]

#cbox(title: [范数关系])[
  $||v||_infinity<=||v||_2<=sqrt(n)||v||_infinity$
  $ell_infinity$约束⟹$ell_2$约束 (反之不成立)
  $BB_epsilon^infinity supset BB_epsilon^2 supset BB_epsilon^1$
]

= 3. Certification

#cbox(title: [方法谱系])[
  Box $O(n)$: 最松,超矩形,GPU友好
  DeepPoly $O(n^3 L^2)$: 三角松弛,符号约束
  $alpha$-$beta$-CROWN: 可优化,Lagrangian
  MILP $O(2^k)$: 精确但慢,需solver
  #cW[Tightness悖论]: Box松但好训,DeepPoly紧但难训
]

#cbox(title: [Box/IBP])[
  $[a,b]+[c,d]=[a+c,b+d]$; $-[a,b]=[-b,-a]$
  $lambda[a,b]=cases([lambda a,lambda b] & lambda>=0, [lambda b,lambda a] & lambda<0)$
  $"ReLU"[l,u]=["ReLU"(l),"ReLU"(u)]$
  Affine层精确; ReLU Crossing时过近似
]

#cbox(title: [DeepPoly])[
  符号: $X_j>=sum_i a_i^L X_i+b^L$; $X_j<=sum_i a_i^U X_i+b^U$
  具体: $l_j<=X_j<=u_j$
  Crossing ReLU: 上界$Y<=lambda(X-l)$, $lambda=u/(u-l)$
  下界$Y>=alpha X$, $alpha in[0,1]$可优化
  #cR[Back-sub]: 递归展开至输入层
  #cW[负系数用opposite bound!]
]

#cbox(title: [$alpha$-$beta$-CROWN])[
  $alpha$: ReLU下界斜率$in[0,1]$,梯度优化
  $beta$: Lagrange乘子$>=0$,编码split
  #cG[关键]: $alpha,beta$只影响Tightness,不影响Soundness!
  任意合法值都Sound,只是松紧不同
]

#cbox(title: [MILP编码])[
  Crossing ReLU: $a in{0,1}$
  $y>=x$, $y<=x-l(1-a)$, $y<=u dot a$, $y>=0$
  $a=1$: $y=x$; $a=0$: $y=0$
  #cW[$ell_2$球是二次约束,MILP不完备!]
  浮点: 理论Sound$neq$硬件Sound
]

#cbox(title: [Branch & Bound])[
  1. Bound: DeepPoly/CROWN算bounds
  2. $l>0$⟹SAFE; $u<0$⟹UNSAFE
  3. Branch: split unstable ReLU
  4. 递归两子问题
  KKT: $(max f "s.t." g<=0)<=max_x min_(beta>=0)[f+beta g]$
  #cB[Weak Duality]: $max min<=min max$
]

= 4. Randomized Smoothing

#cbox(title: [Smoothed Classifier])[
  $G(x)=arg max_c PP_(epsilon tilde cal(N)(0,sigma^2 I))[F(x+epsilon)=c]$
  Base $F$可脆弱,Smooth后$G$有认证保证
  #cR[定理deterministic,采样估计probabilistic!]
]

#cbox(title: [认证半径])[
  $R=sigma dot Phi^(-1)(underline(p_A))$, 要求$p_A>0.5$
  $Phi^(-1)$: 标准正态CDF逆(probit)
  #cW[$sigma$↑不一定$R$↑!] 间接效应:$p_A$↓
  存在最优$sigma^*$需tuning
]

#cbox(title: [两阶段采样])[
  Stage1 ($n_0 approx 100$): 猜top class $c_A$
  Stage2 ($n approx 10^5$): 估计$p_A$,Binomial CI
  若$underline(p_A)<=0.5$: ABSTAIN
  #cR[复杂度$O(n_"samples")$] 与网络结构解耦!
]

#cbox(title: [为何限$ell_2$?])[
  Gaussian旋转不变→等概率面是球→$ell_2$解析
  其他: Laplace→$ell_1$, Uniform→$ell_infinity$无闭式
  #cW[别混淆DP]: DP中Gaussian对$ell_2$敏感度; RS中对$ell_2$认证
]

= 5. Certified Training

#cbox(title: [PGD-AT vs Certified])[
  PGD-AT: 具体对抗样本,Heuristic
  Certified: 抽象区域$gamma(f^\#(S(x)))$,Sound
  #cR[Paradox]: Box松但smooth好优化; DeepPoly紧但discontinuous难训
]

#cbox(title: [SABR/COLT])[
  SABR: 中间层冻结,后续层PGD (解决投影困难)
  COLT: DeepPoly传播至k层,后续PGD
  投影问题: $ell_infinity$=clip; DeepPoly需QP
]

= 6. DP 隐私

#cbox(title: [$epsilon$-DP])[
  $PP(M(D) in S)<=e^epsilon PP(M(D') in S)$
  $e^epsilon approx 1+epsilon$ (小$epsilon$时)
  Laplace: $f(D)+"Lap"(Delta_1/epsilon)$
]

#cbox(title: [$(epsilon,delta)$-DP])[
  $PP(M(D) in S)<=e^epsilon PP(M(D') in S)+delta$
  $delta$: 尾部质量界 #cW[非"泄露概率"!]
  Gaussian: $sigma>=Delta_2 sqrt(2ln(1.25/delta))/epsilon$
]

#cbox(title: [三大性质])[
  *Post-processing*: $g compose M$仍DP
  *Composition*: $(epsilon_1+epsilon_2,delta_1+delta_2)$
  *Subsampling*: 采样比$q$→$(q epsilon,q delta)$
  Advanced: $T$步后$epsilon_"tot"=O(sqrt(T)epsilon)$
]

#cbox(title: [DP-SGD])[
  1. 梯度裁剪: $g_"clip"=g dot min(1,C/||g||_2)$
  2. 加噪: $g_"noisy"=g_"avg"+cal(N)(0,sigma^2 C^2/L^2)$
  裁剪控制$Delta_2<=C$; $sigma=(C sqrt(2ln(1.25/delta)))/(L epsilon)$
]

#cbox(title: [PATE])[
  M教师独立训练,投票+加噪标注公开数据
  #cR[argmax前加噪!] $Delta_1=2$(非$|Y|$)
  $arg max(n_j(x)+"Lap"(2/epsilon))$
  每次查询消耗$epsilon$,总预算累积
]

#cbox(title: [DP vs RS 对偶])[
  *DP*: 使分布不可区分 $P[M(D)] approx P[M(D')]$
  *RS*: 使预测可区分 $P[G(x)=c]>>$其他
  相同工具(噪声,指数界),#cR[相反目标!]
]

= 7. Privacy Attacks

#cbox(title: [攻击层次])[
  Attribute Inference: 推断敏感属性(无需membership!)
  Data Extraction: 逐字记忆(K-extractable)
  MIA: 判断是否在训练集
  Dataset Inference: 聚合统计检验
  Gradient Inversion: 从梯度反推数据(FL)
]

#cbox(title: [MIA])[
  Shadow Model: 训练K个影子,训练攻击分类器
  LiRA: $log(P(ell|x in D)/P(ell|x in.not D))$
  Min-K% Prob: 最低K个token概率均值
  #cW[实际AUC≈0.5-0.7很弱!] 低FPR时TPR仅2%
]

#cbox(title: [Gradient Inversion])[
  $x^*=arg min_x ||nabla_theta cal(L)(x,y)-nabla_"obs"||^2+R(x)$
  Prior: TV(图像), Perplexity(文本), Entropy(表格)
  FedSGD(单步): BS=1可精确重构
  FedAvg(多步): 需模拟多步轨迹,更难
]

= 8. Fairness

#cbox(title: [Individual])[
  $(D,d)$-Lipschitz: $D(M(x),M(x'))<=d(x,x')$
  等价于Robustness: $forall delta in BB_S(0,1/L): M(x)=M(x+delta)$
]

#cbox(title: [Group])[
  *Dem Parity*: $PP(hat(Y)=1|S=0)=PP(hat(Y)=1|S=1)$
  *Equal Opp*: 上式条件于$Y=1$
  *Eq Odds*: 条件于$Y=0$和$Y=1$都成立
  ⟺ $hat(Y) perp S|Y$ (条件独立)
]

#cbox(title: [$Delta_"EO"$])[
  $Delta_"EO"=|"FPR"_0-"FPR"_1|+|"TPR"_0-"TPR"_1|$
  定理: $Delta_"EO"(g)<=2 dot "BA"(h^*)-1$
  $h^*$: 最优对抗器预测敏感属性
]

= 9. Logic & Watermark

#cbox(title: [Logic→Loss])[
  $T(phi)(x)=0 arrow.l.r.double x models phi$
  $t_1<=t_2$: $max(0,t_1-t_2)$
  $t_1=t_2$: $(t_1-t_2)^2$
  $phi and psi$: $T(phi)+T(psi)$
  $phi or psi$: $T(phi) dot T(psi)$
  #cW[不支持量词!] $forall$用max近似
]

#cbox(title: [Red-Green Watermark])[
  hash(context)+key→分Green/Red
  Generate: Green token加$delta$偏置
  Detect: 统计Green比例,二项检验 #cR[无需LLM!]
  $p$-value$<alpha$→有水印
]

#cbox(title: [ITS/SynthID])[
  ITS: 期望不改分布,但确定性输出
  SynthID: Distortion-Free+Non-Deterministic
  Tournament采样: 高G值token更易赢
]

#cbox(title: [Contamination])[
  Data污染: benchmark在训练集(背答案)
  Task污染: 针对任务优化(学格式)
  检测: N-gram(L1), Perplexity(L2), Completion(L3)
]

= 10. Post-Training

#cbox(title: [Quantization Attack])[
  FP32正常,INT8恶意
  Box约束$[w_"low",w_"high"]$使量化值不变
  #cW[检测盲点]: 检测FP32,部署INT8
]

#cbox(title: [Fine-Tuning Attack])[
  $cal(L)=cal(L)_"clean"(theta)+lambda cal(L)_"attack"(theta-nabla cal(L)_"user")$
  现在安全,用户微调后恶意
  需Hessian(二阶导),计算昂贵
]

= 11. 易错点

#cbox(title: [陷阱速查])[
  MILP复杂度$O(2^k)$, $k$=Crossing数!
  RS定理deterministic,估计probabilistic
  $sigma$↑不一定$R$↑ ($p_A$会↓)
  GCG用梯度筛选,不是更新!
  $n_0$(猜类别~100) vs $n$(估概率~100k)
  PATE: argmax前加噪,$Delta_1=2$
  Tighter$neq$更好训练 (Box好训)
  Back-sub负系数用opposite bound
  $ell_infinity$约束⟹$ell_2$约束
  MILP对$ell_2$球不完备(二次约束)
  MIA AUC≈0.5-0.7很弱
  PGD$neq$CW: 目标函数不同
  FGSM必在$ell_infinity$边界
  FedSGD比FedAvg更易反演
]

#cbox(title: [公式记忆])[
  RS半径: $R=sigma Phi^(-1)(p_A)$
  Gaussian DP: $sigma=(C sqrt(2ln(1.25/delta)))/(L epsilon)$
  PGD: $x'=Pi_(BB_epsilon)(x+alpha"sign"(nabla cal(L)))$
  FGSM: $eta=epsilon dot "sign"(nabla cal(L))$
  DeepPoly上界: $Y<=u/(u-l)(X-l)$
  Box传播: $[W^+ l+W^- u+b, W^+ u+W^- l+b]$
]

= Appendix

#cbox(title: [范数&分布])[
  $||x||_p=(sum|x_i|^p)^(1/p)$; $||x||_infinity=max|x_i|$
  $cal(N)=(2pi)^(-d/2)|Sigma|^(-1/2)exp(-1/2(x-mu)^top Sigma^(-1)(x-mu))$
  $"Lap"=1/(2b)exp(-|x-mu|/b)$
]

#cbox(title: [Softmax&CE])[
  $sigma(z)_i=e^(z_i)/sum_j e^(z_j)$
  $"CE"(z,y)=-log sigma(z)_y=-z_y+log sum_j e^(z_j)$
]

#cbox(title: [导数])[
  $diff_x b^top x=b$; $diff_x x^top x=2x$
  $diff_x x^top A x=(A+A^top)x$
  $diff_x ||A x-b||_2^2=2A^top(A x-b)$
]

#cbox(title: [不等式])[
  Cauchy-Schwarz: $angle.l x,y angle.r<=||x||_2||y||_2$
  Hölder: $||x dot y||_1<=||x||_p||y||_q$, $1/p+1/q=1$
  Jensen: $g$凸⟹$g(EE[X])<=EE[g(X)]$
  Chebyshev: $PP(|X-EE[X]|>=epsilon)<=VV[X]/epsilon^2$
  Weak Duality: $max_a min_b f<=min_b max_a f$
]

#cbox(title: [MILP编码])[
  $y=|x|$: $y>=x,y>=-x$,$y<=x+2u(1-a),y<=-x+2|l|a$
  $y=max(x_1,x_2)$: $y>=x_1,y>=x_2$,
  $y<=x_1+a(u_2-l_1),y<=x_2+(1-a)(u_1-l_2)$
]

#cbox(title: [概率])[
  $VV(X)=EE[X^2]-EE[X]^2$
  $VV(a X+b Y)=a^2 VV(X)+b^2 VV(Y)+2a b"Cov"$
  Bayes: $P(X|Y)=P(Y|X)P(X)/P(Y)$
  $Phi(z)=PP(cal(N)(0,1)<=z)$; $Phi^(-1)(0.5)=0$
]

#cbox(title: [Matrix])[
  $A^(-1)=mat(a,b;c,d)^(-1)=1/(a d-b c)mat(d,-b;-c,a)$
]

#cbox(title: [Logic])[
  De Morgan: $not(phi and psi)=not phi or not psi$
  Implication: $phi arrow.r.double psi equiv not phi or psi$
  $T(not phi)$: 用De Morgan展开
]
