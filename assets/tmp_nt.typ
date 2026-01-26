// ========================================
// Yellow-Purple Note Theme
// ========================================
#import "tmp_math.typ": * // Math notation

#let summary_project(
  title: "",
  authors: (),
  // Font
  western_font: "Libertinus Serif",
  chinese_serif: (
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "Songti SC",
    "SimSun",
  ),
  chinese_sans: (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Microsoft YaHei",
  ),
  chinese_italic: (
    "FangSong",
    "Kaiti SC",
    "STKaiti",
  ),
  code_font: "Fira Code",
  // Font size
  base_size: 9pt,
  heading1_size: 1.5em,
  heading2_size: 1.25em,
  heading3_size: 1.15em,
  math_size: 0.95em,
  code_size: 0.9em, // 🔥 代码字号调大
  // Spacing间距
  par_spacing: 0.5em,
  par_leading: 0.5em,
  heading_above: 0.78em,
  heading_below: 0.4em,
  // Colors
  primary_color: rgb("#8B5CF6"),
  secondary_color: rgb("#F59E0B"),
  accent_color: rgb("#A78BFA"),
  code_bg: rgb("#FFFEF5"), // 🔥 更柔和的浅黄
  def_color: rgb("#7C3AED"),
  note_color: rgb("#78716C"), // 🔥 改为暖灰色
  // 页面配置
  margin: (x: 0.5cm, y: 0.8cm),
  body,
) = {
  set document(
    title: title,
    author: authors.map(a => a.name),
  )

  set page(
    paper: "a4",
    margin: margin,
    numbering: "1",
    number-align: center,
  )

  // 高亮语法：==text== 变成金黄色高亮
  show regex("==([^=]+)=="): it => {
    let content = it.text.slice(2, -2)  // 去掉前后的 ==
    highlight(fill: rgb("#FEF3C7"), text(fill: rgb("#92400E"))[#content])
  }

  // 字体设置（支持中文）
  set text(
    font: (western_font, ..chinese_serif),
    size: base_size,
    lang: "en", // zh
    region: "cn",
  )

  // 中文粗体：使用黑体
  show strong: it => {
    text(
      font: (western_font, ..chinese_sans),
      weight: "bold",
      it,
    )
  }

  // 只对中文应用楷体，英文保持默认 italic
  show emph: it => {
    // 不设置 style: "italic"，让字体自己决定
    // Libertinus Serif 的 italic 变体会被自动选择
    // 楷体本身就是"斜体风格"，不需要额外的 italic style
    text(
      font: (western_font, ..chinese_italic),
    )[#it]
  }


  // 段落设置
  set par(
    justify: true,
    leading: par_leading,
    spacing: par_spacing,
  )

  // 标题样式
  set heading(numbering: "1.1")

  show heading.where(level: 1): it => {
    //pagebreak(weak: true)
    v(heading_above, weak: true)
    block(
      width: 100%,
      fill: primary_color.lighten(92%),
      inset: (x: 0.8em, y: 0.6em),
      radius: 3pt,
      stroke: (left: 3pt + primary_color),
      [
        #set text(
          font: (western_font, ..chinese_sans),
          size: heading1_size,
          weight: "bold",
          fill: primary_color,
        )
        #it
      ],
    )
    v(heading_below, weak: true)
  }

  show heading.where(level: 2): it => {
    v(heading_above * 0.8, weak: true)
    set text(
      font: (western_font, ..chinese_sans),
      size: heading2_size,
      weight: "bold",
      fill: primary_color.darken(10%),
    )
    it
    v(heading_below * 0.8, weak: true)
  }

  show heading.where(level: 3): it => {
    v(heading_above * 0.6, weak: true)
    set text(
      font: (western_font, ..chinese_sans),
      size: heading3_size,
      weight: "bold",
      fill: primary_color.darken(5%),
    )
    it
    v(heading_below * 0.6, weak: true)
  }

  // 数学公式
  show math.equation.where(block: true): it => {
    v(0.4em, weak: true)
    set text(size: math_size)
    it
    v(0.4em, weak: true)
  }

  show math.equation.where(block: false): it => {
    set text(size: math_size)
    it
  }

  // 列表样式
  set list(tight: true, indent: 0.8em, body-indent: 0.5em)
  set enum(tight: true, indent: 0.8em, body-indent: 0.5em)

  // 标题页
  align(center)[
    #v(2fr)
    #text(size: 2.2em, weight: "bold", fill: primary_color)[#title]
    #v(0.5em, weak: true)
    #text(size: 1em, style: "italic", fill: accent_color.darken(30%))[
      Lecture Notes
    ]
    #v(0.5em, weak: true)
    #for author in authors [
      #text(size: 1.1em)[#author.name] \
    ]
    #v(3fr)
  ]

  // pagebreak()

  // 新目录部分（双栏紧凑版）
{
  // 样式覆盖
  show outline.entry: it => {
    set text(size: 0.9em, fill: black)
    v(0.45em, weak: true)
    it
  }
  // 多栏布局
  columns(3, gutter: 1em)[
    #outline(
      title: [目录 / Contents],
      depth: 3,
      indent: 1em,
    )
  ]
}

  // pagebreak()

  body
}

// ========================================
// 内容环境
// ========================================

/// 通用内容盒子（保持左边框设计）
#let cbox(
  title: none,
  color: rgb("#8B5CF6"),
  bg_lighten: 95%,
  border_left: 2.5pt,
  body,
) = {
  block(
    width: 100%,
    fill: color.lighten(bg_lighten),
    stroke: (left: border_left + color),
    inset: (left: 0.7em, right: 0.7em, y: 0.5em),
    radius: (right: 3pt),
    breakable: true,
    [
      #if title != none [
        #text(weight: "semibold", fill: color.darken(10%), size: 0.95em)[
          #title
        ]
        #v(0.3em, weak: true)
      ]
      #body
    ],
  )
}

/// 定义环境（深紫色）
#let definition(title: "Definition", body) = {
  cbox(
    title: title,
    color: rgb("#7C3AED"),
    bg_lighten: 96%,
    border_left: 3pt,
    body,
  )
}

/// 定理环境（黄色）
#let theorem(title: "Theorem", body) = {
  cbox(
    title: title,
    color: rgb("#F59E0B"),
    bg_lighten: 95%,
    border_left: 3pt,
    body,
  )
}

/// 示例环境（浅紫色）
#let example(title: "Example", body) = {
  cbox(
    title: title,
    color: rgb("#A78BFA"),
    bg_lighten: 98%,
    border_left: 2pt,
    body,
  )
}

// ========================================
// 🔥 改进：算法/代码环境
// ========================================

/// 算法环境（更大字体 + 柔和浅黄）
#let algorithm(title: "Algorithm", body) = {
  v(0.5em, weak: true)
  block(
    width: 100%,
    fill: rgb("#FFFEF5"), // 🔥 极浅米黄色（柔和）
    stroke: (
      left: 3pt + rgb("#EAB308"), // 黄色左边框
      rest: 1pt + rgb("#FEF3C7"),
    ),
    inset: (x: 1em, y: 0.8em),
    radius: 3pt,
    breakable: true,
    [
      #if title != none [
        #text(
          weight: "bold",
          fill: rgb("#A16207"), // 深黄褐色标题
          size: 1em, // 🔥 标题正常大小
        )[⚙️ #title]
        #v(0.4em)
        #line(length: 100%, stroke: 0.5pt + rgb("#FDE68A"))
        #v(0.35em)
      ]
      #set text(
        font: ("Fira Code", "Noto Sans Mono CJK SC"),
        size: 0.95em, // 🔥 代码字体调大！
        fill: rgb("#1C1917"), // 深色文字
      )
      #set par(leading: 0.6em, justify: false)
      #body
    ],
  )
  v(0.5em, weak: true)
}

/// 行内代码（紧凑柔和标签）
#let code(body) = box(
  fill: rgb("#FEF9E7"), // 🔥 更柔和的浅黄
  stroke: 0.5pt + rgb("#E5E7EB"), // 浅灰边框（低调）
  inset: (x: 0.4em, y: 0.18em),
  radius: 3pt,
  baseline: 0.15em,
  text(
    font: ("Fira Code", "Noto Sans Mono CJK SC"),
    size: 0.92em, // 🔥 行内代码也调大
    fill: rgb("#78350F"), // 深棕色
  )[#body],
)

/// 代码块（紫色左边框 + 浅灰背景）
#let codeblock(body) = {
  v(0.4em, weak: true)
  block(
    width: 100%,
    fill: rgb("#F9FAFB"), // 浅灰背景
    stroke: (left: 2.5pt + rgb("#A78BFA")), // 紫色左边框
    inset: (x: 0.8em, y: 0.6em),
    radius: 2pt,
    breakable: true,
    {
      set text(
        font: ("Fira Code", "Noto Sans Mono CJK SC"),
        size: 0.92em, // 🔥 调大
      )
      set par(leading: 0.55em, justify: false)
      body
    },
  )
  v(0.4em, weak: true)
}

// ========================================
// 🔥 改进：Note 环境（差异化设计）
// ========================================

/// Note 环境：暖灰色 + 双边距紧凑设计
/// 与紫色/黄色形成对比，用于辅助说明
#let note(body) = {
  v(0.3em, weak: true)
  block(
    width: 92%, // 🔥 双边距效果
    inset: (x: 1em, y: 0.6em),
    fill: rgb("#FAF9F7"), // 🔥 极浅暖灰/米色
    stroke: 1pt + rgb("#E7E5E4"), // 浅灰边框（四周）
    radius: 4pt,
    breakable: true,
    [
      #set text(
        size: 0.9em,
        fill: rgb("#57534E"), // 🔥 暖灰色文字（与紫色形成对比）
      )
      #set par(leading: 0.5em)
      *💡* #body
    ],
  )
  v(0.3em, weak: true)
}

/// 备选：提示环境（绿色系，如果需要更多颜色层次）
#let tip(body) = {
  block(
    width: 100%,
    fill: rgb("#F0FDF4"), // 极浅绿
    stroke: (left: 2.5pt + rgb("#22C55E")),
    inset: (left: 0.7em, right: 0.7em, y: 0.5em),
    radius: (right: 3pt),
    breakable: true,
    [
      #set text(size: 0.92em, fill: rgb("#166534"))
      *✅* #body
    ],
  )
}

/// 备选：警告环境（红色系）
#let warning(body) = {
  block(
    width: 100%,
    fill: rgb("#FEF2F2"), // 极浅红
    stroke: (left: 2.5pt + rgb("#EF4444")),
    inset: (left: 0.7em, right: 0.7em, y: 0.5em),
    radius: (right: 3pt),
    breakable: true,
    [
      #set text(size: 0.92em, fill: rgb("#991B1B"))
      *⚠️* #body
    ],
  )
}

