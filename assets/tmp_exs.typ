// ========================================
// RTAI Exercise Template - 习题集模板
// ========================================
#import "tmp_math.typ": *

// ========== 超参数配置 ==========
#let num_columns = 2  // 栏数（可改为 3）
#let page_margin = (x: 0.6cm, y: 0.8cm)
#let base_font_size = 8pt

// ========== 颜色配置 ==========
#let primary = rgb("#3a58ed")
#let secondary = rgb("#78716C")  // 暖灰色

// ========== 问题样式组件（接受 content 类型）==========
#let prob(title) = block(
  width: 100%,
  fill: rgb("#F5F5F4"), // 浅灰背景
  inset: 5pt,
  radius: 2pt,
  [#text(weight: "bold", size: 0.9em)[#title]],
)

#let tip(body) = block(
  width: 100%,
  fill: rgb("#ECFDF5"),
  inset: 4pt,
  radius: 2pt,
  stroke: (left: 2pt + rgb("#10B981")),
  text(size: 0.9em)[#body],
)

// ========== 文档设置函数 ==========
#let exercise_doc(title: "", body) = {
  set document(title: title, author: ("RTAI Course",))
  set page(paper: "a4", margin: page_margin, numbering: "1", number-align: center)
  set text(font: ("Libertinus Serif", "Songti SC"), size: base_font_size, lang: "en")
  set par(justify: true, leading: 0.4em, spacing: 0.4em)
  set list(tight: true, indent: 0.5em, body-indent: 0.3em)
  set enum(tight: true, indent: 0.5em, body-indent: 0.3em)
  set heading(numbering: "1.1")

  show heading.where(level: 1): it => {
    v(0.5em, weak: true)
    block(
      width: 100%,
      fill: primary.lighten(90%),
      inset: (x: 0.5em, y: 0.35em),
      radius: 2pt,
      stroke: (left: 2.5pt + primary),
      text(font: ("Libertinus Serif", "PingFang SC"), size: 1.2em, weight: "bold", fill: primary)[#it],
    )
    v(0.3em, weak: true)
  }

  show heading.where(level: 2): it => {
    v(0.3em, weak: true)
    text(size: 1.05em, weight: "bold", fill: primary.darken(10%))[#it]
    v(0.15em, weak: true)
  }

  show heading.where(level: 3): it => {
    v(0.2em, weak: true)
    text(size: 0.95em, weight: "bold")[#it]
    v(0.1em, weak: true)
  }

  show math.equation.where(block: true): it => {
    v(0.2em, weak: true)
    set text(size: 0.9em)
    it
    v(0.2em, weak: true)
  }

  // 标题
  align(center)[
    #v(0.8em)
    #text(size: 1.8em, weight: "bold", fill: primary)[#title]
    #v(0.4em)
  ]

  // 双栏正文
  columns(num_columns, gutter: 0.7em)[#body]
}
