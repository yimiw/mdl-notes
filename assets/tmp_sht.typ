#import "tmp_math.typ": * // Math notation

#let project(title: "", authors: (), date: none, body) = {
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: none, number-align: center, flipped: true, margin: 0.25em)
  set text(font: "Libertinus Serif", lang: "en")
  set heading(numbering: none)
  body
}

// 极简box：无背景色，仅左边框
#let cbox(title: none, content) = block(
  stroke: (left: 1.8pt + rgb("#4e3269")),
  inset: (left: 4pt, rest: 2pt),
  width: 100%,
  if title != none { [*#title*: ] } + content,
)

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