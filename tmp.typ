#let project(title: "", authors: (), date: none, body) = {
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: none, number-align: center, flipped: true, margin: 0.25em)
  set text(font: "Libertinus Serif", lang: "en")
  set heading(numbering: none)
  body
}

// 极简box：无背景色，仅左边框
#let cbox(title: none, content) = block(
  stroke: (left: 1.8pt + rgb("#4e3269") ),
  inset: (left: 4pt, rest: 2pt),
  width: 100%,
  if title != none { [*#title*: ] } + content
)