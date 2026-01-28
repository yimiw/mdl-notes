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