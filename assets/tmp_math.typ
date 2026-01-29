// Mathematical Notation Definitions

// Operators
#let opt = text([$*$], fill: red)
#let score = $op("score")$
#let softmax = $op("softmax")$
#let count = $op("count")$
#let argmin = $op("arg min")$
#let argmax = $op("arg max")$
#let trans = $op("trans")$
#let emit = $op("emit")$
#let bos = text(smallcaps("bos"))
#let eos = text(smallcaps("eos"))

// Blackboard bold sets
#let RR = $bb(R)$
#let NN = $bb(N)$
#let EE = $bb(E)$
#let PP = $bb(P)$
#let VV = $bb(V)$

// Vector and matrix notation
#let vec(x) = $bold(#x)$
// #let mat(x) = $bold(#x)$ <- 2026.01.29改动

// Common symbols
#let doteq = $:=$
#let grad = $nabla$
#let hess = $nabla^2$
#let pm = $plus.minus$
#let neq = $eq.not$
#let sim = $tilde.op$
#let iff = $"iff"$

//plattue
#let c1 = rgb("#FFF3CD") 
// 🧪highlight:
#let hl(body) = highlight(fill: rgb("#FEF3C7"), text(fill: rgb("#92400E"))[#body])
