
#import "../assets/tmp_nt.typ": *

// Configure the document with custom settings
#show: summary_project.with(
  title: "25HS_RTAI_Note",
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
  primary_color: rgb("#997933"),
  secondary_color: rgb("#663399"),

  // Compact margins
  margin: (x: 1.25cm, y: 1.25cm),
)

// ========== CONTENT BEGINS ==========
#pagebreak()

= Verification
