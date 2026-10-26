# Results dashboard (v1.0.0)

`urban-block-types_dashboard.html` is the final dashboard of the study (Cities A–C, 3,850
blocks, local and pooled models). It is a single self-contained file built with
`blockmorph dashboard` from the results archive v1.0.0.

- View in the browser: https://cdml-lab.github.io/urban-block-morphology/
- Download (release v1.0.0): https://github.com/cdml-lab/urban-block-morphology/releases/download/v1.0.0/urban-block-types_dashboard.html

Requirements:
- **Internet**: the interactive map loads deck.gl 9.0.35 (unpkg.com) and the Lato font (Google Fonts).
  Offline, the maps fall back to a static 2D view and the Types and Model quality views work fully.
- **WebGL 2**: needed for the 3D view; without it the 2D map is shown.
- Size about 6 MB; current Chrome, Edge, Firefox or Safari.
