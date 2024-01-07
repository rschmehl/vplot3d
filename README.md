# vplot3d

This library extends the Python toolkit mplot3d to programmatically generate 3D vector diagrams for SVG output.
The following diagram objects can be used

- Points,
- Lines and circular arcs,
- Vectors and arc measures.

Points and arrowheads (for vectors and arc measures) are generated as SVG markers.
Because Spyder's SVG renderer does not support markers, these are not drawn in the IPython console window. They do show in a webbrowser or in Inkscape.

## Diagram layout and

- `ZOOM`