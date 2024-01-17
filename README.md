# vplot3d

This library extends the Python toolkit mplot3d to programmatically generate 3D vector diagrams for SVG output.
The following diagram objects can be used

- Points,
- Lines and circular arcs,
- Vectors and arc measures.

Points and arrowheads (for vectors and arc measures) are generated as SVG markers.
Because Spyder's SVG renderer does not support markers, these are not drawn in the IPython console window. They do show in a webbrowser or in Inkscape.

## System requirements

- Matplotlib 3.3.0

If you want to include Latex code

## Installation

To use the library for sketching a vector diagram, set the variable `lib_path` at the start of your code to the directory where you have placed `vplot3d.py`.

    lib_path = Path('/home/user/projects/vplot3d')
    sys.path.append(str(lib_path))

The installation path `lib_path` is also needed for the later postprocessing and display of the diagram.

## Diagram layout and 3D perspective

The diagram will be generated as an SVG file. You have to specify the width and height of this SVG diagram in pixels using the `figsize` function of `vplot3d`:

    rcParams['figure.figsize'] = figsize(width_in_pixels, height_in_pixels)

When displaying the SVG file in a web browser or including it in html without explicit dimensions, these dimensions are used. But as a native vector format, SVG is also scalable to any dimensions without quality loss.

[!TIP]
Using constant size parameters across all diagrams of a document, while adjusting only the data range and figure size per diagram should lead to a uniform graphical representation.

At the start of your drawing you also need to define the anticipated 3D data range:

    set_xlim3d([xmin, xmax])
    set_ylim3d([ymin, ymax])
    set_zlim3d([zmin, zmax])

The limiting values define the position of the 3D-diagram in the 2D SVG canvas. For the convenience of the user, the distance of the viewer to the object can be modified by 

    ZOOM = value

where the default value of 1 depicts the original data range, a value > 1 zooms out and a value < 1 zooms in.

The perspective of the 3D-diagram can be set by calling `view_init` with desired elevation and azimuth angle values:

    elev =  30
    azim = -60
    view_init(elev, azim)

I recommend using an orthographic projection of the 3D-diagram onto the 2D canvas (only tested the library for this).