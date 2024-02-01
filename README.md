# vplot3d

This library extends the Python toolkit `mplot3d` to programmatically generate 3D vector diagrams in SVG format. The following diagram objects can be used

- Points,
- Lines and circular arcs,
- Vectors and arc measures,
- Annotations.

Points and arrowheads (for vectors and arc measures) are generated as native SVG markers to facilitate later postprocessing of the diagram in vector drawing tools, like Illustrator and Inkscape. For precise positioning of arrowheads, the underlying line or polyline is shortened. This algorithm is one of the key contributions of `vplot3d`.

## System requirements

- Matplotlib 3.3.0

Optional

- Inkscape (to render the generated SVG in the IDE and for Latex postprocessing)
- Scour (to optimize SVG output)
- pdflatex (for Latex postprocessing, to include annotations and automatically render the Latex code)

## Installation

To use the library for sketching a vector diagram, set the variable `lib_path` at the start of your code to the directory where you have placed `vplot3d.py`.

    lib_path = Path('/home/user/projects/vplot3d')
    sys.path.append(str(lib_path))

The installation path `lib_path` is also needed for the later postprocessing and display of the diagram.

## Diagram layout and 3D perspective

The diagram will be generated as an SVG file. You have to specify the width and height of this SVG diagram in pixels using the `figsize` function of `vplot3d`:

    rcParams['figure.figsize'] = figsize(width_in_pixels, height_in_pixels)

When displaying the SVG file in a web browser or including it in html without explicit dimensions, these dimensions are used. But as a native vector format, SVG is also scalable to any dimensions without quality loss.

At the start of your drawing you also need to define the anticipated 3D data range:

    set_xlim3d([xmin, xmax])
    set_ylim3d([ymin, ymax])
    set_zlim3d([zmin, zmax])

The limiting values define the position of the 3D-diagram in the 2D SVG canvas. For the convenience of the user, the distance of the viewer to the object can be modified by

    ZOOM = value

where the default value of 1 depicts the original data range, a value > 1 zooms out and a value < 1 zooms in.

> [!TIP]
> Using constant size parameters of graphical objects (line width, arrowhead size, etc) across all diagrams of a document, while adjusting only the data range, zoom value and figure size per diagram leads to a uniform graphical representation.

The perspective of the 3D-diagram can be set in the usual way by calling `view_init` with desired elevation and azimuth angle values:

    elev =  30
    azim = -60
    view_init(elev, azim)

> [!CAUTION]
> The library has only been tested for orthographic projection so far. A different projection method could affect the shortening of arrowheads.

## Postprocessing

Because Spyder's SVG renderer does not support markers, these are not drawn in the IPython console window. They do show in a webbrowser or in Inkscape. The included postprocessing with Inkscape, or Inscape-Latex-Inkscape generates a PNG file for output in the IDE's renderer.

## Stepwise diagram buildup or animation

To buildup a diagram in several steps, objects can be added, removed or updated and the current state of the diagram saved with a separate filename.

In this way , it should also be possible to create animations by updating the diagram in an animation look, updating, for example, the position of an object. The generated PNG files could then be easily converted to a video file, using ffmpeg.

## Adding new markers

New markers are added in the markers library `data/markers.svg` in the defs section, using a unique `id`. Only for arrowhead markers, the Marker class dictionary `deltas` needs to be expanded by the line-shortening value matching the new marker path.

## Gallery

![](examples/test.svg)
![](examples/kite_kinematics_3d.svg)
![](examples/kite_kinematics_3d_a.svg)