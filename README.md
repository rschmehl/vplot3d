# vplot3d - 3D vector diagrams in SVG format

The library extends the Python toolkit `mplot3d` to programmatically generate 3D vector diagrams in SVG format with a minimum of drawing-related commands. The user can focus on the geometrical or physical problem instead of takling its visualization, delegating the bulk of drawing-related code to the library. The following 3D objects can be instantiated:

- Points,
- Lines and circular arcs,
- Vectors and arc measures,
- Polygons,
- Surface meshes,
- Annotations.

Points and arrowheads (for vectors and arc measures) are generated as native SVG markers to facilitate later postprocessing of the diagram in vector drawing tools, like Illustrator and Inkscape. For the precise positioning of arrowheads, the underlying line or polyline is shortened. This algorithm is one of the key contributions of `vplot3d`.

## System requirements

The library can be used with two options to generate SVG files: a call to `save_svg` generates plain SVG output, while a call to `save_svg_tex` generates plain SVG output and pipes this through Inkscape, using its PDF+Latex output option, and then pdflatex to compile into PDF, with a final conversion back to SVG. This post-processing is useful to render Latex code of mathematical symbols and expressions, using pdflatex.

For using  `save_svg_tex`, the following two executables need to be installed and in the search path:

- [Inkscape](https://inkscape.org/) (free and open-source vector graphics editor)
- [pdflatex](https://www.tug.org/texlive/) (Latex typesetting program)

## Installation

1. Locally clone the repository or download it as zip-file and unpack it.
2. Go to the root-folder (where the file `pyproject.toml` resides.)
3. Create a virtual environment (in this case `.venv`):
   ```bash
   python -m venv .venv
   ```
4. Activate the virtual environment: on Linux
   ```bash
   source .venv/bin/activate
   ```
   On Windows (Command Prompt)
   ```bash
   .venv\Scripts\activate
   ```
   On Windows (PowerShell)
   ```bash
   .\.venv\Scripts\Activate
   ```
5. Locally install the package and its dependencies:
   ```bash
   pip install -e .
   ```
   The option `-e` ensures editable mode.
6. Now you are ready to use the library. Open your favorite development environment and start coding. The `examples` folder contains several Python files with implemented examples demonstrating the features of `vplot3d`.
6. Once you are finished you can deactivate the virtual environment.
   ```bash
   deactivate
   ```

{% tabs %}

{% tab title="Windows" %} Here are the instructions for Windows {% endtab %}

{% tab title="OSX" %} Here are the instructions for macOS {% endtab %}

{% tab title="Linux" %} Here are the instructions for Linux {% endtab %}

{% endtabs %}

> [!TIP]
> The example `kite.py` shows the definition of a more complex composite object in a separate, user-specified Python file, `kiteV3.py`. 

## Use

Once configured and imported, `vplot3d` provides the following interface:

 - an `init_view` function, to initialize the 3D scenario and a certain perspective, programatically generated 3D vector diagrams 
 - various constructors and utility functions to generate and manipulate 3D objects, and 
 - diffenent `save_svg` and `save_svg_tex` 

### Configuration

A set of baseline default parameters is read initially from the package configuration file `config/vplot3d.yaml`. Via environment variable `CONF_PATH`, a user can control the path from where the file `vplot3d.yaml` with superseding definitions is read. If `CONF_PATH` is not set, a file `vplot3d.yaml` in the current working directory is searched. When not located the file is ignored.

### Import

### Initialization

For initializing the 3D scene, `vplot3d` provides the `init_view` function:
```
init_view(width=600, height=600,
          xmin=0, xmax=1, ymin=0, ymax=1, zmin=-0.3, zmax=1.5,
          zoom=1.5, elev=30, azim=-60)
```

The `width` and `height` parameters specify the size of the SVG-file in terms of pixels. 
The `xmin`, `xmax`, `ymin`, `ymax`, `zmin` and `zmax` parameters define the expected value ranges in the model space.
The `zoom` parameter specifies the viewing distance to the model.
The `elev` and `azim` parameters specify the elevation and azimuth angles of the applied perspective of the model.

> [!CAUTION]
> The library has only been tested for orthographic projection so far. A different projection method could affect the shortening of arrowheads.

## Output

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
![](examples/kite.svg)