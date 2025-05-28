import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from vplot3d.vplot3d import (
    init_view, Point, Line, Vector, save_svg_tex, ArcMeasure, Polygon
)
def compute_positions(chord, ld, angle_deg, x_cp):
    angle_rad = np.radians(angle_deg)
    LE = np.array([0, ld, 0])
    TE = np.array([chord * np.cos(angle_rad), chord * np.sin(angle_rad) + ld, 0])
    B = np.array([0, 0, 0])
    CP = LE + (TE - LE) * x_cp
    return LE, TE, B, CP

def aero_coeffs_polynomial(alpha):
    c0, c1, c2 = 0.1, 5.87, -7.04
    d0, d1, d2 = 0.12, 0.3, 0.22
    CL = c0 + c1 * alpha + c2 * alpha**2
    CD = d0 + d1 * alpha + d2 * alpha**2
    return CL, CD

def compute_aero_forces(alpha_wind_rad, LE, TE, AR):
    d_foil = TE[:2] - LE[:2]
    theta_foil = np.arctan2(d_foil[1], d_foil[0])
    alpha = alpha_wind_rad - theta_foil
    CL, CD = aero_coeffs_polynomial(alpha)
    airflow_dir = np.array([np.cos(alpha_wind_rad), np.sin(alpha_wind_rad), 0])
    lift_dir = np.array([-airflow_dir[1], airflow_dir[0], 0])
    L = CL * lift_dir
    D = CD * airflow_dir
    F = L + D
    return F, L, D, np.degrees(alpha)

def find_alpha_for_cp_alignment(LE, TE, B, CP, AR):
    cp_vec = CP - B
    cp_unit = cp_vec / np.linalg.norm(cp_vec)
    best_alpha_wind = None
    min_angle = np.inf
    best_force = None
    best_alpha_attack = None
    for alpha_wind_deg in np.linspace(-10, 30, 720):
        alpha_wind_rad = np.radians(alpha_wind_deg)
        F, L, D, alpha_attack_deg = compute_aero_forces(alpha_wind_rad, LE, TE, AR)
        F_unit = F / np.linalg.norm(F)
        angle = np.degrees(np.arccos(np.clip(np.dot(F_unit, cp_unit), -1.0, 1.0)))
        if angle < min_angle:
            min_angle = angle
            best_alpha_wind = alpha_wind_deg
            best_force = F
            lift = L
            drag = D
            best_alpha_attack = alpha_attack_deg
    return best_alpha_wind, best_alpha_attack, best_force, min_angle, lift, drag

def rotate_xy(v, angle_rad):
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    v_xy = rot_matrix @ v[:2]
    return np.array([v_xy[0], v_xy[1], v[2]])

def rotate_all(angle, *vectors):
    return [rotate_xy(v, angle) for v in vectors]

PALETTE = {
        "Black": "#000000", "Orange": "#E69F00", "Sky Blue": "#56B4E9",
        "Bluish Green": "#009E73", "Yellow": "#F0E442", "Blue": "#0072B2",
        "Vermillion": "#D55E00", "Reddish Purple": "#CC79A7",
    }
colors = list(PALETTE.values())
def render_vplot3d_diagram(output_path, chord=2.6, ld=8, x_cp=0.32, AR=4.0, angle_depower=0):


    
    dat_path = Path.cwd().parent / 'data'



    # Start building the plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')
    ax.set_axis_off()

    init_view(width=600, height=500,
              xmin=0, xmax=chord*0.2, ymin=0.0, ymax=ld * 1.2, zmin=0, zmax=0,
              zoom=1, elev=90, azim=270)

    LE, TE, B, CP = compute_positions(chord, ld, angle_depower, x_cp)
    cp_vec = CP - B
    angle_to_y = np.arctan2(cp_vec[0], cp_vec[1])

    alpha_wind, alpha_attack, F_align, min_angle, L, D = find_alpha_for_cp_alignment(LE, TE, B, CP, AR)

    alpha_d_rad = np.radians(angle_depower)
    Px = np.array([np.cos(alpha_d_rad), np.sin(alpha_d_rad), 0])
    Py = np.array([np.sin(alpha_d_rad), np.cos(alpha_d_rad), 0])
    v_a = np.array([np.cos(np.radians(alpha_wind)), np.sin(np.radians(alpha_wind)), 0])

    LE, TE, B, CP = rotate_all(angle_to_y, LE, TE, B, CP)
    F_align, L, D, Px, Py, v_a = rotate_all(angle_to_y, F_align, L, D, Px, Py, v_a)

    Polygon.rotated(LE, file=dat_path / 'naca0012.dat',
                    e1=Px, e2=Py, edgecolor='k', facecolor='k',
                    scale=chord, linewidth=1, alpha=0.1, edgecoloralpha=0.4)

    Line(B, LE, zorder=0, scale=1, color="k", linestyle='-', linewidth=2)
    Line(B, TE, zorder=0, scale=1, color="k", linestyle='-', linewidth=2)
    Line(LE, TE - LE, zorder=0, scale=1, color="k", linestyle='--', linewidth=2)
    Line(LE, -TE + LE, zorder=0, scale=0.5, color="k", linestyle=':', linewidth=2)
    Line(TE, (TE - LE) * 0.4, zorder=0, scale=0.05, color="k", linestyle=':', linewidth=2)
    Line(B, CP, zorder=0, scale=1, color="k", linestyle=':', linewidth=2)
    Vector(B, -CP*0.2, zorder=0, scale=1, color="k", linewidth=2)
    ax.annotate3D('$\\lambda_0$', xyz=0.65 * LE + np.array([0.1, 0, 0]), fontsize=12, color='k')

    v_ble = LE - B
    v_ble = -v_ble / np.linalg.norm(v_ble)  # unit vector
    v_perp = np.array([-v_ble[1], v_ble[0], 0])
    Line(LE, v_perp * (chord), zorder=0, scale=1, color="k", linestyle=':', linewidth=2)
    small_len = 0.2  # mida del símbol de perpendicularitat
    Line(LE+v_perp * small_len, v_ble * small_len, zorder=0, scale=1, color="k", linewidth=1)
    Line(LE + v_ble * small_len, v_perp * (small_len+0.01), zorder=0, scale=1, color="k", linewidth=1)


    Point(CP, shape='Point1M', zorder=0, color=colors[1], bgcolor=colors[1], scale=0.7)
    ax.annotate3D('$\\mathbf{x}_{cp}$', xyz=CP + np.array([0.1, -0.4, 0]), fontsize=12, color='k')

    scaling_force = 4
    Vector(CP, F_align, zorder=0, scale=scaling_force, color=colors[2], shape='Arrow1Mend', linewidth=2)
    ax.annotate3D('$\\mathbf{F}_{a}$', xyz=CP+F_align*4 + np.array([0.3, -0.4, 0]), fontsize=12, color='k')
    Vector(CP, L, zorder=0, scale=scaling_force, color=colors[2], shape='Arrow1Mend', linewidth=2)
    Vector(CP, D, zorder=0, scale=scaling_force, color=colors[2], shape='Arrow1Mend', linewidth=2)
    Line(CP + D * scaling_force, L * scaling_force, zorder=0, color=colors[2], linestyle=':', linewidth=2)
    Line(CP + L * scaling_force, D * scaling_force, zorder=0, color=colors[2], linestyle=':', linewidth=2)

    Vector(LE - 2 * v_a, 2 * v_a, zorder=0, scale=1, color=colors[3], shape='Arrow1Mend', linewidth=2)
    ax.annotate3D(r'$\alpha$', xyz=LE-2*v_a+np.array([-0.3,0,0]), fontsize=12, color='k')
    

    # Angle annotations
    ArcMeasure(B, v1=LE - B, v2=CP - B, radius=3, linewidth=1, shape="Arrow1Mend", zorder=100, color="k")
    
    # ArcMeasure(LE, v1=np.array([chord, 0, 0]), v2=TE - LE, radius=3, linewidth=1, shape="Arrow1Mend", zorder=100, color="k")
    # ax.annotate3D(r'$\alpha_d$', xyz=LE + np.array([chord * 1.2, 0.2, 0]), fontsize=12, color='k')

    # ArcMeasure(LE, v1=-v_a, v2=LE - TE, radius=2, linewidth=1, shape="Arrow1Mend", zorder=100, color="k")
    ax.annotate3D(r'$\alpha$', xyz=LE-2*v_a+np.array([-0.3,0,0]), fontsize=12, color='k')
    plt.tight_layout()
    save_svg_tex(output_path, fontsize=14)
    plt.close()


# Generate two separate diagrams with different depower angles
# render_vplot3d_diagram('../figures/2d_symmetric_actuation_b', angle_depower=10, x_cp=0.4, ld = 4)
render_vplot3d_diagram('../figures/2d_symmetric_actuation_a', angle_depower=-10, x_cp=0.32, ld = 4)


# Assumes compute_positions, find_alpha_for_cp_alignment, aero_coeffs_polynomial are already defined


alpha_range = np.linspace(0, 20, 720)
chord = 2.6
ld = 11.5
x_cp = 0.32
AR = 0

depower_angles = [1, 10]
CL_curve, CD_curve = aero_coeffs_polynomial(np.radians(alpha_range))
from color_palette import set_plot_style

set_plot_style()
plt.figure(figsize=(3, 4))


for i, angle_depower in enumerate(depower_angles):
    LE, TE, B, CP = compute_positions(chord, ld, angle_depower, x_cp)
    alpha_wind, alpha_attack, F_align, min_angle, L, D = find_alpha_for_cp_alignment(LE, TE, B, CP, AR=AR)
    CL_pt, CD_pt = aero_coeffs_polynomial(np.radians(alpha_attack))
    label = f"$\\alpha_d$: {angle_depower}°"
    plt.scatter(alpha_attack, CL_pt / CD_pt, color=colors[i+1], label=label)
    print(f"[{label}] Alpha attack: {alpha_attack:.2f}°, CL: {CL_pt:.2f}, CD: {CD_pt:.2f}, CL/CD: {CL_pt / CD_pt:.2f}")


max_CLCD_idx = np.argmax(CL_curve / CD_curve)
plt.plot(alpha_range, CL_curve / CD_curve, color=colors[0], lw=1)
plt.vlines(
    alpha_range[max_CLCD_idx], 0, 6, color='gray', linestyle='--', lw=1)

# Coordinates for arrow base (at max CL/CD)
alpha_max = alpha_range[max_CLCD_idx]
y_arrow = 4  # vertical position for arrow and labels

# Left-pointing arrow: Stable
plt.annotate('Stable', xy=(alpha_max - 0.5, y_arrow + 0.2),
             ha='right', va='bottom', fontsize=10)
plt.arrow(alpha_max, y_arrow, -2, 0,
          head_width=0.15, head_length=0.4, fc='black', ec='black', length_includes_head=True)

# Right-pointing arrow: Unstable
plt.annotate('Unstable', xy=(alpha_max + 0.5, y_arrow + 0.2),
             ha='left', va='bottom', fontsize=10)
plt.arrow(alpha_max, y_arrow, 2, 0,
          head_width=0.15, head_length=0.4, fc='black', ec='black', length_includes_head=True)

plt.xlabel(r"$\alpha (\circ)$")
plt.ylabel(r'$C_L/C_D$')
plt.legend(frameon= True)
plt.tight_layout()
plt.savefig('../figures/2d_symmetric_ld.pdf', dpi=300, bbox_inches='tight')
plt.show()

