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

def find_alpha_for_cp_alignment(LE, TE, B, CP, AR, weight= 0):
    cp_vec = CP - B
    cp_unit = cp_vec / np.linalg.norm(cp_vec)
    best_alpha_wind = None
    min_angle = np.inf
    best_force = None
    best_alpha_attack = None
    for alpha_wind_deg in np.linspace(-20, 20, 720):
        alpha_wind_rad = np.radians(alpha_wind_deg)
        F_aero, L, D, alpha_attack_deg = compute_aero_forces(alpha_wind_rad, LE, TE, AR)
        vtau = 1/np.arctan(alpha_wind_deg)
        F_aero = F_aero * (vtau**2+1)**2
        L = L * (vtau**2+1)**2
        D = D * (vtau**2+1)**2
        F = F_aero + weight
        F_unit = F / np.linalg.norm(F)
        angle = np.degrees(np.arccos(np.clip(np.dot(F_unit, cp_unit), -1.0, 1.0)))
        if angle < min_angle:
            min_angle = angle
            best_alpha_wind = alpha_wind_deg
            best_force = F_aero
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


    weight = -np.array([0.5,0,0])*0.5
    # Start building the plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')
    ax.set_axis_off()

    init_view(width=600, height=500,
              xmin=0, xmax=chord*0.4, ymin=0.0, ymax=ld * 1.5, zmin=0, zmax=0,
              zoom=1, elev=90, azim=270)

    LE, TE, B, CP = compute_positions(chord, ld, angle_depower, x_cp)
    cp_vec = CP - B
    angle_to_y = np.arctan2(cp_vec[0], cp_vec[1])

    LE, TE, B, CP = rotate_all(angle_to_y, LE, TE, B, CP)

    alpha_wind, alpha_attack, F_align, min_angle, L, D = find_alpha_for_cp_alignment(LE, TE, B, CP, AR, weight)
    print(f'Aoa: {alpha_attack:.2f}°')
    print(f'Alpha wind: {alpha_wind:.2f}°')
    print("Min angle: ", min_angle)
    # Define perturbation in degrees
    delta_deg = 5.0
    scaling_force = 1.5

    print(f'L/D: {np.linalg.norm(L) / np.linalg.norm(D):.2f}')

    # Lower and higher wind angles
    alpha_wind_low = np.radians(alpha_wind - delta_deg)
    alpha_wind_high = np.radians(alpha_wind + delta_deg)

    # Compute forces at perturbed angles
    F_low, L_low, D_low, alpha_low_deg = compute_aero_forces(alpha_wind_low, LE, TE, AR)
    F_high, L_high, D_high, alpha_high_deg = compute_aero_forces(alpha_wind_high, LE, TE, AR)
    print(f'Low L/D: {np.linalg.norm(L_low) / np.linalg.norm(D_low):.2f}')
    print(f'High L/D: {np.linalg.norm(L_high) / np.linalg.norm(D_high):.2f}')

    print(f'Aoa low: {alpha_low_deg:.2f}°')
    print(f'Aoa high: {alpha_high_deg:.2f}°')


    alpha_d_rad = np.radians(angle_depower)
    Px = np.array([np.cos(alpha_d_rad), np.sin(alpha_d_rad), 0])
    Py = np.array([np.sin(alpha_d_rad), np.cos(alpha_d_rad), 0])
    v_a = np.array([np.cos(np.radians(alpha_wind)), np.sin(np.radians(alpha_wind)), 0])
    v_a_low = np.array([np.cos(alpha_wind_low), np.sin(alpha_wind_low), 0])
    v_a_high = np.array([np.cos(alpha_wind_high), np.sin(alpha_wind_high), 0])

    # LE, TE, B, CP = rotate_all(angle_to_y, LE, TE, B, CP)
    Px, Py = rotate_all(angle_to_y, Px, Py)
    # F_low, v_a_low, F_high, v_a_high,weight = rotate_all(angle_to_y, F_low, v_a_low, F_high, v_a_high,weight)
    CP_low = CP + (LE - TE) * 0.08
    CP_high = CP - (LE - TE) * 0.02

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
    ax.annotate3D('$\\mathbf{F}_{t}$', xyz=B+np.array([0.1, -0.8, 0]), fontsize=12, color='k')

    v_ble = LE - B
    v_ble = -v_ble / np.linalg.norm(v_ble)  # unit vector
    v_perp = np.array([-v_ble[1], v_ble[0], 0])
    Line(LE, v_perp * (chord), zorder=0, scale=1, color="k", linestyle=':', linewidth=2)
    small_len = 0.2  # mida del símbol de perpendicularitat
    Line(LE+v_perp * small_len, v_ble * small_len, zorder=0, scale=1, color="k", linewidth=1)
    Line(LE + v_ble * small_len, v_perp * (small_len+0.01), zorder=0, scale=1, color="k", linewidth=1)


    Point(CP, shape='Point1M', zorder=100, color=colors[0], bgcolor=colors[0], scale=0.4)
    # Point(CP_low, shape='Point1M', zorder=100, color=colors[1], bgcolor=colors[1], scale=0.4)
    # Point(CP_high, shape='Point1M', zorder=100, color=colors[2], bgcolor=colors[2], scale=0.4)
    ax.annotate3D('$\\mathbf{x}_{cp}$', xyz=CP + np.array([0.1, -0.4, 0]), fontsize=12, color='k')

    
    Vector(CP, F_align, zorder=0, scale=scaling_force, color=colors[0], shape='Arrow1Mend', linewidth=2)
    ax.annotate3D('$\\mathbf{F}_{a}$', xyz=CP+F_align*scaling_force + np.array([0.3, -0.4, 0]), fontsize=12, color='k')
    Vector(CP, L, zorder=0, scale=scaling_force, color=colors[0], shape='Arrow1Mend', linewidth=2)
    Vector(CP, D, zorder=0, scale=scaling_force, color=colors[0], shape='Arrow1Mend', linewidth=2)
    Vector(CP, weight, zorder=0, scale=scaling_force, color=colors[2], shape='Arrow1Mend', linewidth=1.5)
    ax.annotate3D('$\\mathbf{W}$', xyz=CP+weight*scaling_force + np.array([0, 0.3, 0]), fontsize=12, color=colors[2])
    Line(CP + D * scaling_force, L * scaling_force, zorder=0, color=colors[0], linestyle=':', linewidth=2)
    Line(CP + L * scaling_force, D * scaling_force, zorder=0, color=colors[0], linestyle=':', linewidth=2)

    # Add extra force vectors at slightly perturbed AoAs
    # Vector(CP+(LE-TE)*0.08, F_low, zorder=0, scale=scaling_force, color=colors[1], shape='Arrow1Mend', linewidth=1.5)
    # Vector(CP-(LE-TE)*0.02, F_high, zorder=0, scale=scaling_force, color=colors[2], shape='Arrow1Mend', linewidth=1.5)

    F_low_x = F_low[0] * scaling_force
    F_high_x = F_high[0] * scaling_force
    # Vector(B, F_low_x * np.array([1, 0, 0]), zorder=20, scale=1, color=colors[1], shape='Arrow1Mend', linewidth=1.5)
    # Vector(B, F_high_x * np.array([1, 0, 0]), zorder=20, scale=1, color=colors[2], shape='Arrow1Mend', linewidth=1.5)

    # Optional labels
    # ax.annotate3D(r'$\mathbf{F}_{a}^{-}$', xyz=CP + F_low*scaling_force + np.array([-0.1, 0.1, 0]), fontsize=12, color=colors[1])
    # ax.annotate3D(r'$\mathbf{F}_{a}^{+}$', xyz=CP + F_high*scaling_force + np.array([0.1, -0.1, 0]), fontsize=12, color=colors[2])

    Vector(LE - 2 * v_a, 2 * v_a, zorder=0, scale=1, color=colors[0], shape='Arrow1Mend', linewidth=2)
    # ax.annotate3D(r'$\alpha$', xyz=LE-2*v_a+np.array([-0.3,0,0]), fontsize=12, color='k')
    ax.annotate3D(r'$\mathbf{v}_a$', xyz=LE + np.array([-chord * 0.8, -0.8, 0]), fontsize=12, color='k')
    # Vector(LE - 2 * v_a_low, 2 * v_a_low, zorder=0, scale=1, color=colors[1], shape='Arrow1Mend', linewidth=1.5)
    # ax.annotate3D(r'$\alpha^{-}$', xyz=LE-2*v_a_low+np.array([-0.3,0,0]), fontsize=12, color=colors[1])
    # Vector(LE - 2 * v_a_high, 2 * v_a_high, zorder=0, scale=1, color=colors[2], shape='Arrow1Mend', linewidth=1.5)
    # ax.annotate3D(r'$\alpha^{+}$', xyz=LE-2*v_a_high+np.array([-0.3,0,0]), fontsize=12, color=colors[2])
    
    Line(B-np.array([chord/2, 0, 0]), B+np.array([chord, 0, 0]), zorder=10, scale=1, color="k", linestyle='--', linewidth=1)
    # Angle annotations
    ArcMeasure(B, v1=LE - B, v2=CP - B, radius=3, linewidth=1, shape="Arrow1Mend", zorder=100, color="k")
    
    ArcMeasure(LE, v1=v_perp * (chord), v2=TE - LE, radius=2.5, linewidth=1, shape="Arrow1Mend", zorder=100, color="k")
    ax.annotate3D(r'$\alpha_d$', xyz=LE + np.array([chord * 1, 0.2, 0]), fontsize=12, color='k')

    ArcMeasure(LE, v1=-v_a, v2=LE - TE, radius=2, linewidth=1, shape="Arrow1Mend", zorder=100, color="k")
    ax.annotate3D(r'$\alpha$', xyz=LE-2*v_a+np.array([-0.3,0,0]), fontsize=12, color='k')

    moment_v1 = np.array([0, -1, 0])*1
    moment_v2 = np.array([-0, -1, 0])*1
    moment_v3 = np.array([0, -1, 0])
    def draw_moment_arc(center, radius, theta1_deg, theta2_deg, n_points=300, color="k", linewidth=1, zorder=0):

        theta = np.radians(np.linspace(theta1_deg, theta2_deg, n_points))
        arc_points = [center + radius * np.array([np.cos(t), np.sin(t), 0]) for t in theta]
        for i in range(len(arc_points) - 30):
            Line(arc_points[i], arc_points[i+1] - arc_points[i], zorder=zorder, scale=1, color=color, linewidth=linewidth)
        
        Vector(arc_points[i], -arc_points[-1] + arc_points[i], zorder=zorder, scale=1, color=color, shape='Arrow1Mend', linewidth=linewidth)

    # draw_moment_arc(center=B+np.array([0.5,0.3,0]), radius=0.2, theta1_deg=0, theta2_deg=-330, color=colors[1], linewidth=1)
    # draw_moment_arc(center=B+np.array([-0.5,0.3,0]), radius=0.2, theta1_deg=0, theta2_deg=330, color=colors[2], linewidth=1)
    plt.tight_layout()
    # plt.show()
    save_svg_tex(output_path, fontsize=14)
    plt.close()


# Generate two separate diagrams with different depower angles
# render_vplot3d_diagram('../figures/2d_symmetric_actuation_b', angle_depower=10, x_cp=0.4, ld = 4)
render_vplot3d_diagram('../figures/2d_symmetric_stability', angle_depower=-12, x_cp=0.32, ld = 4)
