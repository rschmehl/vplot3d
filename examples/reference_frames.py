import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from vplot3d.vplot3d import (
    init_view,
    Point,
    Line,
    Vector,
    save_svg_tex,
    Arc,
    ArcMeasure,
    Polygon,
)
import os
import sys
from pathlib import Path

# Folder with configuration and shared data
dat_path = Path.cwd().parent / "data"
# os.environ['CONF_PATH'] = str(dat_path)
sys.path.append(str(dat_path))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.kiteV3 import KiteV3


PALETTE = {
    "Black": "#000000",
    "Orange": "#E69F00",
    "Sky Blue": "#56B4E9",
    "Bluish Green": "#009E73",
    "Yellow": "#F0E442",
    "Blue": "#0072B2",
    "Vermillion": "#D55E00",
    "Reddish Purple": "#CC79A7",
}
colors = list(PALETTE.values())


def transformation_C_from_A(theta_a, chi_a, roll):
    # Define the Pitch matrix
    Pitch = np.array(
        [
            [np.cos(theta_a), 0, np.sin(theta_a)],
            [0, 1, 0],
            [-np.sin(theta_a), 0, np.cos(theta_a)],
        ]
    )

    # Define the Yaw matrix
    Yaw = np.array(
        [
            [np.cos(chi_a), -np.sin(chi_a), 0],
            [np.sin(chi_a), np.cos(chi_a), 0],
            [0, 0, 1],
        ]
    )

    # Define the Roll matrix
    Roll = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    # Compute the transformation matrix T using matrix multiplication
    T = Yaw @ Pitch @ Roll

    return T


def aerodynamic_angles_from_apparent_wind(v_apparent_wind):
    yaw = -np.atan(v_apparent_wind[1] / v_apparent_wind[0])
    pitch = np.atan2(
        v_apparent_wind[2], np.sqrt(v_apparent_wind[0] ** 2 + v_apparent_wind[1] ** 2)
    )
    return pitch, yaw


def transformation_AZR_from_W(azimuth, elevation):
    phi = azimuth
    beta = elevation
    transformation = np.array(
        [
            [-np.sin(phi), np.cos(phi), 0],
            [-np.sin(beta) * np.cos(phi), -np.sin(beta) * np.sin(phi), np.cos(beta)],
            [np.cos(beta) * np.cos(phi), np.cos(beta) * np.sin(phi), np.sin(beta)],
        ]
    )
    return transformation


def transformation_B_from_C(depower_angle):
    # Y rotation
    transformation = np.array(
        [
            [np.cos(depower_angle), 0, np.sin(depower_angle)],
            [0, 1, 0],
            [-np.sin(depower_angle), 0, np.cos(depower_angle)],
        ]
    )
    return transformation


def transformation_C_from_AZR(chi):
    transformation = np.array(
        [
            [np.sin(chi), np.cos(chi), 0],
            [-np.cos(chi), np.sin(chi), 0],
            [0, 0, 1],
        ]
    )
    return transformation


def z_rotation_matrix(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def rotation_matrix_earth2body(roll, pitch, yaw, sequence="321"):
    # Returns rotation matrix to transform from earth to body reference frame.
    # Earth: East, North, up
    # Body: front, left, up

    # Rotational matrix for roll.
    r_roll = np.array(
        [[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]]
    )

    # Rotational matrix for pitch (nose down).
    r_pitch = np.array(
        [
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    # Rotational matrix for yaw.
    r_yaw = np.array(
        [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    rs = [r_roll, r_pitch, r_yaw]
    sequence = [int(i) - 1 for i in sequence]
    r = rs[sequence[2]].dot(rs[sequence[1]].dot(rs[sequence[0]]))
    return r


def plot_circle(roll, pitch, yaw, axis, radius, theta_max=2 * np.pi, **kwargs):
    rm_b2e = rotation_matrix_earth2body(roll, pitch, yaw).T

    n_elem = abs(int(theta_max * 180 / np.pi))
    theta = np.linspace(0, theta_max, n_elem)
    s = radius * np.sin(theta)
    c = radius * np.cos(theta)
    if axis == 0:  # TODO: not checked z rotation
        xyz = np.vstack([[-s], [c]])
    else:
        xyz = np.vstack([[s], [c]])
    xyz = np.insert(xyz, axis, 0, axis=0)

    xyz = rm_b2e.dot(xyz)

    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)


def plot_ground_square(roll, pitch, yaw, **kwargs):
    xyz = np.zeros((3, 5))
    xyz[:2, :] = get_square()

    rm = rotation_matrix_earth2body(roll, pitch, yaw).T
    xyz = rm.dot(xyz)
    plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)


def get_square(side_x=0.5, side_y=1.0):
    xy = np.zeros((2, 5))
    xy[:, 0] = [side_x, side_y]
    xy[:, 1] = [-side_x, side_y]
    xy[:, 2] = [-side_x, -side_y]
    xy[:, 3] = [side_x, -side_y]
    xy[:, 4] = [side_x, side_y]

    return xy


def plot_tangential_plane(
    center_point,
    elevation,
    azimuth,
    course,
    axis=2,
    plot_kappa=False,
    plot=True,
    **kwargs,
):
    rm = transformation_AZR_from_W(azimuth, elevation)

    # Plot square
    rm = transformation_C_from_AZR(course) @ rm
    rm = rm.T
    xyz = get_square(side_x=1.0, side_y=1.0)
    xyz = np.insert(xyz, axis, 0, axis=0)
    xyz = rm.dot(xyz) + np.repeat([center_point], 5, axis=0).T

    # Plot x-axis
    xaxis = np.zeros((3, 2))
    xaxis[0, 1] = 2
    xaxis = rm.dot(xaxis) + np.repeat([center_point], 2, axis=0).T
    if plot:
        # plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)
        ax3d = plt.gca()
        pc = Poly3DCollection([xyz.T])
        pc.set_facecolor((1, 1, 1, 0))
        pc.set_edgecolor("k")
        pc.set_linewidth(1)
        ax3d.add_collection3d(pc)
        # plt.plot(xaxis[0, :], xaxis[1, :], xaxis[2, :], **kwargs)

    return xaxis


def plot_ref_frame(elevation, azimuth, course, radius=10):

    scaling = 60

    # Plot earth reference frame
    Px = np.array([1, 0, 0])
    Py = np.array([0, 1, 0])
    Pz = np.array([0, 0, 1])
    e1 = Vector(
        PO, Px * scaling, shape="Arrow1Mend", linewidth=3, zorder=50, color=colors[0]
    )
    e2 = Vector(
        PO, Py * scaling, shape="Arrow1Mend", linewidth=3, zorder=50, color=colors[0]
    )
    e3 = Vector(
        PO, Pz * scaling, shape="Arrow1Mend", linewidth=3, zorder=50, color=colors[0]
    )

    ax.annotate3D(r'$\vec{O}$', xyz=PO, xytext=(-0.5,-1)) 
    ax.annotate3D(r'$\vec{e}_x$', xyz=Py, xytext=(-3,-1.5))
    ax.annotate3D(r'$\vec{e}_y$', xyz=Px, xytext=(2,-1.5))
    ax.annotate3D(r'$\vec{e}_z$', xyz=Pz, xytext=(-1.3,4.5))

    dcm = np.array([Px, Py, Pz])
    # Plot azimuth reference frame
    dcm_AZR = transformation_AZR_from_W(azimuth, elevation) @ dcm
    e1 = Vector(
        PO,
        dcm_AZR[0, :] * scaling,
        shape="Arrow1Mend",
        linewidth=3,
        zorder=50,
        color=colors[1],
    )
    e2 = Vector(
        PO,
        dcm_AZR[1, :] * scaling,
        shape="Arrow1Mend",
        linewidth=3,
        zorder=50,
        color=colors[1],
    )
    e3 = Vector(
        PO,
        dcm_AZR[2, :] * scaling,
        shape="Arrow1Mend",
        linewidth=3,
        zorder=50,
        color=colors[1],
    )

    ax.annotate3D(r'$\vec{e}_r$', xyz=PO, xytext=(-2.5,1.7))
    ax.annotate3D(r'$\vec{e}_\phi$', xyz=PO, xytext=(4.8,-0.5))
    ax.annotate3D(r'$\vec{e}_\beta$', xyz=PO, xytext=(1.7,4.2))

    # Plot course reference frame
    dcm_C = transformation_C_from_AZR(course) @ dcm_AZR
    pos = np.array(
        [
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ]
    )
    dcm_C = dcm_C
    e1 = Vector(
        PO,
        dcm_C[0, :] * scaling,
        shape="Arrow1Mend",
        linewidth=3,
        zorder=10,
        color=colors[2],
    )
    e2 = Vector(
        PO,
        dcm_C[1, :] * scaling,
        shape="Arrow1Mend",
        linewidth=3,
        zorder=10,
        color=colors[2],
    )
    e3 = Vector(
        PO,
        dcm_C[2, :] * scaling,
        shape="Arrow1Mend",
        linewidth=3,
        zorder=10,
        color=colors[2],
    )
    ax.annotate3D(r'$\vec{e}_\chi$', xyz=PO, xytext=(4.8,0.5))
    ax.annotate3D(r'$\vec{e}_n$', xyz=PO, xytext=(0.4,4.8))
    am1 = ArcMeasure(
        PO,
        v1=dcm_AZR[1, :],
        v2=dcm_C[0, :],
        vn=dcm_C[2, :],
        radius=scaling / 2,
        linewidth=1,
        shape="Arrow1Mend",
        zorder=100,
        color="k",
    )
    ax.annotate3D(r'$\chi$', xyz=PO, xytext=(2,2.1))
    return dcm, dcm_AZR, dcm_C


fig = plt.figure()
ax = fig.add_subplot(projection="3d", proj_type="ortho")
ax.set_axis_off()
elevation0 = 20 * np.pi / 180.0
radius = 200
# Initialize vector diagram
# See also https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
init_view(
    width=1200,
    height=800,
    xmin=-radius / 2,
    xmax=radius,
    ymin=-radius,
    ymax=radius,
    zmin=0,
    zmax=radius,
    zoom=1.5,
    elev=5,
    azim=60,
)

# Origin
PO = np.array([0, 0, 0])
O = Point(PO, shape="Point1M", zorder=100, color="k")

# Lissajous
t = np.linspace(0, 2 * np.pi, 361)
phi_liss = 35 * np.pi / 180.0 * np.sin(t)
beta_liss = 10 * np.pi / 180.0 * np.sin(2 * t) + elevation0

i_plot_instance = 42
elevation = beta_liss[i_plot_instance]
azimuth = phi_liss[i_plot_instance]
course = np.radians(80)

dcm, dcm_AZR, dcm_C = plot_ref_frame(elevation, azimuth, course)

x = radius * np.cos(phi_liss) * np.cos(beta_liss)
y = radius * np.sin(phi_liss) * np.cos(beta_liss)
z = radius * np.sin(beta_liss)
xyz_liss = np.vstack([[x], [y], [z]])

# # Ground circle
theta = np.linspace(-np.pi / 2, np.pi / 2, 361)
s = radius * np.sin(theta)
c = radius * np.cos(theta)
xyz = np.vstack([[c], [s]])

xyz = np.insert(xyz, 2, 0, axis=0)
plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], color="k", linestyle="--", linewidth=0.7)

plt.plot([0, 0], [-radius, radius], [0, 0], color="k", linestyle="--", linewidth=0.7)
kwargs = {"color": "k", "linewidth": 0.7, "linestyle": "--"}
# Elevated circle
r = np.cos(elevation) * radius
s = r * np.sin(theta)
c = r * np.cos(theta)
xyz = np.vstack([[c], [s]])
z = np.sin(elevation) * radius
xyz = np.insert(xyz, 2, z, axis=0)
plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)

# Half circles
theta = np.linspace(0, np.pi, 181)
s = radius * np.sin(theta)
c = radius * np.cos(theta)
xyz = np.vstack([[c], [s]])
xyz = np.insert(xyz, 0, 0, axis=0)
plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)

theta = np.linspace(0, np.pi / 2, 181)
s = radius * np.sin(theta)
c = radius * np.cos(theta)
xyz = np.vstack([[c], [s]])
xyz = np.insert(xyz, 1, 0, axis=0)
plt.plot(xyz[0, :], xyz[1, :], xyz[2, :], **kwargs)

# xyz_kite = z_rotation_matrix(azimuth) @ xyz
# plt.plot(xyz_kite[0, :], xyz_kite[1, :], xyz_kite[2, :], **kwargs)

pos = [
    radius * np.cos(azimuth) * np.cos(elevation),
    radius * np.sin(azimuth) * np.cos(elevation),
    radius * np.sin(elevation),
]

# Plot lissajous
plt.plot(
    xyz_liss[0, :],
    xyz_liss[1, :],
    xyz_liss[2, :],
    linewidth=10,
    color="#3185cfff",
    alpha=0.25,
)

plot_tangential_plane(pos, elevation, azimuth, course, color="k")

pos = np.array(
    [
        radius * np.cos(elevation) * np.cos(azimuth),
        radius * np.cos(elevation) * np.sin(azimuth),
        radius * np.sin(elevation),
    ]
)



pos_t = pos - dcm_C[2, :]*20 #Tether attachemnt point
kv3 = KiteV3.rotated(pos_t, e1=dcm_C[0, :], e3=dcm_C[2, :], voff=pos_t, scale=5)



vtau = 30
vw = 10
vr = 2
depower_angle = np.radians(-5)

velocity_kite = np.array([vtau, 0, vr])
velocity_apparent_wind = vw * dcm[0, :] - velocity_kite

velocity_apparent_wind_C = dcm_C @ ([vw,0,0]) - velocity_kite
theta_a, chi_a = aerodynamic_angles_from_apparent_wind(velocity_apparent_wind_C)


Vector(PO, pos, shape="Arrow1Mend", zorder=100, color="k", linewidth=1)
wing_pos = pos + dcm_C[2, :] * 35
Point(pos, shape="Point1M", scale=1, zorder=50, color=colors[0])
dcm_A = transformation_C_from_A(theta_a, chi_a, 0).T @ dcm_C

Vector(wing_pos, dcm_A[0, :] * 60, shape="Arrow1Mend", zorder=11, color=colors[3])
Vector(wing_pos, dcm_A[1, :] * 60, shape="Arrow1Mend", zorder=11, color=colors[3])
Vector(wing_pos, dcm_A[2, :] * 60, shape="Arrow1Mend", zorder=11, color=colors[3])


dcm_WB = transformation_C_from_A(depower_angle, chi_a, 0).T @ dcm_C

Point(wing_pos, shape="Point1M", scale=1, zorder=50, color=colors[0])
Vector(wing_pos, dcm_WB[0, :] * 60, shape="Arrow1Mend", zorder=11, color=colors[4])
Vector(wing_pos, dcm_WB[1, :] * 60, shape="Arrow1Mend", zorder=11, color=colors[4])
Vector(wing_pos, dcm_WB[2, :] * 60, shape="Arrow1Mend", zorder=11, color=colors[4])


velocity_kite_W = dcm_C.T @ velocity_kite
# Kite velocity plot
Vector(pos, velocity_kite_W, shape="Arrow1Mend", zorder=11, scale = 2,color=colors[0])
# normal direction
Vector(pos, -dcm_C[1, :], shape="Arrow1Mend", zorder=11, scale = 30,color=colors[0], linestyle=':', linewidth=1)

save_svg_tex(
    "../figures/reference_frame", macro_file_path=dat_path / "macros.tex", scour=True
)
