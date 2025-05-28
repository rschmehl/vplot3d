# Visualize: applying roll (x-rotation) in aerodynamic frame (A), and finding the necessary compensation in body frame (B)

# Construct the rotation in A (aerodynamic frame): roll = phi
R_aero_roll = rot_x(phi)

# Apply to A and B (rotate both using the aerodynamic frame)
R_A_roll = R_aero_roll @ R_A
R_B_rolled = R_aero_roll @ R_B

# Now compute what rotation in B's local frame would produce this transformation
# We want R_local_in_B such that: R_local_in_B @ R_B = R_B_rolled
# => R_local_in_B = R_B_rolled @ R_B.T
R_local_in_B = R_B.T @ R_B_rolled

# Decompose the compensation rotation into x-z Euler angles
rot_local = R.from_matrix(R_local_in_B)
euler_angles_xyz = rot_local.as_euler('xyz', degrees=True)
roll_body = euler_angles_xyz[0]  # x rotation
yaw_body = euler_angles_xyz[2]   # z rotation

# Plot the result
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
draw_frame(ax1, R_A_roll, np.array([0, 0, 0]), 'A_roll', ['r', 'g', 'b'])
draw_frame(ax1, R_B_rolled, np.array([0, 0, 0]), 'B_rolled', ['m', 'c', 'y'])
ax1.set_title("Aero frame roll (R_x(phi)) applied to A and B")
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_zlim([-1, 1])
ax1.set_box_aspect([1,1,1])

ax2 = fig.add_subplot(122, projection='3d')
draw_frame(ax2, R_B @ R_local_in_B, np.array([0, 0, 0]), 'B_equiv', ['m', 'c', 'y'])
draw_frame(ax2, R_A_roll, np.array([0, 0, 0]), 'A_roll', ['r', 'g', 'b'])
ax2.set_title("Equivalent B-frame rotation to match A roll")
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim([-1, 1])
ax2.set_box_aspect([1,1,1])

plt.tight_layout()
plt.show()

(roll_body, yaw_body)
