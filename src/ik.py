import mujoco
import numpy as np


def damped_pseudo_inverse(J, damping=1e-4):
    """
    Damped Least Squares pseudo-inverse.

    Args:
        J: Jacobian matrix, shape (m, n).
        damping: Damping coefficient.

    Returns:
        Damped pseudo-inverse, shape (n, m).
    """
    return J.T @ np.linalg.inv(J @ J.T + damping * np.eye(J.shape[0]))


def get_joint_ids(model, joint_names):
    """
    Convert joint names to MuJoCo joint ids.
    """
    joint_ids = []

    for name in joint_names:
        try:
            joint_id = model.joint(name).id
        except KeyError:
            raise RuntimeError(f"Joint not found: {name}")

        joint_ids.append(joint_id)

    return joint_ids


def rotation_error_so3(R_current, R_target):
    """
    Compute 3D orientation error vector using cross products of rotation axes.

    Args:
        R_current: Current rotation matrix, shape (3, 3).
        R_target: Target rotation matrix, shape (3, 3).

    Returns:
        rot_error: Orientation error vector, shape (3,).
    """
    rot_error = 0.5 * (
        np.cross(R_current[:, 0], R_target[:, 0])
        + np.cross(R_current[:, 1], R_target[:, 1])
        + np.cross(R_current[:, 2], R_target[:, 2])
    )

    return rot_error


def solve_position_ik(
    model,
    data,
    ee_site_id,
    target_pos,
    joint_ids,
    max_iters=300,
    pos_tol=1e-4,
    step_size=0.5,
    damping=1e-4,
    verbose=True,
):
    """
    Position-only inverse kinematics for a MuJoCo site.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        ee_site_id: End-effector site id.
        target_pos: Target position in world frame, shape (3,).
        joint_ids: Joint ids to optimize.
        max_iters: Maximum IK iterations.
        pos_tol: Position convergence tolerance.
        step_size: IK update step size.
        damping: Damping coefficient for pseudo-inverse.
        verbose: Whether to print convergence message.

    Returns:
        q_target: Target joint positions.
        final_pos: Final end-effector position.
        final_pos_error_norm: Final position error norm.
    """
    nv = model.nv

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    qpos_ids = [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
    dof_ids = [model.jnt_dofadr[joint_id] for joint_id in joint_ids]

    final_pos_error_norm = None

    for i in range(max_iters):
        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[ee_site_id].copy()
        pos_error = target_pos - current_pos
        pos_error_norm = np.linalg.norm(pos_error)
        final_pos_error_norm = pos_error_norm

        if pos_error_norm < pos_tol:
            if verbose:
                print(f"[IK] Converged at iter {i}, pos_error={pos_error_norm:.6f}")
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

        J_pos = jacp[:, dof_ids]
        dq = damped_pseudo_inverse(J_pos, damping=damping) @ pos_error
        dq = step_size * dq

        for qpos_id, delta in zip(qpos_ids, dq):
            data.qpos[qpos_id] += delta

    mujoco.mj_forward(model, data)

    q_target = data.qpos[qpos_ids].copy()
    final_pos = data.site_xpos[ee_site_id].copy()

    return q_target, final_pos, final_pos_error_norm


def solve_pose_ik(
    model,
    data,
    ee_site_id,
    target_pos,
    target_rot,
    joint_ids,
    max_iters=500,
    pos_tol=1e-4,
    rot_tol=1e-3,
    step_size=0.3,
    damping=1e-4,
    rot_weight=0.5,
    verbose=True,
):
    """
    Position + orientation inverse kinematics for a MuJoCo site.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        ee_site_id: End-effector site id.
        target_pos: Target position in world frame, shape (3,).
        target_rot: Target rotation matrix in world frame, shape (3, 3).
        joint_ids: Joint ids to optimize.
        max_iters: Maximum IK iterations.
        pos_tol: Position convergence tolerance.
        rot_tol: Rotation convergence tolerance.
        step_size: IK update step size.
        damping: Damping coefficient for pseudo-inverse.
        rot_weight: Relative weight for orientation error.
        verbose: Whether to print convergence message.

    Returns:
        q_target: Target joint positions.
        final_pos: Final end-effector position.
        final_rot: Final end-effector rotation matrix.
        final_pos_error_norm: Final position error norm.
        final_rot_error_norm: Final rotation error norm.
    """
    nv = model.nv

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    qpos_ids = [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
    dof_ids = [model.jnt_dofadr[joint_id] for joint_id in joint_ids]

    final_pos_error_norm = None
    final_rot_error_norm = None

    for i in range(max_iters):
        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[ee_site_id].copy()
        current_rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()

        pos_error = target_pos - current_pos
        rot_error = rotation_error_so3(current_rot, target_rot)

        pos_error_norm = np.linalg.norm(pos_error)
        rot_error_norm = np.linalg.norm(rot_error)

        final_pos_error_norm = pos_error_norm
        final_rot_error_norm = rot_error_norm

        if pos_error_norm < pos_tol and rot_error_norm < rot_tol:
            if verbose:
                print(
                    f"[IK] Converged at iter {i}, "
                    f"pos_error={pos_error_norm:.6f}, "
                    f"rot_error={rot_error_norm:.6f}"
                )
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

        J_pos = jacp[:, dof_ids]
        J_rot = jacr[:, dof_ids]

        J = np.vstack([
            J_pos,
            rot_weight * J_rot,
        ])

        error = np.concatenate([
            pos_error,
            rot_weight * rot_error,
        ])

        dq = damped_pseudo_inverse(J, damping=damping) @ error
        dq = step_size * dq

        for qpos_id, delta in zip(qpos_ids, dq):
            data.qpos[qpos_id] += delta

    mujoco.mj_forward(model, data)

    q_target = data.qpos[qpos_ids].copy()
    final_pos = data.site_xpos[ee_site_id].copy()
    final_rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    return q_target, final_pos, final_rot, final_pos_error_norm, final_rot_error_norm