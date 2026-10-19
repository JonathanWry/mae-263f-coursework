import numpy as np


def stretching_gradient_hessian(q_local, rest_length, EA):
    """Return the gradient and Hessian of one stretching energy."""

    edge = q_local[2:4] - q_local[0:2]
    length = np.linalg.norm(edge)
    if length <= 0.0:
        raise ValueError("a beam edge has zero length")

    edge_gradient = EA / rest_length * (1.0 - rest_length / length) * edge
    edge_hessian = EA / rest_length * (
        (1.0 - rest_length / length) * np.eye(2)
        + rest_length * np.outer(edge, edge) / length**3
    )

    gradient = np.r_[-edge_gradient, edge_gradient]
    hessian = np.block(
        [[edge_hessian, -edge_hessian], [-edge_hessian, edge_hessian]]
    )
    return gradient, hessian


def _cross_matrix(vector):
    """Return the matrix A such that A @ b == cross(vector, b)."""

    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def bending_gradient_hessian(
    q_local, natural_curvature, voronoi_length, EI
):
    """Return the gradient and Hessian of one planar bending energy."""

    node0 = np.array([q_local[0], q_local[1], 0.0])
    node1 = np.array([q_local[2], q_local[3], 0.0])
    node2 = np.array([q_local[4], q_local[5], 0.0])
    material_normal_e = np.array([0.0, 0.0, 1.0])
    material_normal_f = material_normal_e.copy()

    edge_e = node1 - node0
    edge_f = node2 - node1
    length_e = np.linalg.norm(edge_e)
    length_f = np.linalg.norm(edge_f)
    if length_e <= 0.0 or length_f <= 0.0:
        raise ValueError("a beam edge has zero length")

    tangent_e = edge_e / length_e
    tangent_f = edge_f / length_f
    chi = 1.0 + np.dot(tangent_e, tangent_f)
    if chi <= 1.0e-12:
        raise ValueError("adjacent beam edges are nearly antiparallel")

    curvature_binormal = (
        2.0 * np.cross(tangent_e, tangent_f) / chi
    )
    curvature = curvature_binormal[2]
    tangent_tilde = (tangent_e + tangent_f) / chi
    normal_tilde = (material_normal_e + material_normal_f) / chi

    derivative_e = (
        -curvature * tangent_tilde
        + np.cross(tangent_f, normal_tilde)
    ) / length_e
    derivative_f = (
        -curvature * tangent_tilde
        - np.cross(tangent_e, normal_tilde)
    ) / length_f

    curvature_gradient = np.zeros(6)
    curvature_gradient[0:2] = -derivative_e[0:2]
    curvature_gradient[2:4] = derivative_e[0:2] - derivative_f[0:2]
    curvature_gradient[4:6] = derivative_f[0:2]

    identity = np.eye(3)
    tangent_outer = np.outer(tangent_tilde, tangent_tilde)

    cross_f_normal = np.cross(tangent_f, normal_tilde)
    cross_f_outer = np.outer(cross_f_normal, tangent_tilde)
    curvature_normal_e = np.outer(curvature_binormal, material_normal_e)
    second_ee = (
        (2.0 * curvature * tangent_outer - cross_f_outer - cross_f_outer.T)
        / length_e**2
        - curvature
        * (identity - np.outer(tangent_e, tangent_e))
        / (chi * length_e**2)
        + (curvature_normal_e + curvature_normal_e.T)
        / (4.0 * length_e**2)
    )

    cross_e_normal = np.cross(tangent_e, normal_tilde)
    cross_e_outer = np.outer(cross_e_normal, tangent_tilde)
    curvature_normal_f = np.outer(curvature_binormal, material_normal_f)
    second_ff = (
        (2.0 * curvature * tangent_outer + cross_e_outer + cross_e_outer.T)
        / length_f**2
        - curvature
        * (identity - np.outer(tangent_f, tangent_f))
        / (chi * length_f**2)
        + (curvature_normal_f + curvature_normal_f.T)
        / (4.0 * length_f**2)
    )

    second_ef = (
        -curvature
        * (identity + np.outer(tangent_e, tangent_f))
        / (chi * length_e * length_f)
        + (
            2.0 * curvature * tangent_outer
            - cross_f_outer
            + cross_e_outer.T
            - _cross_matrix(normal_tilde)
        )
        / (length_e * length_f)
    )
    second_fe = second_ef.T

    curvature_hessian = np.zeros((6, 6))
    ee = second_ee[0:2, 0:2]
    ef = second_ef[0:2, 0:2]
    fe = second_fe[0:2, 0:2]
    ff = second_ff[0:2, 0:2]
    curvature_hessian[0:2, 0:2] = ee
    curvature_hessian[0:2, 2:4] = -ee + ef
    curvature_hessian[0:2, 4:6] = -ef
    curvature_hessian[2:4, 0:2] = -ee + fe
    curvature_hessian[2:4, 2:4] = ee - ef - fe + ff
    curvature_hessian[2:4, 4:6] = ef - ff
    curvature_hessian[4:6, 0:2] = -fe
    curvature_hessian[4:6, 2:4] = fe - ff
    curvature_hessian[4:6, 4:6] = ff

    curvature_difference = curvature - natural_curvature
    scale = EI / voronoi_length
    gradient = scale * curvature_difference * curvature_gradient
    hessian = scale * (
        np.outer(curvature_gradient, curvature_gradient)
        + curvature_difference * curvature_hessian
    )
    return gradient, hessian
