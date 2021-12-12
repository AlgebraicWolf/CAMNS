import numpy as np
import cvxpy as cp


def is_extreme_point(C, alpha, d, tol):
    """
    Check whether alpha is an extreme point of polyhedron

    Arguments:
    ----------
        C : np.ndarray matrix
        d : np.ndarray vector
            Tuple characterizing affine set
        alpha:
            Point to be tested

    Returns:
    --------
        bool
            Whether point is an extreme point
    """

    if alpha is None:
        return False

    L, D = C.shape

    T = C[np.all(np.abs(C @ alpha + d) <= tol, axis=1), :]

    if T.shape[0] == 0:
        return False

    return np.linalg.matrix_rank(T, tol=tol) == D


def CAMNS_LP(xs, N, lptol, exttol):
    """
    Solve CAMNS problem via reduction to Linear Programming

    Arguments:
    ----------
        xs : np.ndarray of shape (M, L)
            Observation matrix consisting of M observations
        N : int
            Number of observations
        lptol : float
            Tolerance for Linear Programming problem
        exttol : float
            Tolerance for extreme point check

    Returns:
    --------
        np.ndarray of shape (N, L)
            Estimated source matrix
    """
    M, L = xs.shape  # Extract dimensions
    xs = xs.T

    d = np.mean(xs, axis=1, keepdims=True)
    C, _, _ = np.linalg.svd(xs - d, full_matrices=False)

    C = C[:, :(N - 1)]  # Truncate the redundant one

    B = np.diag(np.ones(L))

    l = 0  # Number of extracted sources
    S = np.zeros((0, L))  # Source matrix

    while l < N:
        w = np.random.multivariate_normal(np.zeros(L), np.diag(np.ones(L)))
        r = B @ w

        # Solving LP using CVXPY
        alpha1_star = cp.Variable(C.shape[1])
        alpha2_star = cp.Variable(C.shape[1])

        problem1 = cp.Problem(cp.Minimize(
            r.T @ (C @ alpha1_star + d.reshape(C.shape[0]))), [C @ alpha1_star + d.reshape(C.shape[0]) >= 0])
        problem2 = cp.Problem(cp.Maximize(
            r.T @ (C @ alpha2_star + d.reshape(C.shape[0]))), [C @ alpha2_star + d.reshape(C.shape[0]) >= 0])

        p_star = problem1.solve()
        q_star = problem2.solve()

        alpha1_star = alpha1_star.value
        alpha2_star = alpha2_star.value

        if l == 0:
            if is_extreme_point(C, alpha1_star, d, exttol):
                S = np.append(S, C @ alpha1_star + d, axis=0)
            if is_extreme_point(C, alpha2_star, d, exttol):
                S = np.append(S, C @ alpha2_star + d)

        else:
            if np.abs(p_star) / (np.linalg.norm(r) * np.linalg.norm(C @ alpha1_star + d)) >= lptol:
                if is_extreme_point(C, alpha1_star, d, exttol):
                    S = np.append(S, C @ alpha1_star + d)

            if np.abs(q_star) / (np.linalg.norm(r) * np.linalg.norm(C @ alpha2_star + d)) >= lptol:
                if is_extreme_point(C, alpha2_star, d, exttol):
                    S = np.append(S, C @ alpha2_star + d)

        l = S.shape[0]

        Q1, R1 = np.linalg.qr(S.T)
        B = np.diag(np.ones(L)) - Q1 @ Q1.T

    return S
