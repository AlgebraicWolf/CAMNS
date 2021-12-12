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

    # if alpha is None:
    #     return False

    L, D = C.shape

    T = C[np.all(np.abs(C @ alpha + d) < tol, axis=1), :]

    if T.shape[0] == 0:
        return False

    return np.linalg.matrix_rank(T, tol=tol) == D


def CAMNS_LP(xs, N, lptol=1e-8, exttol=1e-8, verbose=True):
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
        verbose : bool
            Whether to print information about progress

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

    # Step 1. Preparing variables
    B = np.diag(np.ones(L))
    l = 0  # Number of extracted sources
    S = np.zeros((0, L))  # Source matrix

    epoch = 1

    while l < N:
        if verbose:
            print("Epoch {}:".format(epoch))
            print("=" * 58)
        epoch += 1
        # Step 2. Choosing random vector and generating direction r
        w = np.random.randn(L)
        r = B @ w

        # Step 3. Solving linear programming problems using CVXPY
        alpha1_star = cp.Variable(C.shape[1])
        alpha2_star = cp.Variable(C.shape[1])

        problem1 = cp.Problem(cp.Minimize(
            r.T @ (C @ alpha1_star)), [C @ alpha1_star + d.flatten() >= 0])
        problem2 = cp.Problem(cp.Maximize(
            r.T @ (C @ alpha2_star)), [C @ alpha2_star + d.flatten() >= 0])

        if verbose:
            print("\tLaunching LP solver 1")
        p_star = problem1.solve()

        if verbose:
            print("\tLaunching LP solver 2")
        q_star = problem2.solve()

        if verbose:
            print("\tLP solvers have finished, checking results")

        alpha1_star = np.expand_dims(alpha1_star.value, axis=1)
        alpha2_star = np.expand_dims(alpha2_star.value, axis=1)

        s1 = C @ alpha1_star + d
        s2 = C @ alpha2_star + d

        # Step 4. Checking results (with augmentations from MATLAB implementation)
        if l == 0:
            if is_extreme_point(C, alpha1_star, d, exttol):
                S = np.append(S, [s1.squeeze()], axis=0)
            if is_extreme_point(C, alpha2_star, d, exttol):
                S = np.append(S, [s2.squeeze()], axis=0)

        else:
            if np.abs(p_star) / (np.linalg.norm(r) * np.linalg.norm(s1)) >= lptol:
                if is_extreme_point(C, alpha1_star, d, exttol):
                    S = np.append(S, [s1.squeeze()], axis=0)

            if np.abs(q_star) / (np.linalg.norm(r) * np.linalg.norm(s2)) >= lptol:
                if is_extreme_point(C, alpha2_star, d, exttol):
                    S = np.append(S, [s2.squeeze()], axis=0)

        # Step 5. Updating l
        l = S.shape[0]

        if verbose:
            print("\tRetrieved {}/{} sources\n".format(l, N))

        # Step 6. Updating B
        Q1, R1 = np.linalg.qr(S.T)
        B = np.diag(np.ones(L)) - Q1 @ Q1.T

        # Step 7 is kinda implicit, as it is hidden in the loop condition

    # Yay, we're done!
    return S
