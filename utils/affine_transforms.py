"""
Affine transformation utilities for 2D coordinate transformations.

This module provides functions for:
- Finding affine transformations between point sets
- Decomposing affine transformations into components
- Analyzing affine transformation invariants (fixed points, invariant lines)
- Applying transformations to points
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy.linalg import polar, lstsq
from shapely.geometry import Polygon
from shapely.errors import GEOSException


# ============================================================================
# Core Affine Transformation Functions
# ============================================================================

def find_linear_transformation(prev_xy, new_xy):
    """
    Computes the best affine transformation using least squares and returns the transformation
    as a 3x3 matrix (homogeneous coordinates).

    Args:
        prev_xy (np.ndarray): Array of shape (N, 2) with original coordinates.
        new_xy (np.ndarray): Array of shape (N, 2) with transformed coordinates.

    Returns:
        A (np.ndarray): 3x3 affine transformation matrix.
        transform (function): Applies the forward transformation to points.
        inverse_transform (function): Applies the inverse transformation to points.
    """
    if not isinstance(prev_xy, np.ndarray) or not isinstance(new_xy, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays")
    if prev_xy.shape[1] != 2 or new_xy.shape[1] != 2:
        raise ValueError("Input arrays must have shape (N, 2)")
    if prev_xy.shape[0] != new_xy.shape[0]:
        raise ValueError("Input arrays must have the same number of rows (N)")

    # Augment with ones for homogeneous coordinates
    X_aug = np.hstack([prev_xy, np.ones((prev_xy.shape[0], 1))])  # Nx3
    Y = new_xy  # Nx2

    # Solve least squares for affine parameters: X_aug @ A[:2, :] = Y  => A[:2, :] = theta.T
    theta, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)  # 3x2

    # Construct full 3x3 affine matrix
    A = np.eye(3)
    A[:2, :] = theta.T  # top 2 rows filled with theta.T

    # Forward transformation
    def transform(points):
        if points.shape[1] != 2:
            raise ValueError("Expected shape (N, 2)")
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        return (A @ points_h.T).T[:, :2]

    # Inverse transformation
    def inverse_transform(points):
        if points.shape[1] != 2:
            raise ValueError("Expected shape (N, 2)")
        A_inv = np.linalg.inv(A)
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        return (A_inv @ points_h.T).T[:, :2]

    return A, transform, inverse_transform


def find_affine_transformation(source_points, target_points):
    """
    Find the affine transformation matrix that maps source_points to target_points.

    Args:
        source_points (np.ndarray): Nx2 array of source points.
        target_points (np.ndarray): Nx2 array of target points.

    Returns:
        np.ndarray: 3x3 affine transformation matrix.
    """
    # Ensure points are NumPy arrays
    source_points = np.asarray(source_points)
    target_points = np.asarray(target_points)

    # Add a column of ones to the source points for affine transformation
    ones = np.ones((source_points.shape[0], 1))
    source_points_augmented = np.hstack([source_points, ones])

    # Solve for the affine transformation matrix
    # A * [a, b, c] = target_x and A * [d, e, f] = target_y
    A = source_points_augmented
    B = target_points

    # Solve for the transformation matrix using least squares
    params, _, _, _ = lstsq(A, B)

    # Reshape the parameters to form the affine matrix
    affine_matrix = np.vstack([params.T, [0, 0, 1]])
    return affine_matrix


def apply_transformation(points, transformation_matrix):
    """
    Apply a transformation matrix to a set of 2D points.

    Args:
        points (np.ndarray): Nx2 array of 2D points.
        transformation_matrix (np.ndarray): 3x3 transformation matrix.

    Returns:
        np.ndarray: Transformed Nx2 array of 2D points.
    """
    # Ensure points are a NumPy array
    points = np.asarray(points)

    # Add a column of ones to the points to convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_augmented = np.hstack([points, ones])  # Shape: Nx3

    # Apply the transformation matrix
    transformed_points_homogeneous = points_augmented @ transformation_matrix.T  # Shape: Nx3

    # Convert back to 2D by dividing by the homogeneous coordinate (should be 1)
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:3]

    return transformed_points


def extract_affine_components(A, center=None):
    """
    Decomposes a 3x3 affine transformation matrix into component steps in the order:
      T_final @ T_back @ shear @ rotation/reflection @ T_to_origin
    All linear steps (rotation/reflection, shear) act around the given center.

    Args:
        A (np.ndarray): 3x3 affine transformation matrix.
        center (np.ndarray): Optional (2,) array for rotation/shear/reflection center. Defaults to origin.

    Returns:
        dict: {
            'T_to_origin', 'rotation_matrix', 'reflection_matrix', 'shear_matrix',
            'T_back', 'final_translation',
            'angle_rad', 'angle_deg',
            'eigenvalues', 'eigenvectors',
            'reflection', 'reflection_axis', 'center_used'
        }
    """
    if A.shape != (3, 3):
        raise ValueError("A must be a 3x3 affine matrix")

    # default center
    if center is None:
        center = np.zeros(2)
    center = np.asarray(center, float)

    # Translate to origin and back
    to_origin_translation = np.eye(3)
    to_origin_translation[:2, 2] = -center

    back_translation = np.eye(3)
    back_translation[:2, 2] = center

    # Centered transformation matrix
    A_centered = to_origin_translation @ A @ back_translation
    A_linear = A_centered[:2, :2]
    t_final = A_centered[:2, 2]

    # Final translation
    final_translation = np.eye(3)
    final_translation[:2, 2] = t_final

    # Left‐polar decomposition: A_linear = S @ U
    # where U is orthonormal (rotation/reflection), S is symmetric (shear/scale)
    U, S = polar(A_linear, side='left')
    detP = np.linalg.det(U)
    reflection = detP < 0

    # build canonical reflection if needed
    if reflection:
        vals, vecs = np.linalg.eig(U)
        idx = np.argmin(np.abs(vals - 1))
        v = np.real(vecs[:, idx])
        v /= np.linalg.norm(v)
        used_reflection_axis = v.copy()
        # reflection across eigenline v
        Mv = 2.0 * np.outer(v, v) - np.eye(2)
        # remove reflection from U, leaving pure rotation
        R = U @ Mv
        # angle is that of the reflection line
        angle_rad = np.arctan2(v[1], v[0])
        # normalize reflection angle to [0, 180)
        angle_deg = (np.degrees(angle_rad) + 180) % 180
    else:
        used_reflection_axis = None
        Mv = np.eye(2)
        R = U
        # angle is rotation angle
        angle_rad = np.arctan2(R[1, 0], R[0, 0])
        angle_deg = (np.degrees(angle_rad) + 360) % 360 #np.degrees(angle_rad) % 360


    # Reflection matrix
    reflect_3x3 = np.eye(3)
    reflect_3x3[:2, :2] = Mv
    # Rotation matrix
    rotate_3x3 = np.eye(3)
    rotate_3x3[:2, :2] = R
    # Shear matrix
    shear_3x3 = np.eye(3)
    shear_3x3[:2, :2] = S

    # Eigen-decomposition of shear
    eigvals_s, eigvecs_s = np.linalg.eigh(S)

    return {
        "to_origin_translation_matrix": to_origin_translation,
        "reflection_matrix": reflect_3x3,
        "rotation_matrix": rotate_3x3,
        "shear_matrix": shear_3x3,
        "back_translation_matrix": back_translation,
        "final_translation_matrix": final_translation,
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "eigenvalues": eigvals_s,
        "max_eigenvalue": np.max(eigvals_s),
        "min_eigenvalue": np.min(eigvals_s),
        "eigenvectors": eigvecs_s,
        "reflection": reflection,
        "reflection_axis": used_reflection_axis,
        "center_used": center,
    }


# ============================================================================
# Affine Invariants Analysis Functions
# ============================================================================

def affine_from_homog(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (A, b) from 3x3 homogeneous matrix M.
    
    Args:
        M: 3x3 homogeneous transformation matrix
        
    Returns:
        A: 2x2 linear part
        b: 2x1 translation part
    """
    A = M[:2, :2]
    b = M[:2, 2]
    return A, b


def is_almost_identity(A: np.ndarray, b: np.ndarray, 
                       tolA: float = 1e-6, tolb: float = 1e-6) -> bool:
    """
    Check if transform (A, b) is approximately identity.
    
    Args:
        A: 2x2 linear part
        b: 2x1 translation part
        tolA: Tolerance for ||A - I||
        tolb: Tolerance for ||b||
        
    Returns:
        True if approximately identity
    """
    # Normalize shapes
    A = np.asarray(A, dtype=float).reshape(2, 2)
    b = np.asarray(b, dtype=float).reshape(2,)
    
    I = np.eye(2)
    A_diff = np.linalg.norm(A - I, ord='fro')
    b_norm = np.linalg.norm(b)
    return A_diff < tolA and b_norm < tolb


def find_fixed_set(A: np.ndarray, b: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None,
                   rank_rtol: float = 1e-12, residual_tol: float = 1e-6, far_factor: float = 3.0) -> Dict:
    """
    Find the fixed set of the affine transformation f(x) = Ax + b.
    
    Solves (I - A)x = b to find:
    - unique fixed point (rank=2)
    - fixed line (rank=1 and consistent)
    - all points fixed / pure translation (rank=0)
    - no fixed points (rank=1 inconsistent)
    
    Properly classifies cases based on rank(I-A):
    - rank=2: Unique fixed point
    - rank=1: Line of fixed points (if b in column space) or no fixed points
    - rank=0: All points fixed (if b=0) or pure translation (if b≠0)
    
    Args:
        A: 2x2 linear part
        b: 2x1 translation part
        bbox: Optional (xmin, xmax, ymin, ymax) for meaningfulness check
        rank_rtol: Relative tolerance for rank computation (default: 1e-12)
        residual_tol: Tolerance for residual to consider solution valid (default: 1e-6)
        far_factor: Factor for bbox diagonal to determine meaningfulness (default: 3.0)
        
    Returns:
        dict with keys:
            'point': (2,) fixed point or point on fixed line (or None)
            'direction': (2,) direction vector for line of fixed points (or None)
            'residual': residual norm ||(I-A)@x - b||
            'rank': rank of (I-A)
            'condition': condition number of (I-A)
            'is_valid': bool indicating if fixed set is mathematically valid
            'case': str - 'unique', 'line', 'all', 'none', 'translation'
            'is_meaningful_to_plot': bool - True if point/line is within reasonable distance of bbox
            'passes_threshold': bool - True if residual < residual_tol (for unique/line cases)
    """
    # Normalize shapes to avoid broadcasting issues
    A = np.asarray(A, dtype=float).reshape(2, 2)
    b = np.asarray(b, dtype=float).reshape(2,)
    
    I = np.eye(2)
    M = I - A
    
    # Compute rank using SVD with relative tolerance (scale-aware)
    s = np.linalg.svd(M, compute_uv=False)
    rank = int(np.sum(s > rank_rtol * s[0])) if len(s) > 0 and s[0] > 0 else 0
    
    try:
        cond = np.linalg.cond(M)
    except:
        cond = np.inf
    
    # Classify based on rank
    if rank == 0:
        # A = I (identity matrix)
        b_norm = np.linalg.norm(b)
        if b_norm < residual_tol:
            # All points are fixed
            return {
                'point': None,  # No specific point - all points are fixed
                'direction': None,
                'residual': 0.0,
                'rank': rank,
                'condition': cond,
                'is_valid': True,
                'case': 'all',
                'is_meaningful_to_plot': True,  # All points fixed, so always meaningful
                'passes_threshold': True
            }
        else:
            # Pure translation, no fixed points
            return {
                'point': None,
                'direction': None,
                'residual': np.inf,
                'rank': rank,
                'condition': cond,
                'is_valid': False,
                'case': 'translation',
                'is_meaningful_to_plot': False,
                'passes_threshold': False
            }
    
    elif rank == 1:
        # Check if b is in the column space of (I-A) by solving and checking residual
        # Find one candidate point using least squares
        p = np.linalg.lstsq(M, b, rcond=None)[0]
        residual = np.linalg.norm(M @ p - b)  # Actual equation residual
        
        if residual < residual_tol:
            # b is in column space: line of fixed points
            # Null space direction (right singular vector corresponding to zero singular value)
            u, s, vh = np.linalg.svd(M, full_matrices=False)
            direction = vh[-1]  # Last row of vh is null space
            direction = direction / np.linalg.norm(direction)  # Normalize
            
            # Choose a nice point on the line for plotting: closest to bbox center
            if bbox is not None:
                xmin, xmax, ymin, ymax = bbox
                bbox_diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
                center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
                # Project center onto line: p_plot = p + t*direction where t = dot(center - p, direction)
                t = np.dot(center - p, direction)
                p_plot = p + t * direction
                # Distance from center to line (simplified: p_plot is projection, so distance is just ||center - p_plot||)
                dist_center_to_line = np.linalg.norm(center - p_plot)
                is_meaningful = dist_center_to_line < far_factor * bbox_diagonal
            else:
                p_plot = p
                is_meaningful = True
        
            return {
                'point': np.real(p_plot),
                'direction': np.real(direction),
                'residual': residual,
                'rank': rank,
                'condition': cond,
                'is_valid': True,
                'case': 'line',
                'is_meaningful_to_plot': is_meaningful,
                'passes_threshold': residual < residual_tol
            }
        else:
            # b is not in column space: no fixed points
            return {
                'point': None,
                'direction': None,
                'residual': residual,
                'rank': rank,
                'condition': cond,
                'is_valid': False,
                'case': 'none',
                'is_meaningful_to_plot': False,
                'passes_threshold': False
        }
    
    else:  # rank == 2
        # Full rank: unique solution
        try:
            x_fixed = np.linalg.solve(M, b)
            residual = np.linalg.norm(M @ x_fixed - b)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            M_pinv = np.linalg.pinv(M)
            x_fixed = M_pinv @ b
            residual = np.linalg.norm(M @ x_fixed - b)
        
        # Check if solution is meaningful (low residual)
        is_valid = residual < residual_tol and cond < 1e10
        
        # Check meaningfulness for plotting (use distance from center to fixed point vs bbox diagonal)
        is_meaningful = True
        if bbox is not None and is_valid:
            xmin, xmax, ymin, ymax = bbox
            bbox_diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
            center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
            dist = np.linalg.norm(x_fixed - center)
            is_meaningful = dist < far_factor * bbox_diagonal
        
        return {
            'point': np.real(x_fixed),
            'direction': None,
            'residual': residual,
            'rank': rank,
            'condition': cond,
            'is_valid': is_valid,
            'case': 'unique',
            'is_meaningful_to_plot': is_meaningful,
            'passes_threshold': residual < residual_tol
        }


def find_invariant_lines(A: np.ndarray, b: np.ndarray, 
                         extra_directions: Optional[List[np.ndarray]] = None,
                         reflection_axis: Optional[np.ndarray] = None,
                         analytic_residual_threshold: float = 1e-6,
                         dir_parallel_tol: float = 1e-6,
                         eig_imag_tol: float = 1e-7) -> List[Dict]:
    """
    Find invariant lines (setwise) for affine transform x -> A @ x + b.
    
    A line L: {p + t*v : t in R} is invariant if A @ L + b = L (setwise).
    This means: A @ v is parallel to v, and (A - I) @ p + b is parallel to v.
    
    For each candidate direction v, solves the constraint:
        (A - I) @ p - μ @ v = -b
    where p is a point on the line and μ is a scalar.
    
    Args:
        A: 2x2 linear part
        b: 2x1 translation part
        extra_directions: Optional list of (2,) direction vectors to try as candidates
                         (e.g., reflection axis, principal stretch axes)
        analytic_residual_threshold: Maximum residual for accepting a candidate line
        dir_parallel_tol: Tolerance for direction invariance test (||Av - λv||)
        eig_imag_tol: Maximum allowed imaginary part for eigenvectors
        
    Returns:
        List of dicts, each with keys:
            'point': (2,) point on line
            'direction': (2,) unit direction vector
            'eigenvalue': float or None (if from extra_directions)
            'is_valid': bool
            'analytic_residual': float - residual from constraint solving
            'is_priority': bool - True for reflection axis and principal stretch axes
            'dir_residual': float - direction invariance residual (||Av - λv||)
            'dir_residual_scaled': float - direction invariance residual (scaled by ||A||)
            'lambda_dir': float - eigenvalue-like quantity from Rayleigh quotient
    """
    # Normalize shapes to avoid broadcasting issues
    A = np.asarray(A, dtype=float).reshape(2, 2)
    b = np.asarray(b, dtype=float).reshape(2,)
    
    candidates = []
    M = A - np.eye(2)  # Note: A - I, not I - A
    
    # Collect candidate directions
    candidate_directions = []
    
    # 1. Real eigenvectors of A
    try:
        eigvals, eigvecs = np.linalg.eig(A)
        eigvecs = eigvecs.T  # Each row is an eigenvector
        
        for val, vec in zip(eigvals, eigvecs):
            # Check if imaginary part is small enough
            if np.max(np.abs(np.imag(vec))) > eig_imag_tol:
                continue  # Reject complex eigenvectors
            
            vec = np.real(vec)
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            direction = vec / vec_norm
            candidate_directions.append((direction, val, False))  # (direction, eigenvalue, is_priority)
    except Exception:
        pass  # If eigendecomposition fails, continue with extra_directions only
    
    # 2. Extra directions (reflection axis, principal stretch axes)
    if extra_directions is not None:
        for direction in extra_directions:
            direction = np.asarray(direction)
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-10:
                continue
            direction = direction / direction_norm
            candidate_directions.append((direction, None, True))  # (direction, eigenvalue=None, is_priority=True)
    
    # Deduplicate candidate directions (v and -v, and near-equals)
    unique_directions = []
    seen_directions = []
    for direction, eigenvalue, is_priority in candidate_directions:
        # Check if this direction (or its negative) is already seen
        is_duplicate = False
        for seen_dir, _, _ in seen_directions:
            dot_product = abs(np.dot(direction, seen_dir))
            if dot_product > 0.99:  # Very close alignment (including opposite)
                is_duplicate = True
                break
        if not is_duplicate:
            # Normalize direction again after dedup (safe if extra_directions aren't perfectly normalized)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            unique_directions.append((direction, eigenvalue, is_priority))
            seen_directions.append((direction, eigenvalue, is_priority))
    
    # For each candidate direction, check direction invariance first, then solve constraint
    A_norm = np.linalg.norm(A, ord=2)  # Spectral norm for scaling
    
    # Normalize reflection axis for tagging (if provided)
    reflection_axis_for_tagging = None
    if reflection_axis is not None:
        reflection_axis_arr = np.asarray(reflection_axis)
        norm = np.linalg.norm(reflection_axis_arr)
        if norm > 1e-10:
            reflection_axis_for_tagging = reflection_axis_arr / norm
    
    for direction, eigenvalue, is_priority in unique_directions:
        # Check direction invariance: A @ v should be parallel to v
        Av = A @ direction
        lam = float(direction @ Av)  # Rayleigh quotient since ||direction||=1
        dir_res = np.linalg.norm(Av - lam * direction)  # == ||component orthogonal to v||
        
        # Make tolerance relative to ||A||
        dir_parallel_tol_scaled = dir_parallel_tol * max(1.0, A_norm)
        dir_res_scaled = dir_res / max(1.0, A_norm)  # For comparability/debugging
        if dir_res > dir_parallel_tol_scaled:
            continue  # Direction is not invariant, skip
        
        # Tag alignment with reflection axis vs normal (if reflection axis provided)
        # Initialize defaults
        align_reflection_axis = 0.0
        align_reflection_normal = 0.0
        is_reflection_axis_dir = False
        is_reflection_normal_dir = False
        
        if reflection_axis_for_tagging is not None:
            r = reflection_axis_for_tagging
            r_normal = np.array([-r[1], r[0]])  # Perpendicular (90° rotation)
            align_reflection_axis = abs(np.dot(direction, r))
            align_reflection_normal = abs(np.dot(direction, r_normal))
            is_reflection_axis_dir = align_reflection_axis > 0.99
            is_reflection_normal_dir = align_reflection_normal > 0.99
        
        # Solve: (A - I) @ p - μ @ v = -b
        # Write as: [A-I, -v] @ [p; μ] = -b
        M_constraint = np.hstack([M, -direction.reshape(-1, 1)])
        
        try:
            result = np.linalg.lstsq(M_constraint, -b, rcond=None)
            solution = result[0]
            p = solution[:2]
            mu = solution[2]
            
            # Compute analytic residual: ||(A-I) @ p - μ @ v + b||
            # This is equivalent to ||M_constraint @ solution + b|| since M_constraint @ solution = (A-I) @ p - μ @ v
            residual = np.linalg.norm(M_constraint @ solution + b)
            
            # Only accept if residual is below threshold
            if residual < analytic_residual_threshold:
                candidates.append({
                    'point': np.real(p),
                    'direction': direction,
                    'eigenvalue': eigenvalue,
                    'is_valid': True,
                    'analytic_residual': residual,
                    'is_priority': is_priority,
                    'dir_residual': dir_res,  # Direction invariance residual (unscaled)
                    'dir_residual_scaled': dir_res_scaled,  # Direction invariance residual (scaled by ||A||)
                    'lambda_dir': lam,  # Eigenvalue-like quantity from Rayleigh quotient
                    'align_reflection_axis': align_reflection_axis,
                    'align_reflection_normal': align_reflection_normal,
                    'is_reflection_axis_dir': is_reflection_axis_dir,
                    'is_reflection_normal_dir': is_reflection_normal_dir
                })
        except Exception:
            # If solving fails, skip this candidate
            continue
    
    return candidates


def _compute_line_bbox_interval(line_point: np.ndarray, line_dir: np.ndarray,
                                bbox: Tuple[float, float, float, float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute t-interval where line p(t) = line_point + t * line_dir intersects bbox.
    
    Returns:
        (t_min, t_max) or (None, None) if no intersection
    """
    xmin, xmax, ymin, ymax = bbox
    
    # For each dimension, compute t where line crosses bbox boundaries
    t_x_min = (xmin - line_point[0]) / line_dir[0] if abs(line_dir[0]) > 1e-12 else None
    t_x_max = (xmax - line_point[0]) / line_dir[0] if abs(line_dir[0]) > 1e-12 else None
    t_y_min = (ymin - line_point[1]) / line_dir[1] if abs(line_dir[1]) > 1e-12 else None
    t_y_max = (ymax - line_point[1]) / line_dir[1] if abs(line_dir[1]) > 1e-12 else None
    
    # Collect valid t values
    t_candidates = []
    if t_x_min is not None:
        t_candidates.append(t_x_min)
    if t_x_max is not None:
        t_candidates.append(t_x_max)
    if t_y_min is not None:
        t_candidates.append(t_y_min)
    if t_y_max is not None:
        t_candidates.append(t_y_max)
    
    if not t_candidates:
        # Line is parallel to both axes (degenerate case)
        # Check if point is inside bbox
        if xmin <= line_point[0] <= xmax and ymin <= line_point[1] <= ymax:
            return (-np.inf, np.inf)  # Entire line is inside
        else:
            return (None, None)
    
    # Check which t values correspond to points inside bbox
    t_valid = []
    for t in t_candidates:
        p = line_point + t * line_dir
        if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax:
            t_valid.append(t)
    
    if not t_valid:
        # Check if line_point itself is inside bbox
        if xmin <= line_point[0] <= xmax and ymin <= line_point[1] <= ymax:
            # Line passes through bbox, use min/max of candidates
            return (min(t_candidates), max(t_candidates))
        else:
            return (None, None)
    
    return (min(t_valid), max(t_valid))


def score_invariant_line(A: np.ndarray, b: np.ndarray, line_point: np.ndarray, 
                        line_dir: np.ndarray, bbox: Tuple[float, float, float, float]) -> Dict:
    """
    Score how well a line is invariant by sampling points and checking distances.
    
    Works entirely in canonical (normalized) coordinates.
    
    Args:
        A: 2x2 linear part
        b: 2x1 translation part
        line_point: (2,) point on line (in canonical frame)
        line_dir: (2,) unit direction vector
        bbox: (xmin, xmax, ymin, ymax) bounding box for sampling (in canonical frame)
        
    Returns:
        dict with keys:
            'mean_dist': mean distance from transformed points to line
            'median_dist': median distance
            'max_dist': maximum distance
            'score': overall score (lower is better)
    """
    # Normalize shapes to avoid broadcasting issues
    A = np.asarray(A, dtype=float).reshape(2, 2)
    b = np.asarray(b, dtype=float).reshape(2,)
    line_point = np.asarray(line_point, dtype=float).reshape(2,)
    line_dir = np.asarray(line_dir, dtype=float).reshape(2,)
    
    # Normalize line_dir and early return if too small
    line_dir_norm = np.linalg.norm(line_dir)
    if line_dir_norm < 1e-12:
        return {
            'mean_dist': np.inf,
            'median_dist': np.inf,
            'max_dist': np.inf,
            'score': np.inf
        }
    line_dir = line_dir / line_dir_norm
    
    n_samples = 50  # Number of points to sample along line
    xmin, xmax, ymin, ymax = bbox
    
    # Compute t-range by intersecting line with bbox analytically
    # Parameterize line: p(t) = line_point + t * line_dir
    t_min, t_max = _compute_line_bbox_interval(line_point, line_dir, bbox)
    
    if t_min is None or t_max is None or t_max <= t_min:
        # Line doesn't intersect bbox
        return {
            'mean_dist': np.inf,
            'median_dist': np.inf,
            'max_dist': np.inf,
            'score': np.inf
        }
    
    # Sample points along the line within bbox
    t_sample = np.linspace(t_min, t_max, n_samples)
    points_on_line = np.array([line_point + t * line_dir for t in t_sample])
    
    # Transform points (in canonical frame)
    points_transformed = (A @ points_on_line.T).T + b
    
    # Compute distances from transformed points to original line
    # Distance from point q to line through p with direction d: ||(q - p) - ((q-p)@d) * d||
    distances = []
    for q in points_transformed:
        v = q - line_point
        proj_length = np.dot(v, line_dir)
        perp = v - proj_length * line_dir
        distances.append(np.linalg.norm(perp))
    
    distances = np.array(distances)
    
    result = {
        'mean_dist': np.mean(distances),
        'median_dist': np.median(distances),
        'max_dist': np.max(distances),
        'score': np.mean(distances)  # Use mean as overall score
    }
    return result


# ============================================================================
# Helper Functions for analyze_affine_invariants
# ============================================================================

def _prepare_extra_directions(reflection_axis: Optional[np.ndarray],
                              principal_stretch_axes: Optional[np.ndarray]) -> List[np.ndarray]:
    """Prepare normalized extra directions for invariant line search."""
    extra_directions = []
    
    # Add reflection axis if provided
    if reflection_axis is not None:
        reflection_axis_arr = np.asarray(reflection_axis)
        norm = np.linalg.norm(reflection_axis_arr)
        if norm > 1e-10:
            extra_directions.append(reflection_axis_arr / norm)
    
    # Add principal stretch axes if provided
    if principal_stretch_axes is not None:
        principal_stretch_axes = np.asarray(principal_stretch_axes)
        if principal_stretch_axes.ndim == 1:
            norm = np.linalg.norm(principal_stretch_axes)
            if norm > 1e-10:
                extra_directions.append(principal_stretch_axes / norm)
        elif principal_stretch_axes.ndim == 2:
            # Determine if columns or rows are axes
            if principal_stretch_axes.shape[0] == 2 and principal_stretch_axes.shape[1] == 2:
                # Columns are axes
                for i in range(2):
                    axis = principal_stretch_axes[:, i]
                    norm = np.linalg.norm(axis)
                    if norm > 1e-10:
                        extra_directions.append(axis / norm)
            elif principal_stretch_axes.shape[1] == 2:
                # Rows are axes
                for i in range(principal_stretch_axes.shape[0]):
                    axis = principal_stretch_axes[i, :]
                    norm = np.linalg.norm(axis)
                    if norm > 1e-10:
                        extra_directions.append(axis / norm)
    
    return extra_directions


def _compute_reflection_axis_score(line_candidates: List[Dict],
                                    reflection_axis: Optional[np.ndarray],
                                    use_score: bool = True) -> Optional[float]:
    """Compute reflection axis score from line candidates (returns best aligned, not first)."""
    if not line_candidates or reflection_axis is None:
        return None
    
    reflection_axis_norm = reflection_axis / np.linalg.norm(reflection_axis)
    
    # Find all aligned candidates
    aligned = []
    for line in line_candidates:
        direction = line['direction']
        alignment = abs(np.dot(direction, reflection_axis_norm))
        if alignment > 0.99:  # Very close alignment
            # Prefer reflection axis direction (lambda_dir ≈ +1) over normal (lambda_dir ≈ -1)
            is_axis = line.get('is_reflection_axis_dir', False)
            lambda_dir = line.get('lambda_dir', 0.0)
            aligned.append((line, alignment, is_axis, abs(lambda_dir - 1.0)))
    
    if not aligned:
        return None
    
    # Sort: prefer axis direction, then by lambda_dir closeness to +1, then by score/residual
    aligned.sort(key=lambda x: (not x[2], x[3], x[0].get('score' if use_score else 'analytic_residual', np.inf)))
    
    key = 'score' if use_score else 'analytic_residual'
    return aligned[0][0].get(key, np.inf)


def _find_axis_aligned_lines(line_candidates: List[Dict],
                             reflection_axis: Optional[np.ndarray],
                             use_existing_tags: bool = True) -> List[Dict]:
    """
    Find and mark axis-aligned invariant lines.
    
    Args:
        line_candidates: List of line candidates (may be filtered or unfiltered)
        reflection_axis: Reflection axis direction vector
        use_existing_tags: If True, use existing alignment tags; otherwise compute them
    
    Returns:
        List of axis-aligned lines
    """
    if not line_candidates or reflection_axis is None:
        return []
    
    reflection_axis_norm = reflection_axis / np.linalg.norm(reflection_axis)
    axis_lines = []
    
    for line in line_candidates:
        if use_existing_tags and 'is_reflection_axis_dir' in line:
            # Use existing tags from find_invariant_lines
            if line.get('is_reflection_axis_dir', False):
                axis_lines.append(line)
        else:
            # Compute alignment on the fly
            direction = line['direction']
            alignment = abs(np.dot(direction, reflection_axis_norm))
            line['alignment_reflection_axis'] = alignment
            line['is_axis_aligned'] = alignment > 0.99
            
            if line['is_axis_aligned']:
                axis_lines.append(line)
    
    return axis_lines


def _process_invariant_lines_with_bbox(result: Dict, line_candidates: List[Dict],
                                       A: np.ndarray, b: np.ndarray, bbox: Tuple[float, float, float, float],
                                       reflection_axis: Optional[np.ndarray],
                                       analytic_residual_threshold: float,
                                       invariant_line_threshold: float) -> None:
    """Process invariant lines when bbox is provided (with scoring)."""
    # Score each candidate line by sampling
    for line in line_candidates:
        score_dict = score_invariant_line(A, b, line['point'], line['direction'], bbox)
        line['score'] = score_dict['score']
        line['score_details'] = score_dict
    
    # Sort by score (lowest is best)
    line_candidates.sort(key=lambda x: x['score'])
    
    # Store best score
    result['best_invariant_line_score'] = line_candidates[0]['score'] if line_candidates else None
    
    # Compute reflection axis score from ALL candidates (before filtering)
    result['reflection_axis_score'] = _compute_reflection_axis_score(
        line_candidates, reflection_axis, use_score=True
    )
    
    # Filter lines: prefer lowest score among those with good analytic residual
    valid_analytic = [line for line in line_candidates 
                     if line.get('analytic_residual', np.inf) < analytic_residual_threshold]
    
    if valid_analytic:
        # Among analytically valid lines, filter by sampling score
        valid_both = [line for line in valid_analytic 
                     if line.get('score', np.inf) < invariant_line_threshold]
        if valid_both:
            result['invariant_lines'] = valid_both
            result['best_invariant_line'] = valid_both[0]
        else:
            # No lines intersect bbox well, but keep best analytic line
            result['invariant_lines'] = [valid_analytic[0]]
            result['best_invariant_line'] = valid_analytic[0]
            result['best_invariant_line']['intersects_bbox'] = False
    else:
        result['invariant_lines'] = []
        result['best_invariant_line'] = None
    
    # Find axis-aligned invariant line from filtered results (for display)
    axis_lines_filtered = _find_axis_aligned_lines(result['invariant_lines'], reflection_axis, use_existing_tags=True)
    if axis_lines_filtered:
        result['best_axis_aligned_invariant_line'] = min(
            axis_lines_filtered, key=lambda x: x.get('score', np.inf)
        )
    
    # Also find best axis-aligned from valid_analytic (before sampling threshold)
    # This captures axis-aligned lines that might not pass the sampling threshold
    if valid_analytic:
        axis_lines_analytic = _find_axis_aligned_lines(valid_analytic, reflection_axis, use_existing_tags=True)
        if axis_lines_analytic:
            result['best_axis_aligned_candidate'] = min(
                axis_lines_analytic, key=lambda x: x.get('score', np.inf)
            )


def _process_invariant_lines_without_bbox(result: Dict, line_candidates: List[Dict],
                                          reflection_axis: Optional[np.ndarray]) -> None:
    """Process invariant lines when bbox is not provided (analytic only)."""
    # Sort by analytic residual (lowest is best)
    line_candidates.sort(key=lambda x: x.get('analytic_residual', np.inf))
    result['invariant_lines'] = line_candidates
    result['best_invariant_line'] = line_candidates[0] if line_candidates else None
    result['best_invariant_line_score'] = None
    
    # Find axis-aligned invariant line (from all candidates, since no filtering)
    axis_lines = _find_axis_aligned_lines(line_candidates, reflection_axis, use_existing_tags=True)
    if axis_lines:
        result['best_axis_aligned_invariant_line'] = min(
            axis_lines, key=lambda x: x.get('analytic_residual', np.inf)
        )
    
    # Compute reflection axis score using analytic_residual
    result['reflection_axis_score'] = _compute_reflection_axis_score(
        line_candidates, reflection_axis, use_score=False
    )


def _extract_decomposition_summary(decomposition: Optional[Dict]) -> Optional[Dict]:
    """
    Extract decomposition summary from decomposition dict.
    
    Args:
        decomposition: Dict from extract_affine_components
        
    Returns:
        Dict with decomposition summary or None
    """
    if decomposition is None:
        return None
    
    shear_matrix = decomposition.get('shear_matrix')
    if shear_matrix is not None:
        S_2x2 = shear_matrix[:2, :2]
        area_scale = np.linalg.det(S_2x2)
    else:
        area_scale = None
    
    final_translation_matrix = decomposition.get('final_translation_matrix')
    if final_translation_matrix is not None:
        t_vec = final_translation_matrix[:2, 2]
        translation_magnitude = np.linalg.norm(t_vec)
    else:
        translation_magnitude = None
    
    max_eig = decomposition.get('max_eigenvalue')
    min_eig = decomposition.get('min_eigenvalue')
    anisotropy_ratio = max_eig / min_eig if (max_eig is not None and min_eig is not None and min_eig > 0) else None
    
    return {
        'reflection': decomposition.get('reflection', False),
        'angle_deg': decomposition.get('angle_deg'),
        's1': max_eig,
        's2': min_eig,
        'anisotropy_ratio': anisotropy_ratio,
        'area_scale': area_scale,
        'principal_stretch_axes': decomposition.get('eigenvectors'),
        'reflection_axis': decomposition.get('reflection_axis'),
        'translation_magnitude': translation_magnitude
    }


def _generate_invariant_message(invariants_result: Dict) -> str:
    """
    Generate a one-liner interpretation message based on the hierarchy classification.
    
    Args:
        invariants_result: Result dict from analyze_affine_invariants
        
    Returns:
        str: One-liner message describing the transformation
    """
    if invariants_result.get('is_identity', False):
        return "No remapping beyond noise."
    
    fixed = invariants_result.get('fixed')
    if fixed is None:
        return "No simple invariant structure detected."
    
    fixed_case = fixed.get('case', 'none')
    
    if fixed_case == 'all':
        return "No remapping."
    
    if fixed_case == 'unique':
        return "Remapping pivots around a stationary point."
    
    if fixed_case == 'line':
        fixed_line_type = fixed.get('fixed_line_type', 'stationary_line')
        if fixed_line_type == 'reflection_axis':
            return "Mirror remapping: axis is fixed pointwise."
        else:
            return "Stationary line exists, but not a mirror flip."
    
    # No fixed set: check invariant lines
    # COMMENTED OUT: Invariant line scenarios disabled
    # if fixed_case in ['none', 'translation']:
    #     best_axis_aligned = invariants_result.get('best_axis_aligned_invariant_line')
    #     best_invariant = invariants_result.get('best_invariant_line')
    #     
    #     if best_axis_aligned is not None:
    #         return "Glide remapping: axis preserved, points slide."
    #     elif best_invariant is not None:
    #         return "A 1D backbone is preserved setwise."
    #     else:
    #         return "No simple invariant structure detected."
    
    if fixed_case in ['none', 'translation']:
        return "No simple invariant structure detected."
    
    return "No simple invariant structure detected."


def analyze_affine_invariants(M: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None,
                             fixed_point_threshold: float = 1e-5,
                             invariant_line_threshold: float = 1e-4,
                             identity_tolA: float = 0.1,
                             identity_tolb: float = 0.1,
                             reflection_axis: Optional[np.ndarray] = None,
                             principal_stretch_axes: Optional[np.ndarray] = None,
                             analytic_residual_threshold: float = 1e-6,
                             dir_parallel_tol: float = 1e-6,
                             eig_imag_tol: float = 1e-7,
                             decomposition: Optional[Dict] = None) -> Dict:
    """
    Complete analysis of affine invariants using fixed set workflow.
    
    Prioritizes fixed set (fixed points/lines) over invariant lines. Only computes
    invariant lines if there are no fixed points/lines (case in ["none", "translation"]).
    
    Note: This function works entirely in canonical (normalized) coordinates.
    All threshold conversions should be done outside this function.
    
    Args:
        M: 3x3 homogeneous transformation matrix
        bbox: Optional (xmin, xmax, ymin, ymax) for line scoring and fixed point meaningfulness
        fixed_point_threshold: Maximum residual for valid fixed point (default: 1e-5 in canonical frame)
        invariant_line_threshold: Maximum score for valid invariant line (default: 1e-4 in canonical frame)
        identity_tolA: Tolerance for ||A - I|| to consider transform as identity (default: 1e-1)
        identity_tolb: Tolerance for ||b|| to consider transform as identity (default: 1e-1)
        reflection_axis: Optional (2,) direction vector for reflection axis (if provided, will be prioritized)
        principal_stretch_axes: Optional (2, 2) array of principal stretch axis directions (eigenvectors of S from polar decomposition)
        analytic_residual_threshold: Threshold for analytic residual in invariant line detection (default: 1e-6)
        dir_parallel_tol: Tolerance for direction invariance check (default: 1e-6)
        eig_imag_tol: Tolerance for rejecting complex eigenvectors (default: 1e-7)
        decomposition: Optional dict from extract_affine_components for sub-classification and decomposition summary
        
    Returns:
        dict with keys:
            'is_identity': bool
            'fixed': dict from find_fixed_set with additional sub-classification fields
            'invariant_lines': list of dicts with scoring (only computed if no fixed set)
            'best_invariant_line': best line (lowest score) or None
            'best_axis_aligned_invariant_line': best axis-aligned invariant line or None
            'best_invariant_line_score': best score (or None)
            'reflection_axis_score': score of reflection axis line (or None)
            'invariant_line_threshold': threshold used
            'decomposition': dict with decomposition summary (if decomposition provided)
            'interpretation_message': str with one-liner interpretation
    """
    A, b = affine_from_homog(M)
    
    # Normalize shapes to avoid broadcasting issues
    A = np.asarray(A, dtype=float).reshape(2, 2)
    b = np.asarray(b, dtype=float).reshape(2,)
    
    # Identity short-circuit
    is_identity = is_almost_identity(A, b, tolA=identity_tolA, tolb=identity_tolb)
    
    result = {
        'is_identity': is_identity,
        'fixed': None,
        'invariant_lines': [],
        'best_invariant_line': None,
        'best_axis_aligned_invariant_line': None,
        'best_invariant_line_score': None,
        'reflection_axis_score': None,
        'invariant_line_threshold': invariant_line_threshold
    }
    
    if is_identity:
        # Return identity case with fixed set indicating all points fixed
        # Compute condition for identity (I - A = 0, so condition is 1.0)
        result['fixed'] = {
            'case': 'all',
            'point': None,
            'direction': None,
            'residual': 0.0,
            'rank': 0,
            'condition': 1.0,  # Identity matrix has condition number 1.0
            'is_valid': True,
            'is_meaningful_to_plot': True,
            'passes_threshold': True
        }
        # Still add decomposition and message even for identity
        result['decomposition'] = _extract_decomposition_summary(decomposition)
        result['interpretation_message'] = _generate_invariant_message(result)
        return result
    
    # Step 2: Compute FIXED SET FIRST
    fixed = find_fixed_set(A, b, bbox=bbox, residual_tol=fixed_point_threshold)
    fixed['passes_threshold'] = fixed.get('is_valid', False) and fixed.get('residual', np.inf) < fixed_point_threshold
    
    # Step 2.1: Sub-classify fixed line cases
    if fixed.get('case') == 'line' and decomposition is not None:
        if decomposition.get('reflection', False):
            reflection_axis_decomp = decomposition.get('reflection_axis')
            if reflection_axis_decomp is not None:
                fixed_direction = fixed.get('direction')
                if fixed_direction is not None:
                    # Normalize directions for comparison
                    fixed_dir_norm = fixed_direction / (np.linalg.norm(fixed_direction) + 1e-12)
                    refl_axis_norm = reflection_axis_decomp / (np.linalg.norm(reflection_axis_decomp) + 1e-12)
                    alignment = abs(np.dot(fixed_dir_norm, refl_axis_norm))
                    fixed['fixed_line_reflection_alignment'] = alignment
                    if alignment > 0.99:
                        fixed['fixed_line_type'] = 'reflection_axis'
                    else:
                        fixed['fixed_line_type'] = 'stationary_line'
                else:
                    fixed['fixed_line_type'] = 'stationary_line'
                    fixed['fixed_line_reflection_alignment'] = 0.0
            else:
                fixed['fixed_line_type'] = 'stationary_line'
                fixed['fixed_line_reflection_alignment'] = 0.0
        else:
            fixed['fixed_line_type'] = 'stationary_line'
            fixed['fixed_line_reflection_alignment'] = 0.0
    
    result['fixed'] = fixed
    
    # Step 3: Decide if we need invariant-line search
    # COMMENTED OUT: Invariant line computation disabled - keeping only fixed point, fixed line, and identity
    # Only compute invariant lines if there are no fixed points/lines
    # need_invariant_lines = (fixed['case'] in ['none', 'translation'])
    
    # if not need_invariant_lines:
    #     # Fixed set exists (unique/line/all): invariants are optional
    #     # Primary plotted invariant should be the fixed point/line
    #     # Still add decomposition and message
    #     result['decomposition'] = _extract_decomposition_summary(decomposition)
    #     result['interpretation_message'] = _generate_invariant_message(result)
    #     return result
    
    # Always add decomposition and message (no invariant line computation)
    result['decomposition'] = _extract_decomposition_summary(decomposition)
    result['interpretation_message'] = _generate_invariant_message(result)
    return result
    
    # COMMENTED OUT: Invariant line search and processing
    # # Prepare extra directions for invariant line search
    # extra_directions = _prepare_extra_directions(reflection_axis, principal_stretch_axes)
    # 
    # # Step 4: Find invariant lines (setwise), analytic + optional bbox scoring
    # line_candidates = find_invariant_lines(A, b, 
    #                                      extra_directions=extra_directions if extra_directions else None,
    #                                      reflection_axis=reflection_axis,
    #                                      analytic_residual_threshold=analytic_residual_threshold,
    #                                      dir_parallel_tol=dir_parallel_tol,
    #                                      eig_imag_tol=eig_imag_tol)
    # 
    # # Process invariant lines (with or without bbox)
    # if bbox is not None and line_candidates:
    #     _process_invariant_lines_with_bbox(
    #         result, line_candidates, A, b, bbox, reflection_axis,
    #         analytic_residual_threshold, invariant_line_threshold
    #     )
    # elif line_candidates:
    #     _process_invariant_lines_without_bbox(
    #         result, line_candidates, reflection_axis
    #     )
    # 
    # # Step 5: Extract decomposition summary (if provided)
    # result['decomposition'] = _extract_decomposition_summary(decomposition)
    # 
    # # Step 6: Generate interpretation message
    # result['interpretation_message'] = _generate_invariant_message(result)
    # 
    # return result



# ============================================================================
# Geometry Fixing Utilities
# ============================================================================

def fix_invalid_polygon(polygon, logger=None):
    """
    Fix an invalid Shapely polygon using make_valid() or buffer(0).
    
    This function handles cases where transformations create self-intersections,
    degenerate edges, or other invalid geometries. It tries make_valid() first
    (available in Shapely 1.8+), then falls back to buffer(0).
    
    Args:
        polygon: Shapely Polygon or MultiPolygon to fix
        logger: Optional logger instance for warning messages
        
    Returns:
        Fixed polygon, or None if the polygon cannot be fixed
        (indicating a near-singular transformation or other severe issue)
    """
    if polygon.is_valid:
        return polygon
    
    # Try make_valid() first (Shapely 1.8+), fallback to buffer(0)
    try:
        if hasattr(polygon, 'make_valid'):
            fixed_polygon = polygon.make_valid()
        else:
            fixed_polygon = polygon.buffer(0)
        
        # Check if the fixed polygon is valid
        if fixed_polygon.is_valid:
            return fixed_polygon
    except Exception as e:
        if logger:
            logger.warning(f"Failed to fix invalid polygon using make_valid/buffer(0): {e}. Trying buffer(0) fallback.")
    
    # Fallback to buffer(0) if make_valid() failed
    try:
        fixed_polygon = polygon.buffer(0)
        if fixed_polygon.is_valid:
            return fixed_polygon
    except Exception as e:
        if logger:
            logger.warning(f"Failed to fix invalid polygon using buffer(0): {e}. Polygon remains invalid.")
        return None
    
    # If buffer(0) succeeded but result is still invalid, return None
    if logger:
        logger.warning("Polygon remains invalid after fixing attempts. Likely near-singular transformation.")
    return None

