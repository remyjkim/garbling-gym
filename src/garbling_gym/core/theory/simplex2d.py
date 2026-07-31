# ABOUTME: 2-D (3-state) simplex concave & quasiconcave envelopes
# ABOUTME: cav via upper convex hull of lifted points; qcav via superlevel-set convex hulls.

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

__all__ = ["SimplexGrid", "EnvelopeFunction", "cav_simplex", "qcav_simplex"]


class SimplexGrid:
    """A triangular grid over the 3-state probability simplex.

    All points ``mu`` satisfy ``mu >= 0`` and ``mu.sum() == 1``.  The grid is
    parametrized by an integer ``n``: it contains every belief of the form
    ``(i/n, j/n, (n-i-j)/n)`` for non-negative integers ``i, j`` with
    ``i + j <= n`` — i.e. ``(n+1)(n+2)/2`` points in total.

    The grid is the discretization of the belief simplex over which the sender
    indirect value ``v(mu)`` is sampled; the envelope routines lift these
    samples to compute cav / qcav.
    """

    def __init__(self, n: int = 30) -> None:
        if n < 2:
            raise ValueError("SimplexGrid requires n >= 2")
        self.n = n
        pts = []
        for i in range(n + 1):
            for j in range(n + 1 - i):
                mx, my = i / n, j / n
                pts.append((mx, my, 1.0 - mx - my))
        self.mu = np.array(pts, dtype=float)  # (G, 3)
        # 2-D projection onto (mu_x, mu_y) used for hull/Delaunay geometry.
        self._xy = self.mu[:, :2]

    @property
    def size(self) -> int:
        return self.mu.shape[0]


@dataclass
class EnvelopeFunction:
    """A concave/quasiconcave envelope evaluated over the simplex.

    Constructed once from a grid + value samples; call ``.evaluate(mu0)`` to
    query the envelope at any belief (on or off the grid).
    """

    evaluate: Callable[[np.ndarray], float]


def _point_in_triangle(p: np.ndarray, tri_xy: np.ndarray, tol: float = 1e-9) -> bool:
    """Barycentric point-in-triangle test with tolerance for boundary points."""
    a, b, c = tri_xy[0], tri_xy[1], tri_xy[2]
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0 @ v0
    d01 = v0 @ v1
    d02 = v0 @ v2
    d11 = v1 @ v1
    d12 = v1 @ v2
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return False
    u = (d11 * d02 - d01 * d12) / denom
    w = (d00 * d12 - d01 * d02) / denom
    return u >= -tol and w >= -tol and (u + w) <= 1.0 + tol


def cav_simplex(grid: SimplexGrid, values: np.ndarray) -> EnvelopeFunction:
    """Concave envelope of ``v`` over the 3-state simplex.

    The full-commitment (Kamenica & Gentzkow 2011) sender value is the concave
    envelope of the indirect value function.  Over the 2-D simplex it is the
    upper surface of the convex hull of the lifted points
    ``{(mu_x, mu_y, v(mu))}``.

    Args:
        grid: a :class:`SimplexGrid` on which ``values`` are sampled.
        values: sender value at each grid point, shape ``(grid.size,)``.

    Returns:
        An :class:`EnvelopeFunction` whose ``evaluate(mu0)`` returns the
        concave envelope at belief ``mu0`` (any point in the simplex).
    """
    values = np.asarray(values, dtype=float)
    xy = grid._xy
    lifted = np.column_stack([xy, values])
    # QJ joggles the input to break coplanar/cocircular degeneracies (e.g. when
    # v is linear, all lifted points are coplanar and the hull would otherwise
    # be lower-dimensional). The joggle is tiny (<1e-13) and does not affect
    # the oracle tolerances.
    hull = ConvexHull(lifted, qhull_options="QJ")
    normals = hull.equations[:, :3]
    offsets = hull.equations[:, 3]
    # Upper facets: outward normal with +z component (top-facing surface).
    upper = np.where(normals[:, 2] > 1e-9)[0]
    upper_facets = hull.simplices[upper]          # (F, 3) vertex indices into lifted
    facet_xy = lifted[upper_facets][:, :, :2]     # (F, 3, 2) the (x,y) of each facet's verts
    upper_normals = normals[upper]
    upper_offsets = offsets[upper]

    def evaluate(mu0: np.ndarray) -> float:
        mu0 = np.asarray(mu0, dtype=float)
        p = mu0[:2]
        best = -np.inf
        for fi in range(len(upper)):
            a, b, c = upper_normals[fi]
            if abs(c) < 1e-12:
                continue
            if _point_in_triangle(p, facet_xy[fi]):
                z = -(a * p[0] + b * p[1] + upper_offsets[fi]) / c
                if z > best:
                    best = z
        if best == -np.inf:
            # Fallback: should not happen for mu0 strictly inside the simplex;
            # if it does (numerical edge), return the raw value at the nearest grid pt.
            nearest = int(np.argmin(np.linalg.norm(grid.mu - mu0, axis=1)))
            return float(values[nearest])
        return float(best)

    return EnvelopeFunction(evaluate=evaluate)


def qcav_simplex(grid: SimplexGrid, values: np.ndarray) -> EnvelopeFunction:
    """Quasiconcave envelope of ``v`` over the 3-state simplex.

    The transparent-motive cheap-talk (Lipnowski & Ravid 2020) sender value is
    the quasiconcave envelope.  Over the simplex,

        qcav v(mu) = sup { s : mu in conv{ mu' : v(mu') >= s } },

    i.e. the largest level ``s`` whose superlevel set's convex hull contains
    ``mu``.  Computed by scanning the distinct value levels top-down and
    testing containment via a Delaunay triangulation of each superlevel set.

    Args:
        grid: a :class:`SimplexGrid` on which ``values`` are sampled.
        values: sender value at each grid point, shape ``(grid.size,)``.

    Returns:
        An :class:`EnvelopeFunction` whose ``.evaluate(mu0)`` returns the
        quasiconcave envelope at belief ``mu0``.
    """
    values = np.asarray(values, dtype=float)
    xy = grid._xy
    distinct_levels = np.unique(values)
    distinct_levels = np.sort(distinct_levels)[::-1]  # descending

    # Precompute, per level, the Delaunay triangulation of the superlevel set.
    # Cache so evaluate() is cheap across many queries (needed for the chi grid).
    level_triangulations = []  # list of (level, Delaunay | None)
    for lvl in distinct_levels:
        mask = values >= lvl - 1e-12
        pts = xy[mask]
        if len(pts) < 3:
            level_triangulations.append((float(lvl), None))
            continue
        try:
            # QJ joggles to handle collinear/coplanar superlevel sets (e.g. when
            # the set lies entirely on a simplex edge, Delaunay would otherwise fail).
            tri = Delaunay(pts, qhull_options="QJ")
            level_triangulations.append((float(lvl), tri))
        except Exception:
            level_triangulations.append((float(lvl), None))

    def evaluate(mu0: np.ndarray) -> float:
        mu0 = np.asarray(mu0, dtype=float)
        p = np.array([mu0[0], mu0[1]])
        # Scan top-down; return the first (highest) level whose convex hull
        # contains mu0. Points exactly on a hull boundary are inside (tolerance).
        for lvl, tri in level_triangulations:
            if tri is None:
                # Degenerate superlevel set: check if any of its points matches.
                mask = values >= lvl - 1e-12
                if np.any(mask) and np.any(np.all(np.isclose(xy[mask], p, atol=1e-9), axis=1)):
                    return lvl
                continue
            if tri.find_simplex(p) >= 0:
                return lvl
        # If no superlevel hull contains mu0, return the global min (floor).
        return float(values.min())

    return EnvelopeFunction(evaluate=evaluate)
