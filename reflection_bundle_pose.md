## Pose estimation for the mirror bundle (summary)

Goal: the bundle moves as a rigid body (rotation `R`, translation `t`). Each mirror keeps its relative pose. Incoming ray is fixed in world as `d_in = (0, 0, -1)` hitting the mirror center. Config provides a set of reflected rays as pairs of points `(p0, p1)`; assignment “which ray belongs to which mirror” is unknown. Find `R, t` and assignment that best fit the observed rays.

### Geometry
- Local mirror data (known): center `c_i` (local), normal `n_i_local` from `cut/tilt`.
- Bundle pose: `n_i = R n_i_local`, `c_i_world = R c_i + t`.
- Reflection: `d_out_i = d_in - 2 (n_i · d_in) n_i` (all unit vectors).
- Observed ray j: direction `d_j = normalize(p1 - p0)`, anchor along line `p_j(s) = p0 + s d_j`.

### Quick normal guess from observed ray
From reflection formula, an observed outgoing direction gives a candidate normal (up to noise):
`n_est_j = normalize(d_in - d_j)`. These normals will be matched to rotated local normals.

### Cost for matching mirror i ↔ ray j
- Angular term: `cost_ang = 1 - (n_i · n_est_j)` or squared angle between `d_out_i` and `d_j`.
- Positional term: project predicted center onto observed line: `s* = d_j · (c_i_world - p0)`, `p_closest = p0 + s* d_j`, `dist = ||c_i_world - p_closest||`.
- Total cost: `w_ang * cost_ang + w_pos * dist` (choose weights).

### Solver outline (iterative EM-style)
1) Precompute all `n_i_local`, `c_i`.
2) Initialization:
   - Compute all `n_est_j` from observed rays.
   - Initial assignment via Hungarian on angular cost between `n_i_local` (assuming R=I) and `n_est_j`.
   - Solve initial rotation `R` with Kabsch aligning matched normals; set `t` so that mean projected centers land on corresponding closest points.
3) Iteration:
   - Given `R, t`, compute `n_i`, `c_i_world`, `d_out_i`.
   - Build cost matrix with angular+positional terms, solve assignment (Hungarian, one-to-one).
   - Refine `R`: run Kabsch on pairs `(n_i_local -> assigned n_est_j)`; refine `t`: for each pair, compute `p_closest` on ray, then set `t` that minimizes Σ|| (R c_i + t) - p_closest ||² (closed form: `t = mean(p_closest - R c_i)`).
   - Repeat until assignment stable or cost change < ε; use a few random restarts to avoid local minima.
4) Output best `R, t`, assignment, and residuals (mean angular error, mean distance to lines).

### Notes
- Parameterize `R` as quaternion or axis-angle to keep it orthonormal during optimization.
- If the number of observed rays differs from mirrors, pad with dummy rows/cols in Hungarian to allow “unassigned” with a large penalty.
- Validation: after solving, plot predicted rays from centers with `d_out_i` and check distances to observed lines; residuals should be small.
