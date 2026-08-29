//! 2D and 3D pose-graph models for the rtabmap arael optimizer backend.
//!
//! Both constraints reproduce the rtabmap Ceres backend's formulation
//! exactly (corelib/src/optimizer/ceres/pose_graph_2d and pose_graph_3d):
//! the translation residual in pose a's frame and, in 3D, the doubled
//! quaternion-vector rotation error, all scaled by the upper-triangular
//! square root S of the information matrix. Total cost is then comparable
//! number for number. The gauge is fixed from the C++ side by freezing the
//! root pose's parameters, matching Ceres' SetParameterBlockConstant.

use arael::angle::AngleParam;
use arael::matrix::matrix3d;
use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::transform::TransformParam;
use arael::vect::{vect2d, vect3d};

/// A 2D pose; the heading is optimized directly as an angle.
///
/// `prior_mode` arms an absolute prior on the pose: 0=none, 1=full
/// (x, y, theta; `ps00..ps22` = upper sqrt information), 2=position
/// only (x, y; `ps00`, `ps01`, `ps11`).
#[arael::model]
#[arael(constraint(hb, name = "prior_full", guard = self.prior_mode == 1, {
    let r0 = pose2.pos.x - pose2.pr_x;
    let r1 = pose2.pos.y - pose2.pr_y;
    let r2 = rad_diff(pose2.rot.angle, pose2.pr_th);
    [pose2.ps00 * r0 + pose2.ps01 * r1 + pose2.ps02 * r2,
     pose2.ps11 * r1 + pose2.ps12 * r2,
     pose2.ps22 * r2]
}))]
#[arael(constraint(hb, name = "prior_pos", guard = self.prior_mode == 2, {
    let r0 = pose2.pos.x - pose2.pr_x;
    let r1 = pose2.pos.y - pose2.pr_y;
    [pose2.ps00 * r0 + pose2.ps01 * r1,
     pose2.ps11 * r1]
}))]
#[derive(Default)]
pub struct Pose2 {
    pub pos: Param<vect2d>,
    pub rot: AngleParam<f64>,
    pub prior_mode: i32,
    pub pr_x: f64,
    pub pr_y: f64,
    pub pr_th: f64,
    pub ps00: f64,
    pub ps01: f64,
    pub ps02: f64,
    pub ps11: f64,
    pub ps12: f64,
    pub ps22: f64,
    pub hb: SelfBlock<Pose2>,
}

/// One relative SE2 measurement of pose b in pose a's frame.
/// (dx, dy, dth) is the measurement; s00..s22 are the entries of the
/// upper-triangular square root of the 3x3 information matrix, row major.
#[arael::model]
#[arael(constraint(hb, {
    let local = a.rot.rotation_matrix.transpose() * (b.pos - a.pos);
    let r0 = local.x - edge2.dx;
    let r1 = local.y - edge2.dy;
    let r2 = rad_diff(b.rot.angle, a.rot.angle + edge2.dth);
    [edge2.s00 * r0 + edge2.s01 * r1 + edge2.s02 * r2,
     edge2.s11 * r1 + edge2.s12 * r2,
     edge2.s22 * r2]
}))]
#[derive(Default)]
pub struct Edge2 {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose2>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose2>,
    pub dx: f64,
    pub dy: f64,
    pub dth: f64,
    pub s00: f64,
    pub s01: f64,
    pub s02: f64,
    pub s11: f64,
    pub s12: f64,
    pub s22: f64,
    pub hb: CrossBlock<Pose2, Pose2>,
}

/// A robustified SE2 closure: the Edge2 residual under a Geman-McClure
/// (loss=1) or Cauchy (loss=2) loss with squared scale `c2`, applied to
/// the squared norm of the whitened residual.
#[arael::model]
#[arael(constraint(hb, name = "gm", guard = self.loss == 1,
    loss = |s| loss_geman_mcclure(s, robustedge2.c2), {
    let local = a.rot.rotation_matrix.transpose() * (b.pos - a.pos);
    let r0 = local.x - robustedge2.dx;
    let r1 = local.y - robustedge2.dy;
    let r2 = rad_diff(b.rot.angle, a.rot.angle + robustedge2.dth);
    [robustedge2.s00 * r0 + robustedge2.s01 * r1 + robustedge2.s02 * r2,
     robustedge2.s11 * r1 + robustedge2.s12 * r2,
     robustedge2.s22 * r2]
}))]
#[arael(constraint(hb, name = "cauchy", guard = self.loss == 2,
    loss = |s| loss_cauchy(s, robustedge2.c2), {
    let local = a.rot.rotation_matrix.transpose() * (b.pos - a.pos);
    let r0 = local.x - robustedge2.dx;
    let r1 = local.y - robustedge2.dy;
    let r2 = rad_diff(b.rot.angle, a.rot.angle + robustedge2.dth);
    [robustedge2.s00 * r0 + robustedge2.s01 * r1 + robustedge2.s02 * r2,
     robustedge2.s11 * r1 + robustedge2.s12 * r2,
     robustedge2.s22 * r2]
}))]
#[derive(Default)]
pub struct RobustEdge2 {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose2>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose2>,
    pub dx: f64,
    pub dy: f64,
    pub dth: f64,
    pub s00: f64,
    pub s01: f64,
    pub s02: f64,
    pub s11: f64,
    pub s12: f64,
    pub s22: f64,
    pub loss: i32,
    pub c2: f64,
    pub hb: CrossBlock<Pose2, Pose2>,
}

/// A 2D point landmark (a landmark whose orientation is unused).
#[arael::model]
#[derive(Default)]
pub struct LmPoint2 {
    pub pos: Param<vect2d>,
    pub hb: SelfBlock<LmPoint2>,
}

/// An observation of a 2D point landmark in the observing pose's frame:
/// `(mx, my)` is the measured landmark position in that frame,
/// `s00/s01/s11` the upper sqrt information of its 2x2 block.
#[arael::model]
#[arael(constraint(hb, {
    let local = a.rot.rotation_matrix.transpose() * (lm.pos - a.pos);
    let r0 = local.x - lmobs2.mx;
    let r1 = local.y - lmobs2.my;
    [lmobs2.s00 * r0 + lmobs2.s01 * r1,
     lmobs2.s11 * r1]
}))]
#[derive(Default)]
pub struct LmObs2 {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose2>,
    #[arael(ref = root.lm_points)]
    pub lm: Ref<LmPoint2>,
    pub mx: f64,
    pub my: f64,
    pub s00: f64,
    pub s01: f64,
    pub s11: f64,
    pub hb: CrossBlock<Pose2, LmPoint2>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct Graph2 {
    pub poses: refs::Vec<Pose2>,
    pub edges: std::vec::Vec<Edge2>,
    pub robust_edges: std::vec::Vec<RobustEdge2>,
    pub lm_points: refs::Vec<LmPoint2>,
    pub lm_obs: std::vec::Vec<LmObs2>,
}

/// A 3D pose: translation and rotation stepped together as one SE(3)
/// transform.
///
/// `prior_mode` arms an absolute prior on the pose: 0=none, 1=full
/// (translation `pr_t`, rotation matrix `pr_r`; `pu_tt`/`pu_tr`/`pu_rr`
/// = blocks of the upper sqrt information, translation first), 2=position
/// only (`pr_t`; `pu_tt`). The rotation residual is the doubled vector
/// part of `q_prior * q^-1`, like the edges.
#[arael::model]
#[arael(constraint(hb, name = "prior_full", guard = self.prior_mode == 1, {
    let terr = pose3.pose.translation - pose3.pr_t;
    let rerr = pose3.pr_r * pose3.pose.rotation_matrix.transpose();
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let wt = pose3.pu_tt * terr + pose3.pu_tr * rrot;
    let wr = pose3.pu_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
#[arael(constraint(hb, name = "prior_pos", guard = self.prior_mode == 2, {
    let wt = pose3.pu_tt * (pose3.pose.translation - pose3.pr_t);
    [wt.x, wt.y, wt.z]
}))]
#[arael(constraint(hb, name = "gravity", guard = self.has_gravity, {
    let g = pose3.pose.rotation_matrix * pose3.gb;
    [g.x * pose3.gw, g.y * pose3.gw, (g.z - 1.0) * pose3.gw]
}))]
#[arael(constraint(hb, name = "rot_anchor", guard = self.has_anchor, {
    let e = pose3.pose.rotation_matrix.transpose() * pose3.an_r;
    let s = e - e.transpose();
    let denom = safe_sqrt(e[0].x + e[1].y + e[2].z + 1.0);
    [(s[2].y / denom) * pose3.an_wrp,
     (s[0].z / denom) * pose3.an_wrp,
     (s[1].x / denom) * pose3.an_wyaw]
}))]
#[derive(Default)]
pub struct Pose3 {
    pub pose: TransformParam<f64>,
    pub prior_mode: i32,
    pub pr_t: vect3d,
    pub pr_r: matrix3d,
    pub pu_tt: matrix3d,
    pub pu_tr: matrix3d,
    pub pu_rr: matrix3d,
    /// Gravity: `gb` is the measured body-frame up direction (unit), the
    /// residual pulls `R * gb` onto world up, weighted `gw` = 1/sigma.
    /// Yaw-invariant by construction.
    pub has_gravity: bool,
    pub gb: vect3d,
    pub gw: f64,
    /// Rotation anchor in the BODY frame (the error taking the pose's
    /// rotation to `an_r`, doubled quaternion vector): roll/pitch rows
    /// weighted `an_wrp`, yaw row `an_wyaw`. Body-frame roll/pitch
    /// corrections have no body-yaw component, so a tight `an_wyaw` pins
    /// the yaw gauge without fighting gravity.
    pub has_anchor: bool,
    pub an_r: matrix3d,
    pub an_wrp: f64,
    pub an_wyaw: f64,
    pub hb: SelfBlock<Pose3>,
}

/// One relative SE3 measurement of pose b in pose a's frame.
///
/// `dt` is the measured translation, `rmeas` the measured rotation
/// matrix. `u_tt`, `u_tr`, `u_rr` are the blocks of the upper-triangular
/// square root `S = [u_tt u_tr; 0 u_rr]` of the 6x6 information matrix,
/// ordered translation then rotation. The rotation residual is the
/// doubled vector part of `q_meas * q_estimated^-1`, extracted from the
/// error rotation matrix.
#[arael::model]
#[arael(constraint(hb, {
    let ra = a.pose.rotation_matrix;
    let rb = b.pose.rotation_matrix;
    let terr = ra.transpose() * (b.pose.translation - a.pose.translation) - edge3.dt;
    let rerr = edge3.rmeas * (rb.transpose() * ra);
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let wt = edge3.u_tt * terr + edge3.u_tr * rrot;
    let wr = edge3.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
#[derive(Default)]
pub struct Edge3 {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose3>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose3>,
    pub dt: vect3d,
    pub rmeas: matrix3d,
    pub u_tt: matrix3d,
    pub u_tr: matrix3d,
    pub u_rr: matrix3d,
    pub hb: CrossBlock<Pose3, Pose3>,
}

/// A robustified SE3 closure: the Edge3 residual under a Geman-McClure
/// (loss=1) or Cauchy (loss=2) loss with squared scale `c2`, applied to
/// the squared norm of the whitened residual.
#[arael::model]
#[arael(constraint(hb, name = "gm", guard = self.loss == 1,
    loss = |s| loss_geman_mcclure(s, robustedge3.c2), {
    let ra = a.pose.rotation_matrix;
    let rb = b.pose.rotation_matrix;
    let terr = ra.transpose() * (b.pose.translation - a.pose.translation) - robustedge3.dt;
    let rerr = robustedge3.rmeas * (rb.transpose() * ra);
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let wt = robustedge3.u_tt * terr + robustedge3.u_tr * rrot;
    let wr = robustedge3.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
#[arael(constraint(hb, name = "cauchy", guard = self.loss == 2,
    loss = |s| loss_cauchy(s, robustedge3.c2), {
    let ra = a.pose.rotation_matrix;
    let rb = b.pose.rotation_matrix;
    let terr = ra.transpose() * (b.pose.translation - a.pose.translation) - robustedge3.dt;
    let rerr = robustedge3.rmeas * (rb.transpose() * ra);
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let wt = robustedge3.u_tt * terr + robustedge3.u_tr * rrot;
    let wr = robustedge3.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
#[derive(Default)]
pub struct RobustEdge3 {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose3>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose3>,
    pub dt: vect3d,
    pub rmeas: matrix3d,
    pub u_tt: matrix3d,
    pub u_tr: matrix3d,
    pub u_rr: matrix3d,
    pub loss: i32,
    pub c2: f64,
    pub hb: CrossBlock<Pose3, Pose3>,
}

/// A 3D point landmark (a landmark whose orientation is unused).
#[arael::model]
#[derive(Default)]
pub struct LmPoint3 {
    pub pos: Param<vect3d>,
    pub hb: SelfBlock<LmPoint3>,
}

/// An observation of a 3D point landmark in the observing pose's frame:
/// `m` is the measured landmark position in that frame, `s` the upper
/// sqrt information of its 3x3 block.
#[arael::model]
#[arael(constraint(hb, {
    let local = a.pose.rotation_matrix.transpose() * (lm.pos - a.pose.translation);
    let r = lmobs3.s * (local - lmobs3.m);
    [r.x, r.y, r.z]
}))]
#[derive(Default)]
pub struct LmObs3 {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose3>,
    #[arael(ref = root.lm_points)]
    pub lm: Ref<LmPoint3>,
    pub m: vect3d,
    pub s: matrix3d,
    pub hb: CrossBlock<Pose3, LmPoint3>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct Graph3 {
    pub poses: refs::Vec<Pose3>,
    pub edges: std::vec::Vec<Edge3>,
    pub robust_edges: std::vec::Vec<RobustEdge3>,
    pub lm_points: refs::Vec<LmPoint3>,
    pub lm_obs: std::vec::Vec<LmObs3>,
}

// ---------------------------------------------------------------------------
// Bundle adjustment (BaGraph): cameras in the world-to-camera convention,
// 3D points, mono/stereo reprojection observations, between-camera links
// and an optional planar (body-z) constraint -- reproducing the Ceres
// backend's residuals (corelib/src/optimizer/ceres/bundle/).
// ---------------------------------------------------------------------------

/// A camera (one rig member of one pose), world-to-camera.
///
/// `has_planar` arms the 2D-mode constraint pinning the BODY z: `pl_d` is
/// `-R_bc^T t_bc` of the camera's local transform, `pl_z` the initial body
/// z, `pl_w` the weight.
#[arael::model]
#[arael(constraint(hb, name = "planar", guard = self.has_planar, {
    let bw = bacam.pose.rotation_matrix.transpose() * (bacam.pl_d - bacam.pose.translation);
    [(bw.z - bacam.pl_z) * bacam.pl_w]
}))]
#[derive(Default)]
pub struct BaCam {
    pub pose: TransformParam<f64>,
    pub has_planar: bool,
    pub pl_d: vect3d,
    pub pl_z: f64,
    pub pl_w: f64,
    pub hb: SelfBlock<BaCam>,
}

/// A 3D point (word).
#[arael::model]
#[derive(Default)]
pub struct BaPoint {
    pub pos: Param<vect3d>,
    pub hb: SelfBlock<BaPoint>,
}

/// A mono reprojection observation: `u`, `v` are the pixel coordinates
/// pre-centered on the principal point, `w_uv` = 1/sigma_pixel. `loss`
/// selects the kernel on the squared residual norm: 0 = Huber with knee
/// `k2` = delta^2 (a huge `k2` keeps it quadratic, i.e. no kernel),
/// 1 = Geman-McClure, 2 = Cauchy, both with squared scale `k2`. Clearing
/// `active` drops the observation (outlier rejection between passes).
#[arael::model]
#[arael(constraint(hb, name = "huber", guard = self.active && self.loss == 0,
    loss = |s| loss_huber(s, baobs.k2), {
    let p = cam.pose.rotation_matrix * point.pos + cam.pose.translation;
    [(baobs.fx * (p.x / p.z) - baobs.u) * baobs.w_uv,
     (baobs.fy * (p.y / p.z) - baobs.v) * baobs.w_uv]
}))]
#[arael(constraint(hb, name = "gm", guard = self.active && self.loss == 1,
    loss = |s| loss_geman_mcclure(s, baobs.k2), {
    let p = cam.pose.rotation_matrix * point.pos + cam.pose.translation;
    [(baobs.fx * (p.x / p.z) - baobs.u) * baobs.w_uv,
     (baobs.fy * (p.y / p.z) - baobs.v) * baobs.w_uv]
}))]
#[arael(constraint(hb, name = "cauchy", guard = self.active && self.loss == 2,
    loss = |s| loss_cauchy(s, baobs.k2), {
    let p = cam.pose.rotation_matrix * point.pos + cam.pose.translation;
    [(baobs.fx * (p.x / p.z) - baobs.u) * baobs.w_uv,
     (baobs.fy * (p.y / p.z) - baobs.v) * baobs.w_uv]
}))]
#[derive(Default)]
pub struct BaObs {
    #[arael(ref = root.cams)]
    pub cam: Ref<BaCam>,
    #[arael(ref = root.points)]
    pub point: Ref<BaPoint>,
    pub u: f64,
    pub v: f64,
    pub fx: f64,
    pub fy: f64,
    pub w_uv: f64,
    pub k2: f64,
    pub loss: u32,
    pub active: bool,
    pub hb: CrossBlock<BaCam, BaPoint>,
}

/// A stereo reprojection observation: mono rows plus a disparity row,
/// `bfx` = baseline * fx, `d` the observed disparity, `w_d` = 1/sigma_d.
/// `loss` as in [`BaObs`].
#[arael::model]
#[arael(constraint(hb, name = "huber", guard = self.active && self.loss == 0,
    loss = |s| loss_huber(s, bastereoobs.k2), {
    let p = cam.pose.rotation_matrix * point.pos + cam.pose.translation;
    [(bastereoobs.fx * (p.x / p.z) - bastereoobs.u) * bastereoobs.w_uv,
     (bastereoobs.fy * (p.y / p.z) - bastereoobs.v) * bastereoobs.w_uv,
     (bastereoobs.bfx / p.z - bastereoobs.d) * bastereoobs.w_d]
}))]
#[arael(constraint(hb, name = "gm", guard = self.active && self.loss == 1,
    loss = |s| loss_geman_mcclure(s, bastereoobs.k2), {
    let p = cam.pose.rotation_matrix * point.pos + cam.pose.translation;
    [(bastereoobs.fx * (p.x / p.z) - bastereoobs.u) * bastereoobs.w_uv,
     (bastereoobs.fy * (p.y / p.z) - bastereoobs.v) * bastereoobs.w_uv,
     (bastereoobs.bfx / p.z - bastereoobs.d) * bastereoobs.w_d]
}))]
#[arael(constraint(hb, name = "cauchy", guard = self.active && self.loss == 2,
    loss = |s| loss_cauchy(s, bastereoobs.k2), {
    let p = cam.pose.rotation_matrix * point.pos + cam.pose.translation;
    [(bastereoobs.fx * (p.x / p.z) - bastereoobs.u) * bastereoobs.w_uv,
     (bastereoobs.fy * (p.y / p.z) - bastereoobs.v) * bastereoobs.w_uv,
     (bastereoobs.bfx / p.z - bastereoobs.d) * bastereoobs.w_d]
}))]
#[derive(Default)]
pub struct BaStereoObs {
    #[arael(ref = root.cams)]
    pub cam: Ref<BaCam>,
    #[arael(ref = root.points)]
    pub point: Ref<BaPoint>,
    pub u: f64,
    pub v: f64,
    pub fx: f64,
    pub fy: f64,
    pub bfx: f64,
    pub d: f64,
    pub w_uv: f64,
    pub w_d: f64,
    pub k2: f64,
    pub loss: u32,
    pub active: bool,
    pub hb: CrossBlock<BaCam, BaPoint>,
}

/// A relative constraint between two cameras (pose-graph links folded into
/// BA, and the stiff rig edges of multi-camera setups). `dt`/`rmeas_t` are
/// the measured relative transform (cam b in cam a's frame; rotation
/// transposed), `u_*` the upper sqrt-information blocks [t; rot].
#[arael::model]
#[arael(constraint(hb, {
    let ra = a.pose.rotation_matrix;
    let rb = b.pose.rotation_matrix;
    let rab = ra * rb.transpose();
    let rerr = balink.rmeas_t * rab;
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let terr = a.pose.translation - rab * b.pose.translation - balink.dt;
    let wt = balink.u_tt * terr + balink.u_tr * rrot;
    let wr = balink.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
#[derive(Default)]
pub struct BaLink {
    #[arael(ref = root.cams)]
    pub a: Ref<BaCam>,
    #[arael(ref = root.cams)]
    pub b: Ref<BaCam>,
    pub dt: vect3d,
    pub rmeas_t: matrix3d,
    pub u_tt: matrix3d,
    pub u_tr: matrix3d,
    pub u_rr: matrix3d,
    pub hb: CrossBlock<BaCam, BaCam>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct BaGraph {
    pub cams: refs::Vec<BaCam>,
    pub points: refs::Vec<BaPoint>,
    pub obs: std::vec::Vec<BaObs>,
    pub stereo_obs: std::vec::Vec<BaStereoObs>,
    pub links: std::vec::Vec<BaLink>,
}
