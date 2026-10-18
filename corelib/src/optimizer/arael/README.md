# arael optimizer model

The optimization models behind `corelib/src/optimizer/OptimizerArael.cpp`,
written against the [arael](https://crates.io/crates/arael) nonlinear
least-squares library. `src/lib.rs` defines them; `capi/` and `cxx/`
are the C interface and C++ headers generated from it by
[cargo-arael](https://crates.io/crates/cargo-arael). The models use the
Ceres backend's residuals and weighting, so reported costs are directly
comparable.

## Building

Built only when rtabmap is configured with `-DWITH_ARAEL=ON`. The build
needs a Rust toolchain; cargo fetches the pinned dependencies
(Cargo.lock) from crates.io on the first build.

After changing `src/lib.rs`, regenerate the interface:

```
cargo install cargo-arael
cargo arael export
```

## Parameters

| parameter | default | meaning |
|---|---|---|
| `Optimizer/Strategy` | | `5` selects this backend |
| `Arael/Preset` | 2 | damping preset: 1=conservative, 2=well conditioned, 3=ill conditioned |
| `Arael/InitialLambda` | 0 | initial damping; 0 = the preset's value |
| `Arael/Threads` | -1 | solve threads: -1 = arael's default (`ARAEL_NUM_THREADS`, else 1), 0 = every core, n = n threads |
| `Arael/PoseGraphSingleThread` | true | graph optimization runs on one thread whatever `Arael/Threads` says; bundle adjustment follows `Arael/Threads` |
| `Optimizer/Robust` | false | robust loss on loop-closure links (see below) |
| `Arael/RobustLoss` | 0 | that loss: 0=Geman-McClure, 1=Cauchy |
| `Arael/RobustC2` | 0 | its squared scale; 0 = the loss default (2.99 Geman-McClure, 1.5 Cauchy) |
| `Arael/BaSolve` | 0 | bundle adjustment solver: 0=direct, 1=conjugate gradients (iterative inexact method; least memory) |
| `Arael/BaRobustLoss` | 1 | bundle adjustment loss: 0=Geman-McClure, 1=Cauchy, 2=Huber |
| `Arael/BaRobustC2` | 256 | its squared scale; also the outlier-report threshold |
| `Arael/BaPruneChi2` | 0 | 0 = outliers only reported; >0 = remove observations above this chi2 after the first solve and solve again |

The shared parameters apply as usual (`Optimizer/Iterations`,
`Optimizer/Epsilon`, `Reg/Force3DoF`, `Optimizer/VarianceIgnored`; for
bundle adjustment `Optimizer/PixelVariance`,
`Optimizer/DisparityVariance`, `Optimizer/Baseline`).
`Optimizer/RobustKernelDelta` is only a switch here: 0 turns the
bundle adjustment loss off; the value itself is not used.

## Scope

- Pose-pose constraints, pose priors (full or position-only), gravity
  links and landmarks, with the same conventions as the g2o and GTSAM
  backends (`Optimizer/PriorsIgnored`, `Optimizer/GravitySigma`,
  `Optimizer/LandmarksIgnored` apply).
- Marginal covariance of the newest pose, like g2o and GTSAM.
- Bundle adjustment, selected as strategy 5 by `Vis/BundleAdjustment`,
  `OdomF2M/BundleAdjustment` or the tools' `--ba_strategy`: the Ceres
  backend's residuals (mono, stereo, pose-graph links, multi-camera
  rigs, planar mode). Unlike the other backends, arael runs one robust
  solve: the loss weighs outliers down and they are only reported, not
  removed. Set `Arael/BaPruneChi2` to also remove them and solve
  again.

## Robust mode

`Optimizer/Robust=true` puts a robust loss on loop-closure links
instead of Vertigo switchable constraints. As with Vertigo, set
`RGBD/OptimizeMaxError` to 0 so suspect closures reach the loss
instead of being rejected outright:

```
--Optimizer/Strategy 5 --Optimizer/Robust true --RGBD/OptimizeMaxError 0
```

- **Geman-McClure** (default): effectively switches a bad closure
  off. On rtabmap's corrupted-closure test database it lands about as
  close to GTSAM-Vertigo's trajectory as g2o-Vertigo does, in a fraction
  of the time (Benchmark, Robust mode).
- **Cauchy**: limits a bad closure's influence but never switches it
  off; it leaves some deformation on large corruptions.

## Tests

`corelib/test/test_optimizer.cpp` runs the backend through the
pose-graph suites and verifies against independent references: cost
parity with the Ceres backend, the marginal covariance against a dense
computation, and corrupted-loop-closure recovery at the same bar as
the Vertigo backends.

## Benchmark

Every backend on rtabmap's test data (`scripts/fetch_test_data.sh`),
measured with `rtabmap-optimizerBenchmark` on an AMD Ryzen 9 7950X3D under
Ubuntu 24.04. Every backend runs with rtabmap's default settings, which
are not necessarily each library's fastest configuration; TORO is not
included. Each solve runs in its own forked process after one untimed
solve of the same problem, so the times leave out one-time process costs.
A time is the median over 3 rounds of each round's median run.
Single-threaded runs are pinned to one core, with BLAS (the linear algebra
library under Ceres) and OpenMP at one thread and `GTSAM/Threads` 1.

### Pose graphs

Cold solve from an initial guess composed from the links. Each backend
is timed to convergence: the fewest iterations whose result comes within
1e-5 of that backend's own result after 1000 iterations, found on the
same problem, then timed at that count with 5 runs a round:

```
rtabmap-optimizerBenchmark --converge --reps 5 --2d data/tests/large_reduced_graph.g2o
rtabmap-optimizerBenchmark --converge --reps 5 data/tests/multisession_3it.db
```

The final cost is the Ceres backend's cost at each backend's output;
arael uses the same residuals. g2o and GTSAM minimize their own
pose-error definitions, so their converged result can sit above the
others' on this scale. arael, g2o and GTSAM compute the newest pose's
marginal covariance inside these times; Ceres returns a placeholder.

#### large_reduced_graph.g2o (2D, 15,879 poses, 25,782 links)

| backend | iterations | time to converge ms | final cost | peak memory MB |
|---|---|---|---|---|
| arael | 4 | 87.3 | 1659.9156 | 154 |
| Ceres | 7 | 480.3 | 1659.9293 | 203 |
| GTSAM | 4 | 446.6 | 1660.7499 | 194 |
| g2o | 11 | 398.8 | 1659.9250 | 166 |

#### large_mapping_session.g2o (2D, 9,032 poses, 11,949 links)

| backend | iterations | time to converge ms | final cost | peak memory MB |
|---|---|---|---|---|
| arael | 6 | 45.2 | 302.8294 | 83 |
| Ceres | 18 | 413.0 | 302.8298 | 109 |
| GTSAM | 6 | 267.2 | 302.8326 | 108 |
| g2o | 185 | 1883 | 16034.3054 | 87 |

#### multisession_3it (3D, 935 poses, 1,206 links)

| backend | iterations | time to converge ms | final cost | peak memory MB |
|---|---|---|---|---|
| arael | 4 | 9.35 | 349.1045 | 42 |
| Ceres | 27 | 130.8 | 349.1068 | 48 |
| GTSAM | 3 | 20.7 | 349.1037 | 52 |
| g2o | 44 | 144.1 | 351.7099 | 42 |

#### loop_3it_gps (3D, 299 poses, 301 links)

| backend | iterations | time to converge ms | final cost | peak memory MB |
|---|---|---|---|---|
| arael | 2 | 1.75 | 33.5625 | 34 |
| Ceres | 7 | 9.73 | 33.5625 | 40 |
| GTSAM | 2 | 6.39 | 33.5625 | 46 |
| g2o | 12 | 8.74 | 33.6751 | 35 |

#### 2loops_workspace_3IT (3D, 127 poses, 204 links)

| backend | iterations | time to converge ms | final cost | peak memory MB |
|---|---|---|---|---|
| arael | 5 | 1.77 | 123.0414 | 33 |
| Ceres | 5 | 5.07 | 123.0414 | 38 |
| GTSAM | 3 | 4.14 | 123.0453 | 45 |
| g2o | 7 | 3.96 | 132.4484 | 33 |

#### robust_graph_optimization_stereo (3D, 111 poses, 144 links) [1]

| backend | iterations | time to converge ms | final cost | peak memory MB |
|---|---|---|---|---|
| arael | 20 | 3.44 | 11765.3531 | 33 |
| Ceres | 18 | 10.3 | 11765.3650 | 38 |
| GTSAM | 28 | 15.3 | 11811.1206 | 44 |
| g2o | 59 | 15.1 | 19864.0439 | 32 |

[1] Deliberately corrupted loop closures (Vertigo demo data). Every
backend runs without a robust loss here, so the optimum itself is
pulled by the bad closures; robust mode follows.

### Robust mode

`Optimizer/Robust true` on the corrupted-closure database: g2o and GTSAM
with Vertigo switchable constraints, arael with its loss. These are
different methods for the same task, not one problem, so the table shows
where each lands and how long it takes. The database has no ground
truth; the distance is the aligned translational RMSE from GTSAM-Vertigo's
trajectory, solved in the same run (`--ref-strategy 2`).

#### robust_graph_optimization_stereo

| method | iterations | time to converge ms | distance from GTSAM, Vertigo cm |
|---|---|---|---|
| arael, Geman-McClure | 10 | 2.03 | 1.23 |
| arael, Cauchy | 5 | 1.35 | 1.35 |
| GTSAM, Vertigo | 15 | 11.3 | 0.00 |
| g2o, Vertigo | 20 | 6.24 | 1.29 |

### Bundle adjustment as shipped

The bundle adjustment problem is built once from the database features
and every backend solves that same problem in a forked process, at most
20 iterations (`Optimizer/Iterations`). The backends handle outliers
differently, as rtabmap ships them: g2o, Ceres and GTSAM run at most 5
iterations, remove the observations above `Optimizer/RobustKernelDelta`,
then run at most 20; arael runs one solve of at most 25 under its robust
loss and removes nothing, reporting the observations above
`Arael/BaRobustC2` (flagged). So the times measure different work, and
these tables compare rtabmap's bundle adjustment as it ships, not solvers
on one problem.

```
rtabmap-optimizerBenchmark --ba --iterations 20 --reps 3 data/tests/2loops_workspace_3IT.db
```

The pixel error is the left-image reprojection error, over the
observations each backend kept and over all observations; p50 and p99
are the errors that 50% and 99% of them stay under. Neither view is an
accuracy verdict: these maps have no ground truth, a backend that removes
observations fits the rest more closely, and the removed ones count
against it over all observations. Feature matching makes the problem
differ by a few observations from run to run; all backends of one run
share it. arael-cg is `Arael/BaSolve` 1. GTSAM runs once per database:
its bundle adjustment takes minutes on the larger databases.

#### multisession_3it (155,613 observations)

| backend | observations removed | flagged, kept | kept p50 | kept p99 | all p50 | all p99 | time ms |
|---|---|---|---|---|---|---|---|
| arael | 0 (0.0%) | 25 | 0.60 | 3.79 | 0.60 | 3.79 | 1370 |
| arael-cg | 0 (0.0%) | 25 | 0.60 | 3.80 | 0.60 | 3.80 | 11337 |
| Ceres | 9,278 (6.0%) | 0 | 0.54 | 2.44 | 0.57 | 49.0 | 47331 |
| GTSAM | 6,901 (4.4%) | 0 | 0.53 | 2.37 | 0.55 | 9.18 | 589029 |
| g2o | 12,624 (8.1%) | 0 | 0.49 | 2.16 | 0.53 | 62.7 | 2653 |

#### 2loops_workspace_3IT (25,601 observations)

| backend | observations removed | flagged, kept | kept p50 | kept p99 | all p50 | all p99 | time ms |
|---|---|---|---|---|---|---|---|
| arael | 0 (0.0%) | 0 | 0.74 | 3.94 | 0.74 | 3.94 | 73.5 |
| arael-cg | 0 (0.0%) | 0 | 0.74 | 3.94 | 0.74 | 3.94 | 134.7 |
| Ceres | 1,568 (6.1%) | 0 | 0.66 | 2.42 | 0.70 | 35.2 | 1592 |
| GTSAM | 1,767 (6.9%) | 0 | 0.63 | 2.39 | 0.68 | 26.1 | 35347 |
| g2o | 2,420 (9.5%) | 0 | 0.61 | 2.23 | 0.68 | 46.5 | 314.3 |

#### robust_graph_optimization_stereo (11,410 observations) [2]

| backend | observations removed | flagged, kept | kept p50 | kept p99 | all p50 | all p99 | time ms |
|---|---|---|---|---|---|---|---|
| arael | 0 (0.0%) | 301 | 0.77 | 854.0 | 0.77 | 854.0 | 93.3 |
| arael-cg | 0 (0.0%) | 285 | 0.78 | 914.9 | 0.78 | 914.9 | 201.2 |
| Ceres | 5,053 (44.3%) | 0 | 0.42 | 1.96 | 0.98 | 3643.2 | 468.4 |
| GTSAM | 6,180 (54.2%) | 0 | 0.39 | 1.99 | 1.94 | 6062.2 | 38080 |
| g2o | 3,510 (30.8%) | 0 | 0.41 | 2.02 | 0.65 | 869.5 | 129.8 |

#### loop_3it_gps (6,342 observations)

| backend | observations removed | flagged, kept | kept p50 | kept p99 | all p50 | all p99 | time ms |
|---|---|---|---|---|---|---|---|
| arael | 0 (0.0%) | 0 | 0.48 | 2.21 | 0.48 | 2.21 | 28.7 |
| arael-cg | 0 (0.0%) | 0 | 0.48 | 2.19 | 0.48 | 2.19 | 1321 |
| Ceres | 30 (0.5%) | 0 | 0.48 | 2.00 | 0.48 | 2.16 | 479.3 |
| GTSAM | 39 (0.6%) | 0 | 0.46 | 2.00 | 0.46 | 2.25 | 6683 |
| g2o | 47 (0.7%) | 0 | 0.45 | 1.97 | 0.46 | 2.32 | 115.8 |

[2] Deliberately corrupted loop closures. g2o, Ceres and GTSAM remove a
third to a half of the observations and fit the rest closely; arael
keeps them all, and the corruption shows in its p99.

### Threads

arael against itself at 1, 2, 4 and 8 threads (`Arael/Threads`; pose
graphs with `Arael/PoseGraphSingleThread` false; bundle adjustment as
shipped), not pinned, 3 rounds. Every thread count reaches the same
pose-graph result in the same iterations.

| case | 1 thread ms | 2 threads ms | 4 threads ms | 8 threads ms | speedup at 8 |
|---|---|---|---|---|---|
| large_reduced_graph.g2o | 100.6 | 85.5 | 77.4 | 71.3 | 1.41x |
| large_mapping_session.g2o | 49.4 | 46.7 | 39.8 | 37.5 | 1.32x |
| multisession_3it | 10.1 | 9.80 | 8.31 | 8.16 | 1.24x |
| 2loops_workspace_3IT | 2.02 | 2.34 | 2.29 | 2.56 | 0.79x |
| bundle adjustment multisession_3it | 1281 | 821.1 | 590.3 | 496.5 | 2.58x |
| bundle adjustment 2loops_workspace_3IT | 82.8 | 47.8 | 36.3 | 26.8 | 3.09x |
| arael-cg multisession_3it | 9097 | 9959 | 9834 | 8303 | 1.10x |
| arael-cg 2loops_workspace_3IT | 146.6 | 102.7 | 80.7 | 87.9 | 1.67x |

Pose graphs run on one thread by default (`Arael/PoseGraphSingleThread`):
threads gain less there than on bundle adjustment and slow the smallest
graph down.

## In-pipeline replays

End-to-end replays of two integration-test databases
(`corelib/test/test_rtabmap_integration.cpp`) on the same machine: rtabmap
as it ships, with the backend doing graph optimization, the visual
registration bundle adjustment and the odometry's local bundle
adjustment. Each backend replays in its own process, the order rotating
between rounds; median of 3 rounds, rtabmap's default thread settings,
not pinned. Odometry time is per frame. RMSE is the final trajectory's
error against ground truth (Stereo20Hz has none); replays differ from run
to run, and the range is the lowest to the highest over the rounds.

#### PR2_Scan2D_Stereo (261 frames, stereo + 2D laser)

| backend | odometry ms/frame | RMSE m | RMSE range m | wall s |
|---|---|---|---|---|
| arael | 14.9 | 0.0302 | 0.0291 to 0.0308 | 7.0 |
| Ceres | 20.5 | 0.0326 | 0.0318 to 0.0339 | 8.5 |
| GTSAM | 60.6 | 0.0361 | 0.0321 to 0.0365 | 19.2 |
| g2o | 20.0 | 0.0351 | 0.0325 to 0.0392 | 8.4 |

#### Stereo20Hz (1035 frames, 20 Hz stereo)

| backend | odometry ms/frame | wall s |
|---|---|---|
| arael | 13.9 | 22.8 |
| Ceres | 20.2 | 29.8 |
| GTSAM | 63.1 | 75.4 |
| g2o | 18.7 | 28.3 |
