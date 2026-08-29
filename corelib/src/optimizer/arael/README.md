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
  off. On the corrupted-closure database below it matches
  GTSAM-Vertigo to 1.2 cm, in 5.4 ms vs 80.7 ms.
- **Cauchy**: limits a bad closure's influence but never switches it
  off; it leaves some deformation on large corruptions.

## Tests

`corelib/test/test_optimizer.cpp` runs the backend through the
pose-graph suites and verifies against independent references: cost
parity with the Ceres backend, the marginal covariance against a dense
computation, and corrupted-loop-closure recovery at the same bar as
the Vertigo backends.

## Benchmark

`rtabmap-optimizerBenchmark <db> --reps 5` on the databases fetched by
`scripts/fetch_test_data.sh`: cold solve of the database's pose graph
from an odometry-composed initial guess, one link per pair, median of
5 forked runs, single-threaded. Cost is the Ceres-convention cost
evaluated at each backend's output.

Cells: solve ms / iterations / final cost / maxrss MB.

| database | poses/links | arael | Ceres | GTSAM | g2o | TORO |
|---|---|---|---|---|---|---|
| 2loops_workspace_3IT | 127/204 | 3.0 / 6 / 123.0414 / 31 | 4.8 / 6 / 123.0414 / 42 | 12.0 / 6 / 123.0452 / 53 | 4.2 / 8 / 132.45 / 34 | 8.6 / 100 / 1165.0 / 28 |
| loop_3it_gps | 299/301 | 3.2 / 3 / 33.5624 / 32 | 9.2 / 8 / 33.5625 / 43 | 13.8 / 6 / 33.5624 / 54 | 9.8 / 13 / 33.68 / 36 | 9.7 / 100 / 5049.0 / 29 |
| multisession_3it | 935/1206 | 10.1 / 5 / 349.1020 / 38 | 99.7 / 28 / 349.1068 / 51 | 30.2 / 6 / 349.1031 / 63 | 110.1 / 46 / 351.71 / 43 | 44.7 / 100 / 578.2 / 33 |
| robust_graph_optimization_stereo [1] | 111/144 | 3.6 / 21 / 11765.26 / 30 | 8.9 / 19 / 11765.37 / 41 | 40.9 / 77 / 11811.2 / 53 | 15.1 / 60 / 19864.0 / 34 | 4.9 / 100 / 19091.7 / 28 |

arael, g2o and GTSAM compute the newest pose's marginal covariance
inside these times; Ceres and TORO return a placeholder.

[1] Deliberately corrupted loop closures (Vertigo demo data); all
backends run non-robust here, so the optimum itself is
outlier-contaminated.

## Bundle adjustment benchmark

`rtabmap-optimizerBenchmark --ba --reps 5 <db>` on the same databases:
the BA problem is built once from the database features and every
backend solves that identical problem in a forked child, with its
default settings. Scored with pixel-error percentiles over ALL
observations at each backend's output, so removing an observation
cannot improve a score. Solve time excludes the shared problem
construction.

Cells: solve ms / px p50 / p90 / p99 / maxrss MB. arael-cg is
`Arael/BaSolve` 1.

| database | obs | g2o | Ceres | arael | arael-cg |
|---|---|---|---|---|---|
| 2loops_workspace_3IT | 25.6k | 457 / 0.68 / 2.0 / 47.4 / 71 | 1098 / 0.70 / 1.9 / 35.1 / 69 | 59 / 0.74 / 1.9 / 3.9 / 56 | 97 / 0.74 / 1.9 / 3.9 / 54 |
| loop_3it_gps | 6.4k | 207 / 0.46 / 1.1 / 2.1 / 46 | 509 / 0.48 / 1.2 / 2.0 / 44 | 24 / 0.48 / 1.2 / 2.1 / 40 | 1120 / 0.48 / 1.2 / 2.1 / 38 |
| robust_graph_optimization_stereo [1] | 11.4k | 401 / 0.83 / 115.5 / 430 / 52 | 1604 / 0.81 / 645.2 / 6284 / 50 | 92 / 0.76 / 3.0 / 1281 / 43 | 418 / 0.76 / 2.9 / 1173 / 42 |
| multisession_3it | 155.4k | 7847 / 0.53 / 1.6 / 43.2 / 274 | 48902 / 0.57 / 1.6 / 9.5 / 267 | 1362 / 0.60 / 1.7 / 3.8 / 222 | 17951 / 0.60 / 1.7 / 3.8 / 194 |

[1] Deliberately corrupted loop closures. g2o and Ceres remove
observations and re-solve; the corruption claims a tenth or more of
their observations (elevated p90). arael's loss confines it to the
tail (clean p90, elevated p99).

GTSAM is not in this table: probes measured solves from 34 s (2loops)
to about 20 minutes (multisession).

## In-pipeline replays

End-to-end replays of the integration-test databases
(`corelib/test/test_rtabmap_integration.cpp`), with the backend doing
graph optimization and the local bundle adjustment inside the odometry
loop. Median of 3 runs; odometry time is per frame. RMSE is against
ground truth (Stereo20Hz has none).

PR2_Scan2D_Stereo (261 frames, stereo + 2D laser):

| backend | odometry ms/frame | RMSE m | wall s |
|---|---|---|---|
| arael | 8.3 | 0.038 | 4.5 |
| g2o | 13.8 | 0.036 | 5.9 |
| Ceres | 13.1 | 0.036 | 5.6 |
| GTSAM | 38.8 | 0.036 | 12.5 |

Stereo20Hz (1035 frames, 20 Hz stereo):

| backend | odometry ms/frame | wall s |
|---|---|---|
| arael | 7.7 | 14.4 |
| g2o | 11.6 | 18.6 |
| Ceres | 11.4 | 18.4 |
| GTSAM | 34.4 | 42.5 |
