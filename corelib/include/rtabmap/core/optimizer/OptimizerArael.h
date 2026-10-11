#ifndef OPTIMIZERARAEL_H_
#define OPTIMIZERARAEL_H_

#include <rtabmap/core/Optimizer.h>

namespace rtabmap {

/**
 * @brief Pose-graph optimizer backed by the arael nonlinear
 * least-squares library (Rust, vendored under corelib/src/optimizer/arael).
 *
 * Solves the same optimization problem as @ref OptimizerCeres --
 * identical residuals and sqrt-information weighting in both 2D and 3D,
 * so reported costs are comparable number for number. Beyond that
 * scope it supports absolute pose priors (full or position-only, the
 * same conventions as the g2o back-end; priors take over the gauge
 * from the fixed root), gravity links (Optimizer/GravitySigma),
 * landmarks (pose or point, the g2o back-end's conventions), and
 * computes the newest pose's marginal covariance like the g2o and
 * GTSAM back-ends. Robust mode puts a
 * robust loss (Geman-McClure or
 * Cauchy, see Arael/RobustLoss) on loop-closure links instead of
 * Vertigo switchable constraints -- no extra variables, the loss
 * reweights bad closures toward zero influence. Bundle adjustment
 * (@ref optimizeBA) reproduces the Ceres back-end's residuals; unlike
 * the other back-ends it runs one full robust solve (Cauchy loss by
 * default, see Arael/BaRobustLoss and Arael/BaRobustC2) and only
 * reports outliers, instead of the two-pass reject-and-resolve flow --
 * Optimizer/RobustKernelDelta acts as the robustness on/off switch,
 * and Arael/BaPruneChi2 optionally restores a prune-and-resolve
 * second stage.
 */
class RTABMAP_CORE_EXPORT OptimizerArael : public Optimizer
{
public:
	static bool available();

public:
	OptimizerArael(
			int iterations         = Parameters::defaultOptimizerIterations(),
			bool slam2d            = Parameters::defaultRegForce3DoF(),
			bool covarianceIgnored = Parameters::defaultOptimizerVarianceIgnored(),
			double epsilon         = Parameters::defaultOptimizerEpsilon()) :
		Optimizer(iterations, slam2d, covarianceIgnored, epsilon),
		preset_(Parameters::defaultAraelPreset()),
		initialLambda_(Parameters::defaultAraelInitialLambda()),
		threads_(Parameters::defaultAraelThreads()),
		poseGraphSingleThread_(Parameters::defaultAraelPoseGraphSingleThread()),
		robustLoss_(Parameters::defaultAraelRobustLoss()),
		robustC2_(Parameters::defaultAraelRobustC2()),
		baSolve_(Parameters::defaultAraelBaSolve()),
		baRobustLoss_(Parameters::defaultAraelBaRobustLoss()),
		baRobustC2_(Parameters::defaultAraelBaRobustC2()),
		baPruneChi2_(Parameters::defaultAraelBaPruneChi2()),
		pixelVariance_(Parameters::defaultOptimizerPixelVariance()),
		disparityVariance_(Parameters::defaultOptimizerDisparityVariance()),
		robustKernelDelta_(Parameters::defaultOptimizerRobustKernelDelta()),
		baseline_(Parameters::defaultOptimizerBaseline()) {}
	OptimizerArael(const ParametersMap & parameters) :
		Optimizer(parameters),
		preset_(Parameters::defaultAraelPreset()),
		initialLambda_(Parameters::defaultAraelInitialLambda()),
		threads_(Parameters::defaultAraelThreads()),
		poseGraphSingleThread_(Parameters::defaultAraelPoseGraphSingleThread()),
		robustLoss_(Parameters::defaultAraelRobustLoss()),
		robustC2_(Parameters::defaultAraelRobustC2()),
		baSolve_(Parameters::defaultAraelBaSolve()),
		baRobustLoss_(Parameters::defaultAraelBaRobustLoss()),
		baRobustC2_(Parameters::defaultAraelBaRobustC2()),
		baPruneChi2_(Parameters::defaultAraelBaPruneChi2()),
		pixelVariance_(Parameters::defaultOptimizerPixelVariance()),
		disparityVariance_(Parameters::defaultOptimizerDisparityVariance()),
		robustKernelDelta_(Parameters::defaultOptimizerRobustKernelDelta()),
		baseline_(Parameters::defaultOptimizerBaseline())
	{
		parseParameters(parameters);
	}
	virtual ~OptimizerArael() {}

	virtual Type type() const {return kTypeArael;}

	virtual void parseParameters(const ParametersMap & parameters);

	/** @brief The active damping/termination preset (see Arael/Preset). */
	int preset() const {return preset_;}

	/** @brief Initial LM damping override; 0 means the preset's value
	 *  (see Arael/InitialLambda). */
	double initialLambda() const {return initialLambda_;}

	/** @brief Solve threads: -1 = arael's default (ARAEL_NUM_THREADS,
	 *  else 1), 0 = every core, n = n (see Arael/Threads). */
	int threads() const {return threads_;}

	/** @brief Whether graph optimization runs on one thread whatever
	 *  threads() says (see Arael/PoseGraphSingleThread). */
	bool poseGraphSingleThread() const {return poseGraphSingleThread_;}

	/** @brief Robust loss for loop closures when isRobust():
	 *  0=Geman-McClure, 1=Cauchy (see Arael/RobustLoss). */
	int robustLoss() const {return robustLoss_;}

	/** @brief Squared robust-loss scale; 0 means the loss-specific
	 *  default (see Arael/RobustC2). */
	double robustC2() const {return robustC2_;}

	/** @brief Bundle adjustment linear solver: 0=auto, 1=conjugate
	 *  gradients on the reduced camera system (see Arael/BaSolve). */
	int baSolve() const {return baSolve_;}

	/** @brief BA observation robust loss: 0=Geman-McClure, 1=Cauchy,
	 *  2=Huber (see Arael/BaRobustLoss). */
	int baRobustLoss() const {return baRobustLoss_;}

	/** @brief Squared scale of the BA robust loss, and the outlier
	 *  report threshold (see Arael/BaRobustC2). */
	double baRobustC2() const {return baRobustC2_;}

	/** @brief 0 = single robust solve; >0 = prune observations above
	 *  this chi2 after the first solve and re-solve
	 *  (see Arael/BaPruneChi2). */
	double baPruneChi2() const {return baPruneChi2_;}

	virtual std::map<int, Transform> optimize(
			int rootId,
			const std::map<int, Transform> & poses,
			const std::multimap<int, Link> & edgeConstraints,
			cv::Mat & outputCovariance,
			std::list<std::map<int, Transform> > * intermediateGraphes = 0,
			double * finalError = 0,
			int * iterationsDone = 0);

	virtual std::map<int, Transform> optimizeBA(
			int rootId,
			const std::map<int, Transform> & poses,
			const std::multimap<int, Link> & links,
			const std::map<int, std::vector<CameraModel> > & models,
			std::map<int, cv::Point3f> & points3DMap,
			const std::map<int, std::map<int, FeatureBA> > & wordReferences,
			BAOutliers * outliers = 0);

private:
	int preset_;
	double initialLambda_;
	int threads_;
	bool poseGraphSingleThread_;
	int robustLoss_;
	double robustC2_;
	int baSolve_;
	int baRobustLoss_;
	double baRobustC2_;
	double baPruneChi2_;
	double pixelVariance_;
	double disparityVariance_;
	double robustKernelDelta_;
	double baseline_;
};

} /* namespace rtabmap */
#endif /* OPTIMIZERARAEL_H_ */
