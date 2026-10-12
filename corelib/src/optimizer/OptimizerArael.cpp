#include <rtabmap/core/optimizer/OptimizerArael.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>
#include <rtabmap/utilite/UStl.h>
#include <rtabmap/utilite/UMath.h>

#include <set>

#ifdef RTABMAP_ARAEL
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <graph2.hpp>
#include <graph3.hpp>
#include <ba_graph.hpp>
#include <arael/eigen.hpp>

#include <algorithm>
#endif

namespace rtabmap {

bool OptimizerArael::available()
{
#ifdef RTABMAP_ARAEL
	return true;
#else
	return false;
#endif
}

void OptimizerArael::parseParameters(const ParametersMap & parameters)
{
	Optimizer::parseParameters(parameters);
	Parameters::parse(parameters, Parameters::kAraelPreset(), preset_);
	if(preset_ < 1 || preset_ > 3)
	{
		UWARN("%s=%d is invalid (1=conservative, 2=well conditioned, "
			"3=ill conditioned), using default %d.",
			Parameters::kAraelPreset().c_str(), preset_,
			Parameters::defaultAraelPreset());
		preset_ = Parameters::defaultAraelPreset();
	}
	Parameters::parse(parameters, Parameters::kAraelInitialLambda(), initialLambda_);
	if(initialLambda_ < 0.0)
	{
		UWARN("%s=%f is invalid (must be >=0; 0 means the preset's value), using 0.",
			Parameters::kAraelInitialLambda().c_str(), initialLambda_);
		initialLambda_ = 0.0;
	}
	Parameters::parse(parameters, Parameters::kAraelThreads(), threads_);
	if(threads_ < -1)
	{
		UWARN("%s=%d is invalid (-1=arael's default, 0=every core, n=n threads), using -1.",
			Parameters::kAraelThreads().c_str(), threads_);
		threads_ = -1;
	}
	Parameters::parse(parameters, Parameters::kAraelPoseGraphSingleThread(), poseGraphSingleThread_);
	Parameters::parse(parameters, Parameters::kAraelRobustLoss(), robustLoss_);
	if(robustLoss_ < 0 || robustLoss_ > 1)
	{
		UWARN("%s=%d is invalid (0=Geman-McClure, 1=Cauchy), using default %d.",
			Parameters::kAraelRobustLoss().c_str(), robustLoss_,
			Parameters::defaultAraelRobustLoss());
		robustLoss_ = Parameters::defaultAraelRobustLoss();
	}
	Parameters::parse(parameters, Parameters::kAraelRobustC2(), robustC2_);
	if(robustC2_ < 0.0)
	{
		UWARN("%s=%f is invalid (must be >=0; 0 means the loss default), using 0.",
			Parameters::kAraelRobustC2().c_str(), robustC2_);
		robustC2_ = 0.0;
	}
	Parameters::parse(parameters, Parameters::kAraelBaSolve(), baSolve_);
	if(baSolve_ < 0 || baSolve_ > 1)
	{
		UWARN("%s=%d is invalid (0=auto, 1=conjugate gradients), using default %d.",
			Parameters::kAraelBaSolve().c_str(), baSolve_,
			Parameters::defaultAraelBaSolve());
		baSolve_ = Parameters::defaultAraelBaSolve();
	}
	Parameters::parse(parameters, Parameters::kAraelBaRobustLoss(), baRobustLoss_);
	if(baRobustLoss_ < 0 || baRobustLoss_ > 2)
	{
		UWARN("%s=%d is invalid (0=Geman-McClure, 1=Cauchy, 2=Huber), using default %d.",
			Parameters::kAraelBaRobustLoss().c_str(), baRobustLoss_,
			Parameters::defaultAraelBaRobustLoss());
		baRobustLoss_ = Parameters::defaultAraelBaRobustLoss();
	}
	Parameters::parse(parameters, Parameters::kAraelBaRobustC2(), baRobustC2_);
	if(baRobustC2_ <= 0.0)
	{
		UWARN("%s=%f is invalid (must be >0), using default %f.",
			Parameters::kAraelBaRobustC2().c_str(), baRobustC2_,
			Parameters::defaultAraelBaRobustC2());
		baRobustC2_ = Parameters::defaultAraelBaRobustC2();
	}
	Parameters::parse(parameters, Parameters::kAraelBaPruneChi2(), baPruneChi2_);
	if(baPruneChi2_ < 0.0)
	{
		UWARN("%s=%f is invalid (must be >=0; 0 disables pruning), using 0.",
			Parameters::kAraelBaPruneChi2().c_str(), baPruneChi2_);
		baPruneChi2_ = 0.0;
	}
	Parameters::parse(parameters, Parameters::kOptimizerPixelVariance(),     pixelVariance_);
	Parameters::parse(parameters, Parameters::kOptimizerDisparityVariance(), disparityVariance_);
	Parameters::parse(parameters, Parameters::kOptimizerRobustKernelDelta(), robustKernelDelta_);
	Parameters::parse(parameters, Parameters::kOptimizerBaseline(),          baseline_);
	UASSERT(pixelVariance_ > 0.0);
	UASSERT(disparityVariance_ > 0.0);
	UASSERT(baseline_ >= 0.0);
}

#ifdef RTABMAP_ARAEL
namespace {

// Per-attempt progress for the debug log; times each damped attempt.
// Costs are halved to the Ceres convention, like everything reported here.
struct IterLog
{
	UTimer timer;
};

bool logIteration(void * user, const arael::LmIterT<double> * attempt)
{
	IterLog * log = (IterLog *)user;
	// iteration/inner: the outer iteration and the damped attempt within it
	UDEBUG("iteration %u/%u: cost %f -> %f, lambda=%g, %s%s, %f ms",
		attempt->iter, attempt->inner,
		0.5 * attempt->cost, 0.5 * attempt->new_cost, attempt->lambda,
		attempt->accepted ? "accepted" : "rejected",
		attempt->factorization_failed ? " (factorization failed)" : "",
		log->timer.ticks() * 1000.0);
	return true;
}

std::map<int, Transform> optimize2d(
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & edgeConstraints,
		bool covarianceIgnored,
		bool priorsIgnored,
		bool landmarksIgnored,
		int closureLoss, // -1=off, 0=Geman-McClure, 1=Cauchy (closures only)
		double robustC2,
		rtabmap_arael::graph2::LmConfig cfg,
		cv::Mat & outputCovariance,
		double * startError,
		double * finalError,
		int * iterationsDone)
{
	namespace ra = rtabmap_arael::graph2;
	std::map<int, Transform> optimizedPoses;
	UTimer setupTimer;

	// absolute pose priors are self-links; position-only when the angular
	// variance marks the orientation unused, like the g2o back-end
	std::map<int, const Link *> priors;
	if(!priorsIgnored)
	{
		for(std::multimap<int, Link>::const_iterator iter=edgeConstraints.begin(); iter!=edgeConstraints.end(); ++iter)
		{
			if(iter->second.from() == iter->second.to() &&
				iter->second.from() > 0 &&
				iter->second.type() == Link::kPosePrior)
			{
				priors[iter->second.from()] = &iter->second;
			}
		}
	}

	// landmarks (negative ids): pose landmarks when the link's angular
	// variance is meaningful, point landmarks when it carries the 9999
	// marker, like the g2o back-end
	std::map<int, bool> landmarkWithRotation;
	if(!landmarksIgnored)
	{
		for(std::multimap<int, Link>::const_iterator iter=edgeConstraints.begin(); iter!=edgeConstraints.end(); ++iter)
		{
			if(iter->second.type() == Link::kLandmark)
			{
				const int lmId = iter->second.from() < 0 ?
						iter->second.from() : iter->second.to();
				if(lmId < 0 && landmarkWithRotation.find(lmId) == landmarkWithRotation.end())
				{
					landmarkWithRotation[lmId] =
						1.0 / iter->second.infMatrix().at<double>(5,5) < 9999.0;
				}
			}
		}
	}

	ra::Graph2 graph;
	std::map<int, uint32_t> index;   // pose vertices, pose landmarks included
	std::map<int, uint32_t> lmIndex; // point landmarks
	graph.poses().reserve(poses.size());
	for(std::map<int, Transform>::const_iterator iter = poses.begin(); iter!=poses.end(); ++iter)
	{
		if(iter->first < 0)
		{
			std::map<int, bool>::const_iterator lmType = landmarkWithRotation.find(iter->first);
			if(lmType == landmarkWithRotation.end())
			{
				continue; // landmarks ignored, or none of its links present
			}
			UASSERT(!iter->second.isNull());
			if(lmType->second)
			{
				uint32_t i = graph.poses().size();
				ra::Pose2 p = graph.poses().push();
				p.set_pos({iter->second.x(), iter->second.y()});
				p.set_rot_angle(iter->second.theta());
				index.insert(std::make_pair(iter->first, i));
			}
			else
			{
				uint32_t i = graph.lm_points().size();
				ra::LmPoint2 p = graph.lm_points().push();
				p.set_pos({iter->second.x(), iter->second.y()});
				lmIndex.insert(std::make_pair(iter->first, i));
			}
		}
		else if(iter->first > 0)
		{
			UASSERT(!iter->second.isNull());
			uint32_t i = graph.poses().size();
			ra::Pose2 p = graph.poses().push();
			p.set_pos({iter->second.x(), iter->second.y()});
			p.set_rot_angle(iter->second.theta());
			index.insert(std::make_pair(iter->first, i));

			std::map<int, const Link *>::const_iterator pit = priors.find(iter->first);
			if(pit != priors.end())
			{
				const Link & l = *pit->second;
				const cv::Mat & m = l.infMatrix();
				p.set_pr_x(l.transform().x());
				p.set_pr_y(l.transform().y());
				if(1.0 / m.at<double>(5,5) >= 9999.0)
				{
					Eigen::Matrix2d info = Eigen::Matrix2d::Identity();
					if(!covarianceIgnored)
					{
						info(0,0) = m.at<double>(0,0);
						info(0,1) = m.at<double>(0,1);
						info(1,0) = m.at<double>(1,0);
						info(1,1) = m.at<double>(1,1);
					}
					const Eigen::Matrix2d S = info.llt().matrixU();
					p.set_prior_mode(2);
					p.set_ps00(S(0,0));
					p.set_ps01(S(0,1));
					p.set_ps11(S(1,1));
				}
				else
				{
					Eigen::Matrix3d info = Eigen::Matrix3d::Identity();
					if(!covarianceIgnored)
					{
						const int idx[3] = {0, 1, 5};
						for(int r = 0; r < 3; ++r)
							for(int c = 0; c < 3; ++c)
								info(r, c) = m.at<double>(idx[r], idx[c]);
					}
					const Eigen::Matrix3d S = info.llt().matrixU();
					p.set_prior_mode(1);
					p.set_pr_th(l.transform().theta());
					p.set_ps00(S(0,0));
					p.set_ps01(S(0,1));
					p.set_ps02(S(0,2));
					p.set_ps11(S(1,1));
					p.set_ps12(S(1,2));
					p.set_ps22(S(2,2));
				}
			}
		}
	}

	graph.edges().reserve(edgeConstraints.size());
	for(std::multimap<int, Link>::const_iterator iter=edgeConstraints.begin(); iter!=edgeConstraints.end(); ++iter)
	{
		int id1 = iter->second.from();
		int id2 = iter->second.to();
		if(id1 != id2 && id1 > 0 && id2 > 0)
		{
			UASSERT(index.find(id1) != index.end() && index.find(id2) != index.end());

			Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
			if(!covarianceIgnored)
			{
				information(0,0) = iter->second.infMatrix().at<double>(0,0); // x-x
				information(0,1) = iter->second.infMatrix().at<double>(0,1); // x-y
				information(0,2) = iter->second.infMatrix().at<double>(0,5); // x-theta
				information(1,0) = iter->second.infMatrix().at<double>(1,0); // y-x
				information(1,1) = iter->second.infMatrix().at<double>(1,1); // y-y
				information(1,2) = iter->second.infMatrix().at<double>(1,5); // y-theta
				information(2,0) = iter->second.infMatrix().at<double>(5,0); // theta-x
				information(2,1) = iter->second.infMatrix().at<double>(5,1); // theta-y
				information(2,2) = iter->second.infMatrix().at<double>(5,5); // theta-theta
			}
			const Eigen::Matrix3d S = information.llt().matrixU();

			auto fill = [&](auto e) {
				e.set_a(graph.poses().ref_at(index.at(id1)));
				e.set_b(graph.poses().ref_at(index.at(id2)));
				e.set_dx(iter->second.transform().x());
				e.set_dy(iter->second.transform().y());
				e.set_dth(iter->second.transform().theta());
				e.set_s00(S(0,0));
				e.set_s01(S(0,1));
				e.set_s02(S(0,2));
				e.set_s11(S(1,1));
				e.set_s12(S(1,2));
				e.set_s22(S(2,2));
				return e;
			};
			if(closureLoss >= 0 &&
				iter->second.type() != Link::kNeighbor &&
				iter->second.type() != Link::kNeighborMerged)
			{
				ra::RobustEdge2 e = fill(graph.robust_edges().push());
				e.set_loss(closureLoss);
				e.set_c2(robustC2);
			}
			else
			{
				fill(graph.edges().push());
			}
		}
		else if(!landmarksIgnored && iter->second.type() == Link::kLandmark &&
			((id1 > 0 && id2 < 0) || (id1 < 0 && id2 > 0)))
		{
			// normalize to node -> landmark, like the GTSAM back-end
			const int nodeId = id1 > 0 ? id1 : id2;
			const int lmId   = id1 > 0 ? id2 : id1;
			const Transform t = id1 > 0 ?
					iter->second.transform() : iter->second.transform().inverse();
			UASSERT(index.find(nodeId) != index.end());
			const cv::Mat & m = iter->second.infMatrix();
			if(index.find(lmId) != index.end())
			{
				// pose landmark: a regular relative edge (robustified with
				// the closures when robust mode is on, like g2o's
				// switchable landmark edges)
				Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
				if(!covarianceIgnored)
				{
					const int idx[3] = {0, 1, 5};
					for(int r = 0; r < 3; ++r)
						for(int c = 0; c < 3; ++c)
							information(r, c) = m.at<double>(idx[r], idx[c]);
				}
				const Eigen::Matrix3d S = information.llt().matrixU();
				auto fillLm = [&](auto e) {
					e.set_a(graph.poses().ref_at(index.at(nodeId)));
					e.set_b(graph.poses().ref_at(index.at(lmId)));
					e.set_dx(t.x());
					e.set_dy(t.y());
					e.set_dth(t.theta());
					e.set_s00(S(0,0));
					e.set_s01(S(0,1));
					e.set_s02(S(0,2));
					e.set_s11(S(1,1));
					e.set_s12(S(1,2));
					e.set_s22(S(2,2));
					return e;
				};
				if(closureLoss >= 0)
				{
					ra::RobustEdge2 e = fillLm(graph.robust_edges().push());
					e.set_loss(closureLoss);
					e.set_c2(robustC2);
				}
				else
				{
					fillLm(graph.edges().push());
				}
			}
			else if(lmIndex.find(lmId) != lmIndex.end())
			{
				// point landmark observation
				Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
				if(!covarianceIgnored)
				{
					information(0,0) = m.at<double>(0,0);
					information(0,1) = m.at<double>(0,1);
					information(1,0) = m.at<double>(1,0);
					information(1,1) = m.at<double>(1,1);
				}
				const Eigen::Matrix2d S = information.llt().matrixU();
				ra::LmObs2 o = graph.lm_obs().push();
				o.set_a(graph.poses().ref_at(index.at(nodeId)));
				o.set_lm(graph.lm_points().ref_at(lmIndex.at(lmId)));
				o.set_mx(t.x());
				o.set_my(t.y());
				o.set_s00(S(0,0));
				o.set_s01(S(0,1));
				o.set_s11(S(1,1));
			}
		}
		//else // priors and gravity handled above
	}

	// gauge freedom: hold one pose constant, same as the Ceres back-end's
	// SetParameterBlockConstant -- unless priors anchor the graph, in which
	// case the g2o/GTSAM back-ends drop the fixed root and so do we
	const bool rootFixed = priors.empty();
	std::map<int, uint32_t>::iterator rootIter = index.end();
	if(rootFixed)
	{
		rootIter = rootId>0?index.find(rootId):index.lower_bound(1);
		UASSERT(rootIter != index.end());
		ra::Pose2 p = graph.poses()[rootIter->second];
		p.set_pos_optimize(false);
		p.set_rot_angle_optimize(false);
	}

	// marginal covariance of the newest pose, like the g2o/GTSAM back-ends,
	// assembled by the solve; it does not exist when the newest pose is the
	// fixed gauge anchor
	const bool newestIsRoot = rootFixed && index.rbegin()->second == rootIter->second;
	if(!newestIsRoot)
	{
		cfg.covariance = ra::CovMode::PerQuery;
		// minimum degree directly: pricing it against nested dissection
		// costs more than it saves on rtabmap's graphs
		cfg.covariance_ordering = ra::CovOrdering::Amd;
	}

	// root id 0: no fixed root (priors anchor the graph); threads=0: every core
	UDEBUG("2d graph: %zu poses, %zu edges (%zu robust), root id=%d, initial lambda=%g, threads=%u, built in %f ms",
		(size_t)graph.poses().size(), (size_t)graph.edges().size(),
		(size_t)graph.robust_edges().size(), rootIter != index.end() ? rootIter->first : 0,
		cfg.initial_lambda, cfg.num_threads, setupTimer.ticks() * 1000.0);
	if(cfg.observer_user)
	{
		// the observer times attempt-to-attempt; align its clock with the
		// solve so the first iteration excludes the graph construction above
		((IterLog *)cfg.observer_user)->timer.start();
	}
	ra::SolveResult result = graph.solve_sparse(cfg);
	if(result.is_err())
	{
		UWARN("arael: solve failed: %s", result.error().message);
		return optimizedPoses;
	}
	UDEBUG("2d solve: %s, cost %f -> %f, %u iterations, final lambda=%g",
		arael::as_str(result->status), 0.5 * result->start_cost,
		0.5 * result->end_cost, result->iterations, result->final_lambda);
	// arael reports the sum of squares; Ceres convention halves it
	*startError = 0.5 * result->start_cost;
	*finalError = 0.5 * result->end_cost;
	*iterationsDone = (int)result->iterations;

	for(std::map<int, Transform>::const_iterator iter = poses.begin(); iter!=poses.end(); ++iter)
	{
		if(index.count(iter->first)) // poses and pose landmarks
		{
			ra::Pose2 p = graph.poses()[index.at(iter->first)];
			ra::vect2d pos = p.pos();
			float roll, pitch, yaw;
			iter->second.getEulerAngles(roll, pitch, yaw);
			Transform newPose(pos.x, pos.y, iter->second.z(), roll, pitch, p.rot_angle());
			UASSERT_MSG(!newPose.isNull(), uFormat("Optimized pose %d is null!?!?", iter->first).c_str());
			optimizedPoses.insert(std::pair<int, Transform>(iter->first, newPose));
		}
		else if(lmIndex.count(iter->first))
		{
			// point landmark: optimized position, orientation kept from input
			ra::vect2d pos = graph.lm_points()[lmIndex.at(iter->first)].pos();
			float roll, pitch, yaw;
			iter->second.getEulerAngles(roll, pitch, yaw);
			optimizedPoses.insert(std::pair<int, Transform>(iter->first,
					Transform(pos.x, pos.y, iter->second.z(), roll, pitch, yaw)));
		}
	}

	if(newestIsRoot)
	{
		// the g2o back-end warns and keeps identity here too
		UWARN("Newest pose %d is the fixed root. Cannot compute last pose covariance.",
			index.rbegin()->first);
		return optimizedPoses;
	}
	auto cov = result->covariance();
	if(cov.is_err())
	{
		if(rootFixed)
		{
			// with a hard anchor this signals a genuine rank problem: fail
			// the optimization, like the GTSAM back-end
			UERROR("arael: could not compute marginal covariance: %s",
				cov.error().message);
			optimizedPoses.clear();
		}
		else
		{
			// a prior-anchored graph can leave gauge directions the priors
			// do not pin (e.g. position-only priors leave rotation free);
			// the solution is still valid -- keep identity, like g2o
			UWARN("arael: could not compute marginal covariance: %s",
				cov.error().message);
		}
		return optimizedPoses;
	}
	auto m = cov->marginal(graph.poses()[index.rbegin()->second]);
	if(m.is_err())
	{
		UERROR("arael: could not compute marginal covariance: %s",
			m.error().message);
		optimizedPoses.clear();
		return optimizedPoses;
	}
	UDEBUG("Computed marginals = %fs (key=%d)", result->timing.covariance, index.rbegin()->first);
	// (x, y, theta) into the (x, y, yaw) slots of the 6x6
	outputCovariance.at<double>(0,0) = m->rows[0].x; // x-x
	outputCovariance.at<double>(0,1) = m->rows[0].y; // x-y
	outputCovariance.at<double>(0,5) = m->rows[0].z; // x-theta
	outputCovariance.at<double>(1,0) = m->rows[1].x; // y-x
	outputCovariance.at<double>(1,1) = m->rows[1].y; // y-y
	outputCovariance.at<double>(1,5) = m->rows[1].z; // y-theta
	outputCovariance.at<double>(5,0) = m->rows[2].x; // theta-x
	outputCovariance.at<double>(5,1) = m->rows[2].y; // theta-y
	outputCovariance.at<double>(5,5) = m->rows[2].z; // theta-theta
	return optimizedPoses;
}

std::map<int, Transform> optimize3d(
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & edgeConstraints,
		bool covarianceIgnored,
		bool priorsIgnored,
		bool landmarksIgnored,
		float gravitySigma, // 0 = gravity links ignored
		int closureLoss, // -1=off, 0=Geman-McClure, 1=Cauchy (closures only)
		double robustC2,
		rtabmap_arael::graph3::LmConfig cfg,
		cv::Mat & outputCovariance,
		double * startError,
		double * finalError,
		int * iterationsDone)
{
	namespace ra = rtabmap_arael::graph3;
	std::map<int, Transform> optimizedPoses;
	UTimer setupTimer;

	// absolute pose priors and gravity constraints are self-links;
	// position-only priors when any angular variance marks the orientation
	// unused, like the g2o back-end
	std::map<int, const Link *> priors;
	std::map<int, const Link *> gravityLinks;
	std::map<int, bool> landmarkWithRotation;
	for(std::multimap<int, Link>::const_iterator iter=edgeConstraints.begin(); iter!=edgeConstraints.end(); ++iter)
	{
		if(iter->second.from() == iter->second.to() && iter->second.from() > 0)
		{
			if(!priorsIgnored && iter->second.type() == Link::kPosePrior)
			{
				priors[iter->second.from()] = &iter->second;
			}
			else if(gravitySigma > 0.0f && iter->second.type() == Link::kGravity)
			{
				gravityLinks[iter->second.from()] = &iter->second;
			}
		}
		else if(!landmarksIgnored && iter->second.type() == Link::kLandmark)
		{
			// pose landmark when the angular variances are meaningful,
			// point landmark on the 9999 marker, like the g2o back-end
			const int lmId = iter->second.from() < 0 ?
					iter->second.from() : iter->second.to();
			if(lmId < 0 && landmarkWithRotation.find(lmId) == landmarkWithRotation.end())
			{
				const cv::Mat & m = iter->second.infMatrix();
				landmarkWithRotation[lmId] =
					1.0 / m.at<double>(3,3) < 9999.0 &&
					1.0 / m.at<double>(4,4) < 9999.0 &&
					1.0 / m.at<double>(5,5) < 9999.0;
			}
		}
	}

	ra::Graph3 graph;
	std::map<int, uint32_t> index;   // pose vertices, pose landmarks included
	std::map<int, uint32_t> lmIndex; // point landmarks
	graph.poses().reserve(poses.size());
	for(std::map<int, Transform>::const_iterator iter = poses.begin(); iter!=poses.end(); ++iter)
	{
		if(iter->first < 0)
		{
			std::map<int, bool>::const_iterator lmType = landmarkWithRotation.find(iter->first);
			if(lmType == landmarkWithRotation.end())
			{
				continue; // landmarks ignored, or none of its links present
			}
			UASSERT(!iter->second.isNull());
			if(lmType->second)
			{
				uint32_t i = graph.poses().size();
				ra::Pose3 p = graph.poses().push();
				Eigen::Quaterniond q = iter->second.getQuaterniond();
				p.set_pose_translation({iter->second.x(), iter->second.y(), iter->second.z()});
				p.set_pose_rotation(ra::quaternd::from_eigen(q));
				index.insert(std::make_pair(iter->first, i));
			}
			else
			{
				uint32_t i = graph.lm_points().size();
				ra::LmPoint3 p = graph.lm_points().push();
				p.set_pos({iter->second.x(), iter->second.y(), iter->second.z()});
				lmIndex.insert(std::make_pair(iter->first, i));
			}
		}
		else if(iter->first > 0)
		{
			UASSERT(!iter->second.isNull());
			uint32_t i = graph.poses().size();
			ra::Pose3 p = graph.poses().push();
			Eigen::Quaterniond q = iter->second.getQuaterniond();
			p.set_pose_translation({iter->second.x(), iter->second.y(), iter->second.z()});
			p.set_pose_rotation(ra::quaternd::from_eigen(q));
			index.insert(std::make_pair(iter->first, i));

			std::map<int, const Link *>::const_iterator git = gravityLinks.find(iter->first);
			if(git != gravityLinks.end())
			{
				// measured body-frame up direction; using the full link
				// rotation, yaw cancels against world up by construction
				Eigen::Vector3d gb = git->second->transform().getQuaterniond()
						.toRotationMatrix().transpose() * Eigen::Vector3d::UnitZ();
				gb.normalize();
				p.set_has_gravity(true);
				p.set_gb(ra::vect3d::from_eigen(gb));
				p.set_gw(1.0 / gravitySigma);
			}

			std::map<int, const Link *>::const_iterator pit = priors.find(iter->first);
			if(pit != priors.end())
			{
				const Link & l = *pit->second;
				const cv::Mat & m = l.infMatrix();
				const Transform & t = l.transform();
				p.set_pr_t({t.x(), t.y(), t.z()});
				if(1.0 / m.at<double>(3,3) >= 9999.0 ||
					1.0 / m.at<double>(4,4) >= 9999.0 ||
					1.0 / m.at<double>(5,5) >= 9999.0)
				{
					Eigen::Matrix3d info = Eigen::Matrix3d::Identity();
					if(!covarianceIgnored)
					{
						for(int r = 0; r < 3; ++r)
							for(int c = 0; c < 3; ++c)
								info(r, c) = m.at<double>(r, c);
					}
					const Eigen::Matrix3d S = info.llt().matrixU();
					p.set_prior_mode(2);
					p.set_pu_tt(ra::matrix3d::from_eigen(S));
				}
				else
				{
					Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
					if(!covarianceIgnored)
					{
						memcpy(info.data(), m.data, m.total()*sizeof(double));
					}
					const Eigen::Matrix<double, 6, 6> S = info.llt().matrixU();
					p.set_prior_mode(1);
					p.set_pr_r(ra::matrix3d::from_eigen(t.getQuaterniond().toRotationMatrix()));
					p.set_pu_tt(ra::matrix3d::from_eigen(S.block<3,3>(0,0)));
					p.set_pu_tr(ra::matrix3d::from_eigen(S.block<3,3>(0,3)));
					p.set_pu_rr(ra::matrix3d::from_eigen(S.block<3,3>(3,3)));
				}
			}
		}
	}

	graph.edges().reserve(edgeConstraints.size());
	for(std::multimap<int, Link>::const_iterator iter=edgeConstraints.begin(); iter!=edgeConstraints.end(); ++iter)
	{
		int id1 = iter->second.from();
		int id2 = iter->second.to();
		if(id1 != id2 && id1 > 0 && id2 > 0)
		{
			UASSERT(index.find(id1) != index.end() && index.find(id2) != index.end());

			Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
			if(!covarianceIgnored)
			{
				memcpy(information.data(), iter->second.infMatrix().data, iter->second.infMatrix().total()*sizeof(double));
			}
			const Eigen::Matrix<double, 6, 6> S = information.llt().matrixU();

			const Transform & t = iter->second.transform();

			auto fill = [&](auto e) {
				e.set_a(graph.poses().ref_at(index.at(id1)));
				e.set_b(graph.poses().ref_at(index.at(id2)));
				e.set_dt({t.x(), t.y(), t.z()});
				e.set_rmeas(ra::matrix3d::from_eigen(t.getQuaterniond().toRotationMatrix()));
				e.set_u_tt(ra::matrix3d::from_eigen(S.block<3,3>(0,0)));
				e.set_u_tr(ra::matrix3d::from_eigen(S.block<3,3>(0,3)));
				e.set_u_rr(ra::matrix3d::from_eigen(S.block<3,3>(3,3)));
				return e;
			};
			if(closureLoss >= 0 &&
				iter->second.type() != Link::kNeighbor &&
				iter->second.type() != Link::kNeighborMerged)
			{
				ra::RobustEdge3 e = fill(graph.robust_edges().push());
				e.set_loss(closureLoss);
				e.set_c2(robustC2);
			}
			else
			{
				fill(graph.edges().push());
			}
		}
		else if(!landmarksIgnored && iter->second.type() == Link::kLandmark &&
			((id1 > 0 && id2 < 0) || (id1 < 0 && id2 > 0)))
		{
			// normalize to node -> landmark, like the GTSAM back-end
			const int nodeId = id1 > 0 ? id1 : id2;
			const int lmId   = id1 > 0 ? id2 : id1;
			const Transform lt = id1 > 0 ?
					iter->second.transform() : iter->second.transform().inverse();
			UASSERT(index.find(nodeId) != index.end());
			const cv::Mat & m = iter->second.infMatrix();
			if(index.find(lmId) != index.end())
			{
				// pose landmark: a regular relative edge (robustified with
				// the closures when robust mode is on, like g2o's
				// switchable landmark edges)
				Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
				if(!covarianceIgnored)
				{
					memcpy(information.data(), m.data, m.total()*sizeof(double));
				}
				const Eigen::Matrix<double, 6, 6> S = information.llt().matrixU();
				auto fillLm = [&](auto e) {
					e.set_a(graph.poses().ref_at(index.at(nodeId)));
					e.set_b(graph.poses().ref_at(index.at(lmId)));
					e.set_dt({lt.x(), lt.y(), lt.z()});
					e.set_rmeas(ra::matrix3d::from_eigen(lt.getQuaterniond().toRotationMatrix()));
					e.set_u_tt(ra::matrix3d::from_eigen(S.block<3,3>(0,0)));
					e.set_u_tr(ra::matrix3d::from_eigen(S.block<3,3>(0,3)));
					e.set_u_rr(ra::matrix3d::from_eigen(S.block<3,3>(3,3)));
					return e;
				};
				if(closureLoss >= 0)
				{
					ra::RobustEdge3 e = fillLm(graph.robust_edges().push());
					e.set_loss(closureLoss);
					e.set_c2(robustC2);
				}
				else
				{
					fillLm(graph.edges().push());
				}
			}
			else if(lmIndex.find(lmId) != lmIndex.end())
			{
				// point landmark observation
				Eigen::Matrix3d information = Eigen::Matrix3d::Identity();
				if(!covarianceIgnored)
				{
					for(int r = 0; r < 3; ++r)
						for(int c = 0; c < 3; ++c)
							information(r, c) = m.at<double>(r, c);
				}
				const Eigen::Matrix3d S = information.llt().matrixU();
				ra::LmObs3 o = graph.lm_obs().push();
				o.set_a(graph.poses().ref_at(index.at(nodeId)));
				o.set_lm(graph.lm_points().ref_at(lmIndex.at(lmId)));
				o.set_m({lt.x(), lt.y(), lt.z()});
				o.set_s(ra::matrix3d::from_eigen(S));
			}
		}
		//else // priors and gravity handled above
	}

	// priors anchor the gauge; without them, hold one pose constant (the
	// g2o/GTSAM back-ends drop the fixed root the same way). With gravity
	// constraints but no priors, the root's rotation must stay free so its
	// own gravity link can correct roll/pitch -- but yaw is then a free
	// gauge direction the whole graph can drift along, so pin the root's
	// yaw with a rotation prior using the GTSAM back-end's gravity-mode
	// weights (roll/pitch variance 2, yaw 1e-9); translation stays held.
	const bool rootFixed = priors.empty() && gravityLinks.empty();
	std::map<int, uint32_t>::iterator rootIter = index.end();
	if(priors.empty())
	{
		rootIter = rootId>0?index.find(rootId):index.lower_bound(1);
		UASSERT(rootIter != index.end());
		ra::Pose3 p = graph.poses()[rootIter->second];
		p.set_pose_optimize_translation(false);
		if(gravityLinks.empty())
		{
			p.set_pose_optimize_rotation(false);
		}
		else
		{
			p.set_has_anchor(true);
			p.set_an_r(ra::matrix3d::from_eigen(
				poses.at(rootIter->first).getQuaterniond().toRotationMatrix()));
			// the GTSAM back-end's gravity-mode root weights: roll/pitch
			// variance 2, yaw 1e-9
			p.set_an_wrp(1.0 / std::sqrt(2.0));
			p.set_an_wyaw(1.0 / std::sqrt(1e-9));
		}
	}

	// marginal covariance of the newest pose, like the g2o/GTSAM back-ends,
	// assembled by the solve; it does not exist when the newest pose is the
	// fixed gauge anchor
	const bool newestIsRoot = rootFixed && index.rbegin()->second == rootIter->second;
	if(!newestIsRoot)
	{
		cfg.covariance = ra::CovMode::PerQuery;
		// minimum degree directly: pricing it against nested dissection
		// costs more than it saves on rtabmap's graphs
		cfg.covariance_ordering = ra::CovOrdering::Amd;
	}

	// root id 0: no fixed root (priors anchor the graph); threads=0: every core
	UDEBUG("3d graph: %zu poses, %zu edges (%zu robust), root id=%d, initial lambda=%g, threads=%u, built in %f ms",
		(size_t)graph.poses().size(), (size_t)graph.edges().size(),
		(size_t)graph.robust_edges().size(), rootIter != index.end() ? rootIter->first : 0,
		cfg.initial_lambda, cfg.num_threads, setupTimer.ticks() * 1000.0);
	if(cfg.observer_user)
	{
		// the observer times attempt-to-attempt; align its clock with the
		// solve so the first iteration excludes the graph construction above
		((IterLog *)cfg.observer_user)->timer.start();
	}
	ra::SolveResult result = graph.solve_sparse(cfg);
	if(result.is_err())
	{
		UWARN("arael: solve failed: %s", result.error().message);
		return optimizedPoses;
	}
	UDEBUG("3d solve: %s, cost %f -> %f, %u iterations, final lambda=%g",
		arael::as_str(result->status), 0.5 * result->start_cost,
		0.5 * result->end_cost, result->iterations, result->final_lambda);
	*startError = 0.5 * result->start_cost;
	*finalError = 0.5 * result->end_cost;
	*iterationsDone = (int)result->iterations;

	for(std::map<int, Transform>::const_iterator iter = poses.begin(); iter!=poses.end(); ++iter)
	{
		if(index.count(iter->first)) // poses and pose landmarks
		{
			ra::Pose3 p = graph.poses()[index.at(iter->first)];
			ra::vect3d pos = p.pose_translation();
			ra::quaternd q = p.pose_rotation();
			Transform newPose(pos.x, pos.y, pos.z, q.v.x, q.v.y, q.v.z, q.t);
			UASSERT_MSG(!newPose.isNull(), uFormat("Optimized pose %d is null!?!?", iter->first).c_str());
			optimizedPoses.insert(std::pair<int, Transform>(iter->first, newPose));
		}
		else if(lmIndex.count(iter->first))
		{
			// point landmark: optimized position, orientation kept from input
			ra::vect3d pos = graph.lm_points()[lmIndex.at(iter->first)].pos();
			float roll, pitch, yaw;
			iter->second.getEulerAngles(roll, pitch, yaw);
			optimizedPoses.insert(std::pair<int, Transform>(iter->first,
					Transform(pos.x, pos.y, pos.z, roll, pitch, yaw)));
		}
	}

	if(newestIsRoot)
	{
		// the g2o back-end warns and keeps identity here too
		UWARN("Newest pose %d is the fixed root. Cannot compute last pose covariance.",
			index.rbegin()->first);
		return optimizedPoses;
	}
	auto cov = result->covariance();
	double m[36];
	if(cov.is_err() ||
		cov->marginal(graph.poses()[index.rbegin()->second], m, 36) != 6)
	{
		if(rootFixed)
		{
			// with a hard anchor this signals a genuine rank problem: fail
			// the optimization, like the GTSAM back-end
			UERROR("arael: could not compute marginal covariance%s%s",
				cov.is_err() ? ": " : "",
				cov.is_err() ? cov.error().message : "");
			optimizedPoses.clear();
		}
		else
		{
			// a prior-anchored graph can leave gauge directions the priors
			// do not pin (e.g. position-only priors leave rotation free);
			// the solution is still valid -- keep identity, like g2o
			UWARN("arael: could not compute marginal covariance%s%s",
				cov.is_err() ? ": " : "",
				cov.is_err() ? cov.error().message : "");
		}
		return optimizedPoses;
	}
	UDEBUG("Computed marginals = %fs (key=%d)", result->timing.covariance, index.rbegin()->first);
	// TransformParam's marginal is rotation-first ([w; d], the body-frame
	// twist); rtabmap wants translation first, like the GTSAM back-end's
	// permutation of gtsam's rotation-first tangent
	Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > mr(m);
	Eigen::Matrix<double, 6, 6> out;
	out.block(0,0,3,3) = mr.block(3,3,3,3); // cov translation
	out.block(3,3,3,3) = mr.block(0,0,3,3); // cov rotation
	out.block(0,3,3,3) = mr.block(3,0,3,3); // off diagonal
	out.block(3,0,3,3) = mr.block(0,3,3,3); // off diagonal
	memcpy(outputCovariance.data, out.data(), outputCovariance.total()*sizeof(double));
	return optimizedPoses;
}

} // namespace
#endif

std::map<int, Transform> OptimizerArael::optimize(
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & edgeConstraints,
		cv::Mat & outputCovariance,
		std::list<std::map<int, Transform> > * intermediateGraphes,
		double * finalError,
		int * iterationsDone)
{
	outputCovariance = cv::Mat::eye(6,6,CV_64FC1);
	std::map<int, Transform> optimizedPoses;
#ifdef RTABMAP_ARAEL
	UDEBUG("Optimizing graph (%s, preset=%d, pose=%d constraints=%d, max iterations=%d, epsilon=%f)...",
		isSlam2d()?"2d":"3d", preset(), (int)poses.size(), (int)edgeConstraints.size(),
		iterations(), epsilon());
	if(edgeConstraints.size()>=1 && poses.size()>=2 && iterations() > 0)
	{
		UTimer timer;
		double initialError = 0.0;
		double error = 0.0;
		int iterationsCount = 0;
		int closureLoss = -1;
		double robustC2Value = 0.0;
		if(isRobust())
		{
			closureLoss = robustLoss();
			robustC2Value = robustC2() > 0.0 ? robustC2()
					: (robustLoss() == 1 ? 1.5 : 2.99);
			UDEBUG("robust: %s loss on loop closures, c2=%f",
				robustLoss() == 1 ? "Cauchy" : "Geman-McClure", robustC2Value);
		}
		// at debug level, log every damped attempt; arael's own diagnostics
		// only go to stderr (the exported interface has no log-sink hook to
		// route them into ULogger), so keep them at their default level
		const bool debug = ULogger::level() == ULogger::kDebug;
		IterLog iterLog;
		if(isSlam2d())
		{
			rtabmap_arael::graph2::LmConfig cfg((arael::LmPreset)preset());
			cfg.max_iters = iterations();
			cfg.rel_precision = this->epsilon();
			if(poseGraphSingleThread())
			{
				cfg.num_threads = 1;
			}
			else if(threads() >= 0)
			{
				cfg.num_threads = threads();
			}
			if(initialLambda() > 0.0)
			{
				cfg.initial_lambda = initialLambda();
			}
			else if(closureLoss >= 0)
			{
				// robust losses flatten the cost surface around outliers;
				// near-Gauss-Newton first steps overshoot into poor local
				// minima, so start at conservative damping
				cfg.initial_lambda = 1e-4;
			}
			if(debug)
			{
				cfg.observer = logIteration;
				cfg.observer_user = &iterLog;
			}
			optimizedPoses = optimize2d(rootId, poses, edgeConstraints,
				isCovarianceIgnored(), priorsIgnored(), landmarksIgnored(),
				closureLoss, robustC2Value, cfg, outputCovariance,
				&initialError, &error, &iterationsCount);
		}
		else
		{
			rtabmap_arael::graph3::LmConfig cfg((arael::LmPreset)preset());
			cfg.max_iters = iterations();
			cfg.rel_precision = this->epsilon();
			if(poseGraphSingleThread())
			{
				cfg.num_threads = 1;
			}
			else if(threads() >= 0)
			{
				cfg.num_threads = threads();
			}
			if(initialLambda() > 0.0)
			{
				cfg.initial_lambda = initialLambda();
			}
			else if(closureLoss >= 0)
			{
				// robust losses flatten the cost surface around outliers;
				// near-Gauss-Newton first steps overshoot into poor local
				// minima, so start at conservative damping
				cfg.initial_lambda = 1e-4;
			}
			if(debug)
			{
				cfg.observer = logIteration;
				cfg.observer_user = &iterLog;
			}
			optimizedPoses = optimize3d(rootId, poses, edgeConstraints,
				isCovarianceIgnored(), priorsIgnored(), landmarksIgnored(),
				gravitySigma(), closureLoss, robustC2Value, cfg,
				outputCovariance, &initialError, &error, &iterationsCount);
		}
		if(finalError)
		{
			*finalError = error;
		}
		if(iterationsDone)
		{
			*iterationsDone = iterationsCount;
		}
		UINFO("arael optimizing end (%d iterations done (error initial=%f final=%f), time=%f s)",
			iterationsCount, initialError, error, timer.ticks());
	}
	else if(poses.size() == 1 || iterations() <= 0)
	{
		optimizedPoses = poses;
	}
	else
	{
		UWARN("This method should be called at least with 1 pose!");
	}
	UDEBUG("Optimizing graph...end!");
#else
	UERROR("Not built with arael support!");
#endif
	return optimizedPoses;
}

std::map<int, Transform> OptimizerArael::optimizeBA(
		int rootId,
		const std::map<int, Transform> & posesIn,
		const std::multimap<int, Link> & links,
		const std::map<int, std::vector<CameraModel> > & models,
		std::map<int, cv::Point3f> & points3DMap,
		const std::map<int, std::map<int, FeatureBA> > & wordReferences,
		BAOutliers * outliers)
{
	if(outliers)
	{
		outliers->clear();
	}
#ifdef RTABMAP_ARAEL
	namespace ra = rtabmap_arael::ba_graph;

	std::map<int, Transform> poses(posesIn.lower_bound(1), posesIn.end());

	ra::BaGraph graph;

	// one camera vertex per (pose, rig member), world-to-camera, like the
	// Ceres back-end
	std::map<std::pair<int,int>, uint32_t> camIdxByKey;
	for(std::map<int, Transform>::const_iterator iter=poses.begin(); iter!=poses.end(); ++iter)
	{
		std::map<int, std::vector<CameraModel> >::const_iterator iterModel = models.find(iter->first);
		UASSERT(iterModel != models.end() && !iterModel->second.empty());
		for(size_t c = 0; c < iterModel->second.size(); ++c)
		{
			const CameraModel & m = iterModel->second[c];
			UASSERT(m.isValidForProjection());
			const Transform t = (iter->second * m.localTransform()).inverse();
			const Eigen::Quaterniond q = t.getQuaterniond();
			uint32_t idx = graph.cams().size();
			ra::BaCam cam = graph.cams().push();
			cam.set_pose_translation({t.x(), t.y(), t.z()});
			cam.set_pose_rotation(ra::quaternd::from_eigen(q));
			camIdxByKey.insert(std::make_pair(std::make_pair(iter->first, (int)c), idx));
		}
	}

	std::map<int, uint32_t> pointIdToIndex;
	graph.points().reserve(points3DMap.size());
	for(std::map<int, cv::Point3f>::const_iterator kter = points3DMap.begin(); kter!=points3DMap.end(); ++kter)
	{
		uint32_t idx = graph.points().size();
		ra::BaPoint p = graph.points().push();
		p.set_pos({kter->second.x, kter->second.y, kter->second.z});
		pointIdToIndex.insert(std::make_pair(kter->first, idx));
	}

	// observations; per-observation metadata kept for the outlier pass
	struct ObsMeta
	{
		bool stereo;
		uint32_t graphIdx; // index in obs() or stereo_obs()
		int wordId;
		int poseId;
		uint32_t camIdx;
		uint32_t pointIdx;
		double u, v, fx, fy, bfx, d;
	};
	std::vector<ObsMeta> obsMeta;
	const double invSigmaUv = 1.0 / std::sqrt(pixelVariance_);
	const double invSigmaD  = 1.0 / std::sqrt(disparityVariance_);
	// Optimizer/RobustKernelDelta is only the robustness switch here (0 =
	// plain quadratic BA); the loss and its scale come from
	// Arael/BaRobustLoss and Arael/BaRobustC2. The model's loss tags are
	// Arael/BaRobustLoss's, plus 3 = no kernel.
	const bool robust = robustKernelDelta_ > 0.0;
	const double obsK2 = baRobustC2_;
	const uint32_t obsLoss = robust ? (uint32_t)baRobustLoss_ : 3u;
	int monoObsCount = 0, stereoObsCount = 0;
	for(std::map<int, std::map<int, FeatureBA> >::const_iterator iter=wordReferences.begin();
		iter!=wordReferences.end(); ++iter)
	{
		for(std::map<int, FeatureBA>::const_iterator jter=iter->second.begin();
			jter!=iter->second.end(); ++jter)
		{
			const int poseId = jter->first;
			const int camC = jter->second.cameraIndex;
			std::map<int, std::vector<CameraModel> >::const_iterator iterModel = models.find(poseId);
			UASSERT(iterModel != models.end());
			UASSERT(camC >= 0 && camC < (int)iterModel->second.size());
			const CameraModel & m = iterModel->second[camC];

			std::map<std::pair<int,int>, uint32_t>::const_iterator camIt =
					camIdxByKey.find(std::make_pair(poseId, camC));
			UASSERT(camIt != camIdxByKey.end());
			const uint32_t pointIdx = pointIdToIndex.at(iter->first);

			ObsMeta meta;
			meta.wordId = iter->first;
			meta.poseId = poseId;
			meta.camIdx = camIt->second;
			meta.pointIdx = pointIdx;
			meta.u = jter->second.kpt.pt.x - m.cx();
			meta.v = jter->second.kpt.pt.y - m.cy();
			meta.fx = m.fx();
			meta.fy = m.fy();
			meta.bfx = 0.0;
			meta.d = 0.0;

			const double Tx = m.Tx();
			const double depth = jter->second.depth;
			const double baseline = Tx < 0.0 ? (-Tx / meta.fx) : baseline_;
			meta.stereo = baseline > 0.0 && uIsFinite(depth) && depth > 0.0;
			if(meta.stereo)
			{
				meta.bfx = baseline * meta.fx;
				meta.d = meta.bfx / depth;
				meta.graphIdx = graph.stereo_obs().size();
				ra::BaStereoObs o = graph.stereo_obs().push();
				o.set_cam(graph.cams().ref_at(meta.camIdx));
				o.set_point(graph.points().ref_at(meta.pointIdx));
				o.set_u(meta.u);
				o.set_v(meta.v);
				o.set_fx(meta.fx);
				o.set_fy(meta.fy);
				o.set_bfx(meta.bfx);
				o.set_d(meta.d);
				o.set_w_uv(invSigmaUv);
				o.set_w_d(invSigmaD);
				o.set_k2(obsK2);
				o.set_loss(obsLoss);
				o.set_active(true);
				++stereoObsCount;
			}
			else
			{
				meta.graphIdx = graph.obs().size();
				ra::BaObs o = graph.obs().push();
				o.set_cam(graph.cams().ref_at(meta.camIdx));
				o.set_point(graph.points().ref_at(meta.pointIdx));
				o.set_u(meta.u);
				o.set_v(meta.v);
				o.set_fx(meta.fx);
				o.set_fy(meta.fy);
				o.set_w_uv(invSigmaUv);
				o.set_k2(obsK2);
				o.set_loss(obsLoss);
				o.set_active(true);
				++monoObsCount;
			}
			obsMeta.push_back(meta);
		}
	}
	UDEBUG("arael BA: %d mono + %d stereo observations", monoObsCount, stereoObsCount);

	// pose-graph links between cam 0 vertices, then the stiff rig edges of
	// multi-camera setups; both use the BaLink constraint
	auto addLink = [&](uint32_t idxA, uint32_t idxB, const Transform & camLink,
			const Eigen::Matrix<double, 6, 6> & S) {
		ra::BaLink e = graph.links().push();
		e.set_a(graph.cams().ref_at(idxA));
		e.set_b(graph.cams().ref_at(idxB));
		e.set_dt({camLink.x(), camLink.y(), camLink.z()});
		e.set_rmeas_t(ra::matrix3d::from_eigen(
			camLink.getQuaterniond().toRotationMatrix().transpose()));
		e.set_u_tt(ra::matrix3d::from_eigen(S.block<3,3>(0,0)));
		e.set_u_tr(ra::matrix3d::from_eigen(S.block<3,3>(0,3)));
		e.set_u_rr(ra::matrix3d::from_eigen(S.block<3,3>(3,3)));
	};
	int linkObsCount = 0;
	for(std::multimap<int, Link>::const_iterator iter=links.begin(); iter!=links.end(); ++iter)
	{
		const Link & link = iter->second;
		if(link.from() <= 0 || link.to() <= 0 || link.from() == link.to())
		{
			continue;
		}
		std::map<std::pair<int,int>, uint32_t>::const_iterator itA =
				camIdxByKey.find(std::make_pair(link.from(), 0));
		std::map<std::pair<int,int>, uint32_t>::const_iterator itB =
				camIdxByKey.find(std::make_pair(link.to(), 0));
		if(itA == camIdxByKey.end() || itB == camIdxByKey.end())
		{
			continue;
		}
		const Transform camLink = models.at(link.from())[0].localTransform().inverse() *
				link.transform() *
				models.at(link.to())[0].localTransform();
		Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
		if(!isCovarianceIgnored())
		{
			memcpy(information.data(), link.infMatrix().data, link.infMatrix().total()*sizeof(double));
		}
		addLink(itA->second, itB->second, camLink,
			Eigen::Matrix<double, 6, 6>(information.llt().matrixU()));
		++linkObsCount;
	}
	int rigEdgeCount = 0;
	{
		Eigen::Matrix<double, 6, 6> rigInfo = Eigen::Matrix<double, 6, 6>::Identity() * 9999999.0;
		const Eigen::Matrix<double, 6, 6> rigS = rigInfo.llt().matrixU();
		for(std::map<int, std::vector<CameraModel> >::const_iterator iter=models.begin(); iter!=models.end(); ++iter)
		{
			if(!uContains(poses, iter->first) || iter->second.size() < 2)
			{
				continue;
			}
			std::map<std::pair<int,int>, uint32_t>::const_iterator cam0It =
					camIdxByKey.find(std::make_pair(iter->first, 0));
			if(cam0It == camIdxByKey.end())
			{
				continue;
			}
			const Transform & lt0 = iter->second[0].localTransform();
			for(size_t c = 1; c < iter->second.size(); ++c)
			{
				std::map<std::pair<int,int>, uint32_t>::const_iterator camCIt =
						camIdxByKey.find(std::make_pair(iter->first, (int)c));
				if(camCIt == camIdxByKey.end())
				{
					continue;
				}
				addLink(cam0It->second, camCIt->second,
					lt0.inverse() * iter->second[c].localTransform(), rigS);
				++rigEdgeCount;
			}
		}
	}
	if(linkObsCount > 0 || rigEdgeCount > 0)
	{
		UDEBUG("arael BA: %d pose-graph links, %d rig edges", linkObsCount, rigEdgeCount);
	}

	// fixed cameras: rootId >= 0 fixes that pose, rootId < 0 fixes all but
	// -rootId (the optimizeBA() contract); all cameras of the rig
	int fixedCamCount = 0;
	for(std::map<int, Transform>::const_iterator iter=poses.begin(); iter!=poses.end(); ++iter)
	{
		const bool fixNode = (rootId >= 0 && iter->first == rootId) ||
				(rootId < 0 && iter->first != -rootId);
		if(!fixNode)
		{
			continue;
		}
		std::map<int, std::vector<CameraModel> >::const_iterator iterModel = models.find(iter->first);
		const size_t rigSize = iterModel != models.end() ? iterModel->second.size() : 0;
		for(size_t c = 0; c < rigSize; ++c)
		{
			std::map<std::pair<int,int>, uint32_t>::const_iterator camIt =
					camIdxByKey.find(std::make_pair(iter->first, (int)c));
			if(camIt != camIdxByKey.end())
			{
				ra::BaCam cam = graph.cams()[camIt->second];
				cam.set_pose_optimize_translation(false);
				cam.set_pose_optimize_rotation(false);
				++fixedCamCount;
			}
		}
	}
	UDEBUG("arael BA: %d fixed camera block(s) (rootId=%d)", fixedCamCount, rootId);

	// 2D / planar mode: pin each non-fixed pose's body z to its initial
	// value through the cam 0 vertex, like the Ceres back-end
	if(isSlam2d())
	{
		for(std::map<int, Transform>::const_iterator iter=poses.begin(); iter!=poses.end(); ++iter)
		{
			const bool fixNode = (rootId >= 0 && iter->first == rootId) ||
					(rootId < 0 && iter->first != -rootId);
			if(fixNode)
			{
				continue;
			}
			std::map<std::pair<int,int>, uint32_t>::const_iterator camIt =
					camIdxByKey.find(std::make_pair(iter->first, 0));
			std::map<int, std::vector<CameraModel> >::const_iterator iterModel = models.find(iter->first);
			if(camIt == camIdxByKey.end() || iterModel == models.end() || iterModel->second.empty())
			{
				continue;
			}
			const Transform & lt = iterModel->second[0].localTransform();
			const Eigen::Matrix3d Rbc = lt.getQuaterniond().toRotationMatrix();
			const Eigen::Vector3d tbc(lt.x(), lt.y(), lt.z());
			const Eigen::Vector3d d = -(Rbc.transpose() * tbc);
			ra::BaCam cam = graph.cams()[camIt->second];
			cam.set_has_planar(true);
			cam.set_pl_d(ra::vect3d::from_eigen(d));
			cam.set_pl_z(iter->second.z());
			cam.set_pl_w(std::sqrt(1e9));
		}
	}

	ra::LmConfig cfg((arael::LmPreset)preset());
	cfg.rel_precision = this->epsilon();
	if(initialLambda() > 0.0)
	{
		cfg.initial_lambda = initialLambda();
	}
	// a camera direction no observation reaches has zero curvature; damp
	// through it like Ceres' min_lm_diagonal default
	cfg.min_diagonal = 1e-6;
	// the robust loss handles outliers during the solve; Arael/BaPruneChi2
	// additionally prunes them and re-solves (converge-prune-refine)
	const bool prune = robust && baPruneChi2_ > 0.0 && !obsMeta.empty();
	cfg.max_iters = iterations();
	if(threads() >= 0)
	{
		cfg.num_threads = threads();
	}
	UDEBUG("arael BA: threads=%u (0 = every core)", cfg.num_threads);

	// Arael/BaSolve=1: the reduced camera system by preconditioned
	// conjugate gradients instead of a factorization
	ra::SparseOptions sparseOpts;
	if(baSolve() == 1)
	{
		sparseOpts.schur = arael::SchurPolicy::Force;
		sparseOpts.schur_solve = arael::SchurSolve::Iterative;
	}
	auto solveGraph = [&]() {
		return baSolve() == 1 ? graph.solve_sparse(cfg, sparseOpts)
							  : graph.solve_sparse(cfg);
	};

	ra::SolveResult result = solveGraph();
	if(result.is_err())
	{
		UWARN("arael BA: solve failed: %s", result.error().message);
		return poses;
	}
	UDEBUG("arael BA solve: %s, cost %f -> %f, %u iterations",
		arael::as_str(result->status), 0.5 * result->start_cost,
		0.5 * result->end_cost, result->iterations);
	if(ULogger::level() == ULogger::kDebug)
	{
		UDEBUG("arael BA solver report:\n%s", result->report());
	}

	// raw (unrobustified) chi^2 at the solution: above Arael/BaPruneChi2
	// (when pruning) observations are removed and the problem re-solved;
	// otherwise above Arael/BaRobustC2 they are only reported
	std::set<int> pointsToRestore;
	if(robust && !obsMeta.empty())
	{
		// current state, for the residual evaluation
		std::vector<Eigen::Matrix3d> camR(graph.cams().size());
		std::vector<Eigen::Vector3d> camT(graph.cams().size());
		for(uint32_t i = 0; i < graph.cams().size(); ++i)
		{
			ra::BaCam cam = graph.cams()[i];
			camR[i] = cam.pose_rotation().to_eigen().toRotationMatrix();
			camT[i] = cam.pose_translation().to_eigen();
		}
		const double flagChi2 = prune ? baPruneChi2_ : baRobustC2_;
		std::map<int, int> observationsPerWord;
		std::map<int, int> rejectedPerWord;
		std::vector<bool> flaggedObs(obsMeta.size(), false);
		int rejectedCount = 0;
		auto flagObs = [&](size_t i) {
			const ObsMeta & meta = obsMeta[i];
			flaggedObs[i] = true;
			if(outliers)
			{
				(*outliers)[meta.wordId].insert(meta.poseId);
			}
			++rejectedPerWord[meta.wordId];
			++rejectedCount;
			if(prune)
			{
				if(meta.stereo)
				{
					graph.stereo_obs()[meta.graphIdx].set_active(false);
				}
				else
				{
					graph.obs()[meta.graphIdx].set_active(false);
				}
			}
		};
		for(size_t i = 0; i < obsMeta.size(); ++i)
		{
			const ObsMeta & meta = obsMeta[i];
			++observationsPerWord[meta.wordId];
			ra::vect3d pv = graph.points()[meta.pointIdx].pos();
			const Eigen::Vector3d p = camR[meta.camIdx] * Eigen::Vector3d(pv.x, pv.y, pv.z)
					+ camT[meta.camIdx];
			const double ru = (meta.fx * p.x() / p.z() - meta.u) * invSigmaUv;
			const double rv = (meta.fy * p.y() / p.z() - meta.v) * invSigmaUv;
			double chi2 = ru * ru + rv * rv;
			if(meta.stereo)
			{
				const double rd = (meta.bfx / p.z() - meta.d) * invSigmaD;
				chi2 += rd * rd;
			}
			if(chi2 > flagChi2)
			{
				flagObs(i);
			}
		}
		// a word left with fewer than 2 surviving views is unreliable: a
		// free point can chase a single (possibly corrupt) observation, so
		// flag its remaining views too and hand the caller's input estimate
		// back untouched
		for(std::map<int, int>::const_iterator iter=rejectedPerWord.begin(); iter!=rejectedPerWord.end(); ++iter)
		{
			if(observationsPerWord.at(iter->first) - iter->second < 2)
			{
				pointsToRestore.insert(iter->first);
			}
		}
		if(!pointsToRestore.empty())
		{
			for(size_t i = 0; i < obsMeta.size(); ++i)
			{
				if(!flaggedObs[i] &&
					pointsToRestore.find(obsMeta[i].wordId) != pointsToRestore.end())
				{
					flagObs(i);
				}
			}
		}
		if(prune)
		{
			for(std::set<int>::const_iterator iter=pointsToRestore.begin(); iter!=pointsToRestore.end(); ++iter)
			{
				graph.points()[pointIdToIndex.at(*iter)].set_pos_optimize(false);
			}
			UDEBUG("arael BA: re-solving without %d pruned observation(s) over %d word(s), %d point(s) restored...",
				rejectedCount, (int)rejectedPerWord.size(), (int)pointsToRestore.size());
			result = solveGraph();
			if(result.is_err())
			{
				UWARN("arael BA: re-solve without the %d pruned observation(s) failed: %s",
					rejectedCount, result.error().message);
				return std::map<int, Transform>();
			}
			UDEBUG("arael BA re-solve: %s, cost %f -> %f, %u iterations",
				arael::as_str(result->status), 0.5 * result->start_cost,
				0.5 * result->end_cost, result->iterations);
		}
		else
		{
			UDEBUG("arael BA: %d observation(s) reported as outliers (chi2 > %f)",
				rejectedCount, flagChi2);
		}
	}

	// read poses back from cam 0 of each rig
	std::map<int, Transform> newPoses = poses;
	for(std::map<int, Transform>::iterator iter=newPoses.begin(); iter!=newPoses.end(); ++iter)
	{
		std::map<std::pair<int,int>, uint32_t>::const_iterator camIt =
				camIdxByKey.find(std::make_pair(iter->first, 0));
		if(camIt == camIdxByKey.end())
		{
			continue;
		}
		ra::BaCam cam = graph.cams()[camIt->second];
		ra::vect3d tv = cam.pose_translation();
		ra::quaternd q = cam.pose_rotation();
		const Transform t(tv.x, tv.y, tv.z, q.v.x, q.v.y, q.v.z, q.t);
		if(isSlam2d())
		{
			// snap z back exactly when the planar lock held; otherwise
			// project the delta onto SE(2), like the Ceres back-end
			Transform body = (models.at(iter->first)[0].localTransform() * t).inverse();
			if(std::fabs(body.z() - iter->second.z()) < 0.001f)
			{
				body.z() = iter->second.z();
				iter->second = body;
			}
			else
			{
				UWARN("Planar constraints didn't work!? original pose (%d), pose %s -> %s. Falling back to old approach.",
					iter->first,
					iter->second.prettyPrint().c_str(),
					body.prettyPrint().c_str());
				const Transform delta = iter->second.inverse() * body;
				iter->second = iter->second * delta.to3DoF();
			}
		}
		else
		{
			iter->second = (models.at(iter->first)[0].localTransform() * t).inverse();
		}
	}

	// updated 3D points; the fully-rejected ones keep the caller's estimate
	for(std::map<int, cv::Point3f>::iterator kter = points3DMap.begin(); kter!=points3DMap.end(); ++kter)
	{
		if(pointsToRestore.find(kter->first) == pointsToRestore.end())
		{
			ra::vect3d p = graph.points()[pointIdToIndex.at(kter->first)].pos();
			kter->second.x = p.x;
			kter->second.y = p.y;
			kter->second.z = p.z;
		}
	}

	return newPoses;
#else
	UERROR("RTAB-Map is not built with arael!");
	return std::map<int, Transform>();
#endif
}

} /* namespace rtabmap */
