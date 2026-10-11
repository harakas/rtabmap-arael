/*
 * Benchmarks the compiled-in graph optimizer backends on the pose graph of
 * a RTAB-Map database.
 *
 * The graph is the database's pose-pose links; initial poses are rebuilt by
 * composing the odometry links (loop closures then bridge any remaining
 * sessions), so every backend starts from the same odometry-shaped guess and
 * runs are exactly repeatable. Each solve runs in a forked child (isolated
 * timing and peak memory); repetitions are interleaved across backends.
 *
 * Backends report their final error in incompatible conventions, so results
 * are scored with one reference: the Ceres-convention cost (0.5 * sum of
 * squared sqrt-information-weighted residuals) evaluated at each backend's
 * output poses.
 *
 * --ba benchmarks global bundle adjustment instead: the pose graph is
 * pre-optimized once with one backend (the same shared input for everyone),
 * the BA problem (camera models, 3D points, mono/stereo observations) is
 * built once from the database features, and each backend's optimizeBA()
 * runs on that identical problem in a forked child. Scoring reference: the
 * robust BA objective (whitened reprojection terms under a Huber loss with
 * knee Optimizer/RobustKernelDelta, plus the folded-in pose-graph links)
 * evaluated at each backend's output poses and points over ALL observations,
 * so outlier rejection cannot improve a backend's score by dropping terms.
 */

#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Graph.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/Link.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <memory>
#include <string>
#include <vector>

using namespace rtabmap;

static void showUsage()
{
	printf("\nUsage:\n"
		"   rtabmap-optimizerBenchmark [options] \"input.db\"\n"
		"   rtabmap-optimizerBenchmark [options] \"input.g2o\"\n\n"
		"  A .g2o pose graph (as saved by rtabmap, link types in the last\n"
		"  column) is read with graph::importPoses; its vertex poses are not\n"
		"  used, the initial guess is composed from the links as for a database.\n"
		"  Bundle adjustment (--ba) needs a database.\n\n"
		"  Options:\n"
		"     --reps #        Repetitions per backend, interleaved (default 5); the median is reported.\n"
		"     --iterations #  Maximum solver iterations (default 100).\n"
		"     --epsilon #.#   Stop when the error improvement is less than this value (default 0.00001,\n"
		"                     every backend's own convention, like Optimizer/Epsilon).\n"
		"     --2d            Optimize in 2D (Reg/Force3DoF) and score with the 2D cost.\n"
		"     --ba            Benchmark global bundle adjustment (optimizeBA) instead of the\n"
		"                     pose graph: the graph is pre-optimized once, the BA problem is\n"
		"                     built once from the database features, and every backend solves\n"
		"                     that identical problem. Scored with the robust BA objective over\n"
		"                     all observations at each backend's output.\n"
		"     --strategy \"# # ...\"  Backends to run (0=TORO 1=g2o 2=GTSAM 3=Ceres 4=cvsba 5=arael);\n"
		"                     default: all compiled-in (TORO has no BA and is skipped with --ba).\n"
		"     --\"parameter name\" \"value\"  Any RTAB-Map parameter, passed to the\n"
		"                     backends (e.g. --Arael/Preset 1, --GTSAM/Optimizer 0).\n\n");
	exit(1);
}

static double wrapAngle(double a)
{
	while(a > M_PI) a -= 2.0 * M_PI;
	while(a < -M_PI) a += 2.0 * M_PI;
	return a;
}

// Ceres-convention reference cost at the given poses; 2D uses the (x, y,
// theta) sub-block of the 6x6 information, 3D the full matrix.
static double refCost(const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		bool slam2d)
{
	double cost = 0.0;
	for(const auto & kv : links)
	{
		const Link & l = kv.second;
		if(!poses.count(l.from()) || !poses.count(l.to()))
		{
			continue;
		}
		const Transform & A = poses.at(l.from());
		const Transform & B = poses.at(l.to());
		if(slam2d)
		{
			Eigen::Matrix3d info;
			const int idx[3] = {0, 1, 5};
			for(int i = 0; i < 3; ++i)
				for(int j = 0; j < 3; ++j)
					info(i, j) = l.infMatrix().at<double>(idx[i], idx[j]);
			const Eigen::Matrix3d S = info.llt().matrixU();
			const double ca = std::cos(A.theta()), sa = std::sin(A.theta());
			const double dx = B.x() - A.x(), dy = B.y() - A.y();
			Eigen::Vector3d r;
			r(0) =  ca * dx + sa * dy - l.transform().x();
			r(1) = -sa * dx + ca * dy - l.transform().y();
			r(2) = wrapAngle((B.theta() - A.theta()) - l.transform().theta());
			r = S * r;
			cost += 0.5 * r.squaredNorm();
		}
		else
		{
			Eigen::Matrix<double, 6, 6> info;
			for(int i = 0; i < 6; ++i)
				for(int j = 0; j < 6; ++j)
					info(i, j) = l.infMatrix().at<double>(i, j);
			const Eigen::Matrix<double, 6, 6> S = info.llt().matrixU();
			const Eigen::Quaterniond qa = A.getQuaterniond();
			const Eigen::Quaterniond qb = B.getQuaterniond();
			const Eigen::Quaterniond qm = l.transform().getQuaterniond();
			const Eigen::Vector3d pa(A.x(), A.y(), A.z());
			const Eigen::Vector3d pb(B.x(), B.y(), B.z());
			const Eigen::Vector3d tm(l.transform().x(), l.transform().y(), l.transform().z());
			Eigen::Matrix<double, 6, 1> r;
			r.head<3>() = qa.conjugate() * (pb - pa) - tm;
			const Eigen::Quaterniond dq = qm * (qa.conjugate() * qb).conjugate();
			r.tail<3>() = 2.0 * dq.vec();
			r = S * r;
			cost += 0.5 * r.squaredNorm();
		}
	}
	return cost;
}

static const char * strategyName(Optimizer::Type t)
{
	switch(t)
	{
	case Optimizer::kTypeTORO:  return "TORO";
	case Optimizer::kTypeG2O:   return "g2o";
	case Optimizer::kTypeGTSAM: return "GTSAM";
	case Optimizer::kTypeCeres: return "Ceres";
	case Optimizer::kTypeCVSBA: return "cvsba";
	case Optimizer::kTypeArael: return "arael";
	default:                    return "?";
	}
}

struct BaScore
{
	double cost;   // robust BA objective (see refBaCost)
	double pxRmse; // raw reprojection RMSE over the pixel rows, in pixels
	double pxP50;  // percentiles of the per-observation pixel error,
	double pxP90;  // over ALL observations
	double pxP95;
	double pxP96;
	double pxP97;
	double pxP98;
	double pxP99;
};

// The BA objective the backends minimize (the Ceres backend's residuals):
// whitened mono/stereo reprojection terms under a Huber loss with knee
// Optimizer/RobustKernelDelta, plus the folded-in pose-graph links between
// the cam-0 vertices. Evaluated at the given poses and points over every
// observation, whether or not a backend rejected it.
static BaScore refBaCost(const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::map<int, std::vector<CameraModel> > & models,
		const std::map<int, cv::Point3f> & points3DMap,
		const std::map<int, std::map<int, FeatureBA> > & wordReferences,
		double pixelVariance,
		double disparityVariance,
		double robustKernelDelta,
		double baseline,
		bool covarianceIgnored,
		const BAOutliers * exclude = 0)   // score without these observations
{
	const double invSigmaUv = 1.0 / std::sqrt(pixelVariance);
	const double invSigmaD = 1.0 / std::sqrt(disparityVariance);
	const double k2 = robustKernelDelta > 0.0 ?
			robustKernelDelta * robustKernelDelta : 1e30;

	BaScore score;
	memset(&score, 0, sizeof(score));
	double pxSum = 0.0;
	std::vector<double> pxErr;
	for(const auto & word : wordReferences)
	{
		auto pointIt = points3DMap.find(word.first);
		if(pointIt == points3DMap.end())
		{
			continue;
		}
		const Eigen::Vector3d P(pointIt->second.x, pointIt->second.y, pointIt->second.z);
		for(const auto & ref : word.second)
		{
			if(exclude)
			{
				auto w = exclude->find(word.first);
				if(w != exclude->end() && w->second.count(ref.first))
				{
					continue;
				}
			}
			const FeatureBA & f = ref.second;
			auto poseIt = poses.find(ref.first);
			auto modelIt = models.find(ref.first);
			if(poseIt == poses.end() || modelIt == models.end() ||
				f.cameraIndex < 0 || f.cameraIndex >= (int)modelIt->second.size())
			{
				continue;
			}
			const CameraModel & m = modelIt->second[f.cameraIndex];
			const Transform w2c = (poseIt->second * m.localTransform()).inverse();
			const Eigen::Vector3d p =
				w2c.getQuaterniond() * P + Eigen::Vector3d(w2c.x(), w2c.y(), w2c.z());
			const double du = m.fx() * (p.x() / p.z()) - ((double)f.kpt.pt.x - m.cx());
			const double dv = m.fy() * (p.y() / p.z()) - ((double)f.kpt.pt.y - m.cy());
			double s = (du * du + dv * dv) * invSigmaUv * invSigmaUv;
			const double bl = m.Tx() < 0.0 ? -m.Tx() / m.fx() : baseline;
			if(bl > 0.0 && std::isfinite(f.depth) && f.depth > 0.0)
			{
				const double bfx = bl * m.fx();
				const double dd = bfx / p.z() - bfx / (double)f.depth;
				s += dd * dd * invSigmaD * invSigmaD;
			}
			score.cost += 0.5 * (s <= k2 ? s : 2.0 * std::sqrt(s * k2) - k2);
			pxSum += du * du + dv * dv;
			pxErr.push_back(std::sqrt(du * du + dv * dv));
		}
	}
	if(!pxErr.empty())
	{
		score.pxRmse = std::sqrt(pxSum / (double)pxErr.size());
		std::sort(pxErr.begin(), pxErr.end());
		auto pct = [&](double q) {
			return pxErr[(size_t)((double)(pxErr.size() - 1) * q)];
		};
		score.pxP50 = pxErr[pxErr.size() / 2];
		score.pxP90 = pct(0.90);
		score.pxP95 = pct(0.95);
		score.pxP96 = pct(0.96);
		score.pxP97 = pct(0.97);
		score.pxP98 = pct(0.98);
		score.pxP99 = pct(0.99);
	}

	for(const auto & kv : links)
	{
		const Link & l = kv.second;
		if(l.from() <= 0 || l.to() <= 0 || l.from() == l.to() ||
			!poses.count(l.from()) || !poses.count(l.to()) ||
			!models.count(l.from()) || !models.count(l.to()))
		{
			continue;
		}
		const Transform & lt0a = models.at(l.from())[0].localTransform();
		const Transform & lt0b = models.at(l.to())[0].localTransform();
		const Transform camLink = lt0a.inverse() * l.transform() * lt0b;
		Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Identity();
		if(!covarianceIgnored)
		{
			memcpy(info.data(), l.infMatrix().data, l.infMatrix().total() * sizeof(double));
		}
		const Eigen::Matrix<double, 6, 6> S = info.llt().matrixU();
		const Transform wa = (poses.at(l.from()) * lt0a).inverse();
		const Transform wb = (poses.at(l.to()) * lt0b).inverse();
		const Eigen::Matrix3d Ra = wa.getQuaterniond().toRotationMatrix();
		const Eigen::Matrix3d Rb = wb.getQuaterniond().toRotationMatrix();
		const Eigen::Matrix3d Rm = camLink.getQuaterniond().toRotationMatrix();
		const Eigen::Matrix3d Rab = Ra * Rb.transpose();
		const Eigen::Matrix3d Rerr = Rm.transpose() * Rab;
		const double denom = std::sqrt(std::max(Rerr.trace() + 1.0, 1e-12));
		const Eigen::Vector3d rrot(
			(Rerr(2, 1) - Rerr(1, 2)) / denom,
			(Rerr(0, 2) - Rerr(2, 0)) / denom,
			(Rerr(1, 0) - Rerr(0, 1)) / denom);
		const Eigen::Vector3d terr =
			Eigen::Vector3d(wa.x(), wa.y(), wa.z()) -
			Rab * Eigen::Vector3d(wb.x(), wb.y(), wb.z()) -
			Eigen::Vector3d(camLink.x(), camLink.y(), camLink.z());
		const Eigen::Vector3d wt =
			S.topLeftCorner<3, 3>() * terr + S.topRightCorner<3, 3>() * rrot;
		const Eigen::Vector3d wr = S.bottomRightCorner<3, 3>() * rrot;
		score.cost += 0.5 * (wt.squaredNorm() + wr.squaredNorm());
	}
	return score;
}

struct RunResult
{
	double ms;
	double cost;
	double pxRmse;   // BA mode only
	double pxP50;    // BA mode only: per-observation pixel error percentiles
	double pxP90;
	double pxP95;
	double pxP96;
	double pxP97;
	double pxP98;
	double pxP99;
	double keptCost; // BA mode only: cost excluding the backend's own rejections
	int iterations;
	int rejected;    // BA mode only: observations dropped by the backend
	long maxRssMb;
	bool ok;
};

// One solve in a forked child: clean timing, per-run peak memory, and a
// crashing backend cannot take the benchmark down.
static RunResult runOnce(Optimizer::Type type,
		const ParametersMap & parameters,
		int iterations,
		double epsilon,
		bool slam2d,
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links)
{
	RunResult res;
	memset(&res, 0, sizeof(res));

	int fd[2];
	if(pipe(fd) != 0)
	{
		return res;
	}
	pid_t pid = fork();
	if(pid == 0)
	{
		close(fd[0]);
		std::unique_ptr<Optimizer> opt(Optimizer::create(type, parameters));
		opt->setSlam2d(slam2d);
		opt->setIterations(iterations);
		opt->setEpsilon(epsilon);
		cv::Mat cov;
		double err = -1.0;
		int iters = 0;
		UTimer timer;
		std::map<int, Transform> out = opt->optimize(rootId, poses, links, cov, 0, &err, &iters);
		double ms = timer.ticks() * 1000.0;
		if(!out.empty())
		{
			struct rusage usage;
			getrusage(RUSAGE_SELF, &usage);
			RunResult r;
			r.ms = ms;
			r.cost = refCost(out, links, slam2d);
			r.iterations = iters;
			r.maxRssMb = usage.ru_maxrss / 1024;
			r.ok = true;
			ssize_t written = write(fd[1], &r, sizeof(r));
			(void)written;
		}
		close(fd[1]);
		_exit(0);
	}
	close(fd[1]);
	if(pid > 0)
	{
		if(read(fd[0], &res, sizeof(res)) != (ssize_t)sizeof(res))
		{
			res.ok = false;
		}
		int status = 0;
		waitpid(pid, &status, 0);
		if(status != 0)
		{
			res.ok = false;
		}
	}
	close(fd[0]);
	return res;
}

// One BA solve in a forked child; the problem is inherited from the parent
// so every backend and repetition sees the identical bit-for-bit input.
static RunResult runOnceBA(Optimizer::Type type,
		const ParametersMap & parameters,
		int iterations,
		double epsilon,
		bool slam2d,
		int rootId,
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::map<int, std::vector<CameraModel> > & models,
		const std::map<int, cv::Point3f> & points3DMap,
		const std::map<int, std::map<int, FeatureBA> > & wordReferences,
		double pixelVariance,
		double disparityVariance,
		double robustKernelDelta,
		double baseline,
		bool covarianceIgnored)
{
	RunResult res;
	memset(&res, 0, sizeof(res));

	int fd[2];
	if(pipe(fd) != 0)
	{
		return res;
	}
	pid_t pid = fork();
	if(pid == 0)
	{
		close(fd[0]);
		std::unique_ptr<Optimizer> opt(Optimizer::create(type, parameters));
		opt->setSlam2d(slam2d);
		opt->setIterations(iterations);
		opt->setEpsilon(epsilon);
		std::map<int, cv::Point3f> points = points3DMap; // refined in place
		BAOutliers outliers;
		UTimer timer;
		std::map<int, Transform> out =
			opt->optimizeBA(rootId, poses, links, models, points, wordReferences, &outliers);
		double ms = timer.ticks() * 1000.0;
		if(!out.empty())
		{
			const BaScore score = refBaCost(out, links, models, points, wordReferences,
				pixelVariance, disparityVariance, robustKernelDelta, baseline,
				covarianceIgnored);
			struct rusage usage;
			getrusage(RUSAGE_SELF, &usage);
			RunResult r;
			memset(&r, 0, sizeof(r));
			r.ms = ms;
			r.cost = score.cost;
			r.pxRmse = score.pxRmse;
			r.pxP50 = score.pxP50;
			r.pxP90 = score.pxP90;
			r.pxP95 = score.pxP95;
			r.pxP96 = score.pxP96;
			r.pxP97 = score.pxP97;
			r.pxP98 = score.pxP98;
			r.pxP99 = score.pxP99;
			for(const auto & w : outliers)
			{
				r.rejected += (int)w.second.size();
			}
			r.keptCost = refBaCost(out, links, models, points, wordReferences,
				pixelVariance, disparityVariance, robustKernelDelta, baseline,
				covarianceIgnored, &outliers).cost;
			r.maxRssMb = usage.ru_maxrss / 1024;
			r.ok = true;
			ssize_t written = write(fd[1], &r, sizeof(r));
			(void)written;
		}
		close(fd[1]);
		_exit(0);
	}
	close(fd[1]);
	if(pid > 0)
	{
		if(read(fd[0], &res, sizeof(res)) != (ssize_t)sizeof(res))
		{
			res.ok = false;
		}
		int status = 0;
		waitpid(pid, &status, 0);
		if(status != 0)
		{
			res.ok = false;
		}
	}
	close(fd[0]);
	return res;
}

// What a solve child starts from: the resident set a fresh forked child
// reports, i.e. the inherited problem data.
static long forkedBaselineRssMb()
{
	int fd[2];
	if(pipe(fd) != 0)
	{
		return -1;
	}
	pid_t pid = fork();
	if(pid == 0)
	{
		close(fd[0]);
		struct rusage usage;
		getrusage(RUSAGE_SELF, &usage);
		long mb = usage.ru_maxrss / 1024;
		ssize_t written = write(fd[1], &mb, sizeof(mb));
		(void)written;
		close(fd[1]);
		_exit(0);
	}
	close(fd[1]);
	long mb = -1;
	if(pid > 0)
	{
		if(read(fd[0], &mb, sizeof(mb)) != (ssize_t)sizeof(mb))
		{
			mb = -1;
		}
		waitpid(pid, 0, 0);
	}
	close(fd[0]);
	return mb;
}

static double median(std::vector<double> v)
{
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}

static bool writeAll(int fd, const void * data, size_t n)
{
	const char * p = (const char *)data;
	while(n > 0)
	{
		ssize_t w = write(fd, p, n);
		if(w <= 0)
		{
			return false;
		}
		p += w;
		n -= (size_t)w;
	}
	return true;
}

static bool readAll(int fd, void * data, size_t n)
{
	char * p = (char *)data;
	while(n > 0)
	{
		ssize_t r = read(fd, p, n);
		if(r <= 0)
		{
			return false;
		}
		p += r;
		n -= (size_t)r;
	}
	return true;
}

static bool writeTransform(int fd, const Transform & t)
{
	return writeAll(fd, t.data(), 12 * sizeof(float));
}

static bool readTransform(int fd, Transform & t)
{
	float d[12];
	if(!readAll(fd, d, sizeof(d)))
	{
		return false;
	}
	t = Transform(d[0], d[1], d[2], d[3], d[4], d[5],
			d[6], d[7], d[8], d[9], d[10], d[11]);
	return true;
}

// Runs in a forked child: pre-optimizes the pose graph, loads the
// signatures, builds the camera models and BA correspondences, and streams
// the problem over the pipe -- only the fields optimizeBA() consumes
// (feature descriptors are construction-time data and are dropped).
// Returns the child's exit code; failures are printed here.
static int buildBaProblem(int fd,
		const std::string & dbPath,
		const ParametersMap & parameters,
		Optimizer::Type preType,
		int iterations,
		double epsilon,
		bool slam2d,
		int rootId,
		const std::map<int, Transform> & posesGuess,
		const std::multimap<int, Link> & links)
{
	std::unique_ptr<Optimizer> pre(Optimizer::create(preType, parameters));
	pre->setSlam2d(slam2d);
	pre->setIterations(iterations);
	pre->setEpsilon(epsilon);
	cv::Mat cov;
	std::map<int, Transform> prePoses = pre->optimize(rootId, posesGuess, links, cov);
	if(prePoses.empty())
	{
		printf("Pose-graph pre-optimization (%s) failed.\n", strategyName(preType));
		return 1;
	}

	std::map<int, Signature> nodes;
	{
		DBDriver * driver = DBDriver::create();
		if(!driver->openConnection(dbPath))
		{
			printf("Cannot re-open database \"%s\"\n", dbPath.c_str());
			return 1;
		}
		std::list<int> ids;
		for(const auto & kv : prePoses)
		{
			ids.push_back(kv.first);
		}
		std::list<Signature *> sigs;
		driver->loadSignatures(ids, sigs);
		driver->closeConnection();
		delete driver;
		for(Signature * s : sigs)
		{
			nodes.insert(std::make_pair(s->id(), *s));
			delete s;
		}
	}

	// camera models per pose (mono, or stereo left with Tx = -baseline*fx):
	// the same construction as the signature-based optimizeBA() overload
	std::map<int, std::vector<CameraModel> > models;
	for(const auto & kv : prePoses)
	{
		std::vector<CameraModel> m;
		auto s = nodes.find(kv.first);
		if(s != nodes.end())
		{
			const SensorData & sd = s->second.sensorData();
			if(sd.cameraModels().size() >= 1 && sd.cameraModels().at(0).isValidForProjection())
			{
				m = sd.cameraModels();
			}
			else if(!sd.stereoCameraModels().empty() &&
					sd.stereoCameraModels()[0].isValidForProjection())
			{
				for(size_t i = 0; i < sd.stereoCameraModels().size(); ++i)
				{
					const CameraModel & left = sd.stereoCameraModels()[i].left();
					m.push_back(CameraModel(left.fx(), left.fy(), left.cx(), left.cy(),
						left.localTransform(),
						-sd.stereoCameraModels()[i].baseline() * left.fx(),
						left.imageSize()));
				}
			}
		}
		if(m.empty())
		{
			printf("Node %d has no valid calibration; database is not BA-capable.\n", kv.first);
			return 1;
		}
		models.insert(std::make_pair(kv.first, m));
	}

	// the BA problem, built once (feature rematch, like the offline tools)
	std::map<int, cv::Point3f> points3DMap;
	std::map<int, std::map<int, FeatureBA> > wordReferences;
	pre->computeBACorrespondences(prePoses, links, nodes, points3DMap, wordReferences,
		true, false, parameters);
	if(points3DMap.empty() || wordReferences.empty())
	{
		printf("No BA correspondences (no features stored in the database?).\n");
		return 1;
	}

	bool ok = true;
	uint64_t n = prePoses.size();
	ok = ok && writeAll(fd, &n, sizeof(n));
	for(const auto & kv : prePoses)
	{
		const int32_t id = kv.first;
		ok = ok && writeAll(fd, &id, sizeof(id)) && writeTransform(fd, kv.second);
	}
	n = models.size();
	ok = ok && writeAll(fd, &n, sizeof(n));
	for(const auto & kv : models)
	{
		const int32_t id = kv.first;
		const uint32_t cnt = kv.second.size();
		ok = ok && writeAll(fd, &id, sizeof(id)) && writeAll(fd, &cnt, sizeof(cnt));
		for(const CameraModel & m : kv.second)
		{
			const double intr[5] = {m.fx(), m.fy(), m.cx(), m.cy(), m.Tx()};
			const int32_t size[2] = {m.imageSize().width, m.imageSize().height};
			ok = ok && writeAll(fd, intr, sizeof(intr)) &&
				writeTransform(fd, m.localTransform()) &&
				writeAll(fd, size, sizeof(size));
		}
	}
	n = points3DMap.size();
	ok = ok && writeAll(fd, &n, sizeof(n));
	for(const auto & kv : points3DMap)
	{
		const int32_t id = kv.first;
		const float p[3] = {kv.second.x, kv.second.y, kv.second.z};
		ok = ok && writeAll(fd, &id, sizeof(id)) && writeAll(fd, p, sizeof(p));
	}
	n = wordReferences.size();
	ok = ok && writeAll(fd, &n, sizeof(n));
	for(const auto & word : wordReferences)
	{
		const int32_t id = word.first;
		const uint32_t cnt = word.second.size();
		ok = ok && writeAll(fd, &id, sizeof(id)) && writeAll(fd, &cnt, sizeof(cnt));
		for(const auto & ref : word.second)
		{
			const int32_t frameId = ref.first;
			const FeatureBA & f = ref.second;
			const float kf[5] = {f.kpt.pt.x, f.kpt.pt.y, f.kpt.size,
				f.kpt.angle, f.kpt.response};
			const int32_t ki[2] = {f.kpt.octave, f.kpt.class_id};
			ok = ok && writeAll(fd, &frameId, sizeof(frameId)) &&
				writeAll(fd, kf, sizeof(kf)) && writeAll(fd, ki, sizeof(ki)) &&
				writeAll(fd, &f.depth, sizeof(f.depth)) &&
				writeAll(fd, &f.cameraIndex, sizeof(f.cameraIndex));
		}
	}
	if(!ok)
	{
		printf("Writing the BA problem to the pipe failed.\n");
		return 1;
	}
	return 0;
}

static bool readBaProblem(int fd,
		std::map<int, Transform> & prePoses,
		std::map<int, std::vector<CameraModel> > & models,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences)
{
	uint64_t n = 0;
	if(!readAll(fd, &n, sizeof(n)))
	{
		return false;
	}
	for(uint64_t i = 0; i < n; ++i)
	{
		int32_t id;
		Transform t;
		if(!readAll(fd, &id, sizeof(id)) || !readTransform(fd, t))
		{
			return false;
		}
		prePoses.emplace(id, t);
	}
	if(!readAll(fd, &n, sizeof(n)))
	{
		return false;
	}
	for(uint64_t i = 0; i < n; ++i)
	{
		int32_t id;
		uint32_t cnt;
		if(!readAll(fd, &id, sizeof(id)) || !readAll(fd, &cnt, sizeof(cnt)))
		{
			return false;
		}
		std::vector<CameraModel> m;
		for(uint32_t c = 0; c < cnt; ++c)
		{
			double intr[5];
			Transform lt;
			int32_t size[2];
			if(!readAll(fd, intr, sizeof(intr)) || !readTransform(fd, lt) ||
				!readAll(fd, size, sizeof(size)))
			{
				return false;
			}
			m.push_back(CameraModel(intr[0], intr[1], intr[2], intr[3], lt, intr[4],
				cv::Size(size[0], size[1])));
		}
		models.emplace(id, std::move(m));
	}
	if(!readAll(fd, &n, sizeof(n)))
	{
		return false;
	}
	for(uint64_t i = 0; i < n; ++i)
	{
		int32_t id;
		float p[3];
		if(!readAll(fd, &id, sizeof(id)) || !readAll(fd, p, sizeof(p)))
		{
			return false;
		}
		points3DMap.emplace(id, cv::Point3f(p[0], p[1], p[2]));
	}
	if(!readAll(fd, &n, sizeof(n)))
	{
		return false;
	}
	for(uint64_t i = 0; i < n; ++i)
	{
		int32_t id;
		uint32_t cnt;
		if(!readAll(fd, &id, sizeof(id)) || !readAll(fd, &cnt, sizeof(cnt)))
		{
			return false;
		}
		std::map<int, FeatureBA> & refs = wordReferences[id];
		for(uint32_t c = 0; c < cnt; ++c)
		{
			int32_t frameId;
			float kf[5];
			int32_t ki[2];
			float depth;
			int32_t cameraIndex;
			if(!readAll(fd, &frameId, sizeof(frameId)) ||
				!readAll(fd, kf, sizeof(kf)) || !readAll(fd, ki, sizeof(ki)) ||
				!readAll(fd, &depth, sizeof(depth)) ||
				!readAll(fd, &cameraIndex, sizeof(cameraIndex)))
			{
				return false;
			}
			refs.emplace(frameId, FeatureBA(
				cv::KeyPoint(kf[0], kf[1], kf[2], kf[3], kf[4], ki[0], ki[1]),
				depth, cv::Mat(), cameraIndex));
		}
	}
	return true;
}

static int runBaBenchmark(const std::string & dbPath,
		const ParametersMap & parameters,
		std::vector<Optimizer::Type> strategies,
		int reps,
		int iterations,
		double epsilon,
		bool slam2d,
		int rootId,
		const std::map<int, Transform> & posesGuess,
		const std::multimap<int, Link> & links)
{
	{
		std::vector<Optimizer::Type> keep;
		for(Optimizer::Type t : strategies)
		{
			if(t == Optimizer::kTypeTORO)
			{
				printf("TORO has no bundle adjustment, skipped.\n");
			}
			else
			{
				keep.push_back(t);
			}
		}
		strategies = keep;
	}
	if(strategies.empty())
	{
		printf("No BA-capable backend to run.\n");
		return 1;
	}

	// shared input: the pose graph pre-optimized once, with one fixed backend
	// whatever the benchmarked set is, so rows stay comparable
	Optimizer::Type preType = strategies.front();
	for(Optimizer::Type t : {Optimizer::kTypeG2O, Optimizer::kTypeGTSAM,
			Optimizer::kTypeCeres, Optimizer::kTypeArael})
	{
		if(Optimizer::isAvailable(t))
		{
			preType = t;
			break;
		}
	}
	// Construction (pre-optimization, signature loading, feature rematch)
	// allocates and frees hundreds of MB; done in this process it would
	// leave the heap fragmented with the freed pages still resident, and
	// every solve child would inherit that. A child builds the problem and
	// pipes back only the data the benchmark needs, so this heap -- what
	// the solve children fork from -- holds just the problem.
	std::map<int, Transform> prePoses;
	std::map<int, std::vector<CameraModel> > models;
	std::map<int, cv::Point3f> points3DMap;
	std::map<int, std::map<int, FeatureBA> > wordReferences;
	{
		int fd[2];
		if(pipe(fd) != 0)
		{
			printf("pipe() failed\n");
			return 1;
		}
		pid_t pid = fork();
		if(pid == 0)
		{
			close(fd[0]);
			int code = buildBaProblem(fd[1], dbPath, parameters, preType, iterations,
				epsilon, slam2d, rootId, posesGuess, links);
			close(fd[1]);
			_exit(code);
		}
		close(fd[1]);
		const bool ok = pid > 0 &&
			readBaProblem(fd[0], prePoses, models, points3DMap, wordReferences);
		close(fd[0]);
		int status = -1;
		if(pid > 0)
		{
			waitpid(pid, &status, 0);
		}
		if(!ok || status != 0)
		{
			// the construction child printed the reason
			return 1;
		}
	}

	double pixelVariance = Parameters::defaultOptimizerPixelVariance();
	double disparityVariance = Parameters::defaultOptimizerDisparityVariance();
	double robustKernelDelta = Parameters::defaultOptimizerRobustKernelDelta();
	double baseline = Parameters::defaultOptimizerBaseline();
	bool covarianceIgnored = Parameters::defaultOptimizerVarianceIgnored();
	Parameters::parse(parameters, Parameters::kOptimizerPixelVariance(), pixelVariance);
	Parameters::parse(parameters, Parameters::kOptimizerDisparityVariance(), disparityVariance);
	Parameters::parse(parameters, Parameters::kOptimizerRobustKernelDelta(), robustKernelDelta);
	Parameters::parse(parameters, Parameters::kOptimizerBaseline(), baseline);
	Parameters::parse(parameters, Parameters::kOptimizerVarianceIgnored(), covarianceIgnored);

	// mono/stereo split with the backends' stereo rule
	size_t monoObs = 0, stereoObs = 0;
	for(const auto & word : wordReferences)
	{
		for(const auto & ref : word.second)
		{
			const auto modelIt = models.find(ref.first);
			if(modelIt == models.end() || ref.second.cameraIndex < 0 ||
				ref.second.cameraIndex >= (int)modelIt->second.size())
			{
				continue;
			}
			const CameraModel & m = modelIt->second[ref.second.cameraIndex];
			const double bl = m.Tx() < 0.0 ? -m.Tx() / m.fx() : baseline;
			if(bl > 0.0 && std::isfinite(ref.second.depth) && ref.second.depth > 0.0)
			{
				++stereoObs;
			}
			else
			{
				++monoObs;
			}
		}
	}

	const int baRoot = prePoses.begin()->first;
	const BaScore initial = refBaCost(prePoses, links, models, points3DMap, wordReferences,
		pixelVariance, disparityVariance, robustKernelDelta, baseline, covarianceIgnored);
	printf("ba_pre_opt=%s poses=%zu points=%zu mono_obs=%zu stereo_obs=%zu "
		"robust_kernel_delta=%g initial_cost=%.4f initial_px_p50/p90/p99=%.3f/%.3f/%.3f "
		"problem_rss_mb=%ld\n",
		strategyName(preType), prePoses.size(), points3DMap.size(), monoObs, stereoObs,
		robustKernelDelta, initial.cost, initial.pxP50, initial.pxP90, initial.pxP99,
		forkedBaselineRssMb());

	// one row per backend; arael additionally as its conjugate-gradient
	// solver (Arael/BaSolve), unless the caller pinned that parameter
	struct BaRow
	{
		Optimizer::Type type;
		std::string label;
		ParametersMap params;
	};
	std::vector<BaRow> rows;
	for(Optimizer::Type t : strategies)
	{
		if(t == Optimizer::kTypeArael &&
			parameters.find(Parameters::kAraelBaSolve()) == parameters.end())
		{
			ParametersMap cg = parameters;
			cg[Parameters::kAraelBaSolve()] = "1";
			rows.push_back(BaRow{t, "arael", parameters});
			rows.push_back(BaRow{t, "arael-cg", cg});
		}
		else
		{
			rows.push_back(BaRow{t, strategyName(t), parameters});
		}
	}

	std::vector<std::vector<RunResult> > results(rows.size());
	for(int rep = 0; rep < reps; ++rep)
	{
		for(size_t i = 0; i < rows.size(); ++i)
		{
			results[i].push_back(runOnceBA(rows[i].type, rows[i].params, iterations,
				epsilon, slam2d, baRoot, prePoses, links, models, points3DMap,
				wordReferences, pixelVariance, disparityVariance, robustKernelDelta,
				baseline, covarianceIgnored));
		}
	}

	for(size_t i = 0; i < rows.size(); ++i)
	{
		std::vector<double> ms, cost, px, rejected, rss;
		for(const RunResult & r : results[i])
		{
			if(r.ok)
			{
				ms.push_back(r.ms);
				cost.push_back(r.cost);
				px.push_back(r.pxRmse);
				rejected.push_back(r.rejected);
				rss.push_back(r.maxRssMb);
			}
		}
		if(ms.empty())
		{
			printf("%-8s FAILED\n", rows[i].label.c_str());
			continue;
		}
		std::vector<double> kept, p50, p90, p95, p96, p97, p98, p99;
		for(const RunResult & r : results[i])
		{
			if(r.ok)
			{
				kept.push_back(r.keptCost);
				p50.push_back(r.pxP50);
				p90.push_back(r.pxP90);
				p95.push_back(r.pxP95);
				p96.push_back(r.pxP96);
				p97.push_back(r.pxP97);
				p98.push_back(r.pxP98);
				p99.push_back(r.pxP99);
			}
		}
		printf("%-8s solve_ms=%-10.2f cost_all=%-14.4f cost_kept=%-14.4f "
			"px_p50=%-7.3f px_p90=%-7.3f px_p95=%-7.3f px_p96=%-7.3f "
			"px_p97=%-7.3f px_p98=%-7.3f px_p99=%-8.3f rejected=%-6.0f "
			"maxrss_mb=%.0f (%zu/%zu runs ok)\n",
			rows[i].label.c_str(), median(ms), median(cost), median(kept),
			median(p50), median(p90), median(p95), median(p96),
			median(p97), median(p98), median(p99),
			median(rejected), median(rss), ms.size(), results[i].size());
	}
	return 0;
}

int main(int argc, char * argv[])
{
	ULogger::setType(ULogger::kTypeConsole);
	ULogger::setLevel(ULogger::kError);

	int reps = 5;
	int iterations = 100;
	double epsilon = 0.00001;
	bool slam2d = false;
	bool ba = false;
	std::vector<Optimizer::Type> strategies;
	std::string dbPath;

	// generic RTAB-Map parameter overrides (--Group/Name value), passed to
	// every backend
	ParametersMap parameters = Parameters::parseArguments(argc, argv, true);

	for(int i = 1; i < argc; ++i)
	{
		if(strcmp(argv[i], "--help") == 0)
		{
			showUsage();
		}
		else if(argv[i][0] == '-' && strchr(argv[i], '/') != NULL)
		{
			// a parameter override, already consumed by parseArguments
			++i;
		}
		else if(strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
		{
			reps = atoi(argv[++i]);
		}
		else if(strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
		{
			iterations = atoi(argv[++i]);
		}
		else if(strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc)
		{
			epsilon = atof(argv[++i]);
		}
		else if(strcmp(argv[i], "--2d") == 0)
		{
			slam2d = true;
		}
		else if(strcmp(argv[i], "--ba") == 0)
		{
			ba = true;
		}
		else if(strcmp(argv[i], "--strategy") == 0 && i + 1 < argc)
		{
			for(const char * p = argv[++i]; *p; ++p)
			{
				if(*p >= '0' && *p <= '9')
				{
					strategies.push_back((Optimizer::Type)(*p - '0'));
				}
			}
		}
		else if(argv[i][0] != '-')
		{
			dbPath = argv[i];
		}
		else
		{
			printf("Unknown option: %s\n", argv[i]);
			showUsage();
		}
	}
	if(dbPath.empty() || reps < 1)
	{
		showUsage();
	}
	if(strategies.empty())
	{
		const std::vector<Optimizer::Type> candidates = ba ?
			std::vector<Optimizer::Type>{Optimizer::kTypeG2O, Optimizer::kTypeGTSAM,
				Optimizer::kTypeCeres, Optimizer::kTypeCVSBA, Optimizer::kTypeArael} :
			std::vector<Optimizer::Type>{Optimizer::kTypeTORO, Optimizer::kTypeG2O,
				Optimizer::kTypeGTSAM, Optimizer::kTypeCeres, Optimizer::kTypeArael};
		for(Optimizer::Type t : candidates)
		{
			if(Optimizer::isAvailable(t))
			{
				strategies.push_back(t);
			}
		}
	}
	else
	{
		// the factory silently falls back to another backend; a mislabeled
		// row is worse than no row
		std::vector<Optimizer::Type> available;
		for(Optimizer::Type t : strategies)
		{
			if(Optimizer::isAvailable(t))
			{
				available.push_back(t);
			}
			else
			{
				printf("%s is not compiled in, skipped.\n", strategyName(t));
			}
		}
		strategies = available;
	}
	if(strategies.empty())
	{
		printf("No requested backend is compiled in.\n");
		return 1;
	}

	std::multimap<int, Link> allLinks;
	const bool g2oInput = dbPath.size() > 4 && dbPath.compare(dbPath.size() - 4, 4, ".g2o") == 0;
	if(g2oInput)
	{
		if(ba)
		{
			printf("Bundle adjustment (--ba) needs a database, not a .g2o graph.\n");
			return 1;
		}
		std::map<int, Transform> filePoses;
		if(!graph::importPoses(dbPath, 4, filePoses, &allLinks))
		{
			printf("Cannot read g2o graph \"%s\"\n", dbPath.c_str());
			return 1;
		}
	}
	else
	{
		DBDriver * driver = DBDriver::create();
		if(!driver->openConnection(dbPath))
		{
			printf("Cannot open database \"%s\"\n", dbPath.c_str());
			return 1;
		}
		driver->getAllLinks(allLinks, true, false);
		driver->closeConnection();
		delete driver;
	}

	// pose-pose links only, one per pair: the database stores each link in
	// both directions, and a doubled set would double every cost and the
	// per-iteration work compared to what the in-product solves see
	std::multimap<int, Link> links;
	std::set<std::pair<int, int> > seenPairs;
	for(const auto & kv : allLinks)
	{
		const Link & l = kv.second;
		if(l.from() != l.to() && l.from() > 0 && l.to() > 0 &&
			seenPairs.insert(std::make_pair(
				std::min(l.from(), l.to()), std::max(l.from(), l.to()))).second)
		{
			links.insert(kv);
		}
	}
	if(links.empty())
	{
		printf("The graph has no pose-pose links.\n");
		return 1;
	}

	// odometry-shaped initial guess: compose neighbor links first, then let
	// the remaining links (closures) bridge disconnected sessions
	std::map<int, Transform> poses;
	int rootId = links.begin()->second.from();
	poses.emplace(rootId, Transform::getIdentity());
	for(int pass = 0; pass < 2; ++pass)
	{
		bool grew = true;
		while(grew)
		{
			grew = false;
			for(const auto & kv : links)
			{
				const Link & l = kv.second;
				if(pass == 0 &&
					l.type() != Link::kNeighbor && l.type() != Link::kNeighborMerged)
				{
					continue;
				}
				if(poses.count(l.from()) && !poses.count(l.to()))
				{
					poses.emplace(l.to(), poses.at(l.from()) * l.transform());
					grew = true;
				}
				else if(poses.count(l.to()) && !poses.count(l.from()))
				{
					poses.emplace(l.from(), poses.at(l.to()) * l.transform().inverse());
					grew = true;
				}
			}
		}
	}
	std::multimap<int, Link> usable;
	for(const auto & kv : links)
	{
		if(poses.count(kv.second.from()) && poses.count(kv.second.to()))
		{
			usable.insert(kv);
		}
	}

	printf("db=%s poses=%zu links=%zu initial_cost=%.4f mode=%s%s iterations=%d epsilon=%g reps=%d\n",
		dbPath.c_str(), poses.size(), usable.size(),
		refCost(poses, usable, slam2d), slam2d ? "2d" : "3d", ba ? "+ba" : "",
		iterations, epsilon, reps);

	if(ba)
	{
		return runBaBenchmark(dbPath, parameters, strategies, reps, iterations, epsilon,
			slam2d, rootId, poses, usable);
	}

	std::map<Optimizer::Type, std::vector<RunResult> > results;
	for(int rep = 0; rep < reps; ++rep)
	{
		for(Optimizer::Type t : strategies)
		{
			results[t].push_back(
				runOnce(t, parameters, iterations, epsilon, slam2d, rootId, poses, usable));
		}
	}

	for(Optimizer::Type t : strategies)
	{
		std::vector<double> ms, cost, iters, rss;
		for(const RunResult & r : results[t])
		{
			if(r.ok)
			{
				ms.push_back(r.ms);
				cost.push_back(r.cost);
				iters.push_back(r.iterations);
				rss.push_back(r.maxRssMb);
			}
		}
		if(ms.empty())
		{
			printf("%-6s FAILED\n", strategyName(t));
			continue;
		}
		printf("%-6s solve_ms=%-9.2f iters=%-4.0f cost=%-12.4f maxrss_mb=%.0f (%zu/%zu runs ok)\n",
			strategyName(t), median(ms), median(iters), median(cost), median(rss),
			ms.size(), results[t].size());
	}
	return 0;
}
