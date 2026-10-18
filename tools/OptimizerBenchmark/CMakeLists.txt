ADD_EXECUTABLE(optimizer_benchmark main.cpp)

TARGET_LINK_LIBRARIES(optimizer_benchmark rtabmap_core)

SET_TARGET_PROPERTIES( optimizer_benchmark
	PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-optimizerBenchmark)

INSTALL(TARGETS optimizer_benchmark
		RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
		BUNDLE DESTINATION "${CMAKE_BUNDLE_LOCATION}" COMPONENT runtime)

# The arael solve of the 2loops sample (fetched by scripts/fetch_test_data.sh)
# reaches the same optimum timed in a fresh process and after a warm-up solve.
SET(_benchmark_db ${PROJECT_SOURCE_DIR}/data/tests/2loops_workspace_3IT.db)
IF(BUILD_TESTING AND WITH_ARAEL AND ARAEL_FOUND AND EXISTS ${_benchmark_db})
	FOREACH(_warmup 0 1)
		ADD_TEST(NAME optimizer_benchmark_warmup_${_warmup}
			COMMAND optimizer_benchmark --strategy 5 --reps 1 --warmup ${_warmup} ${_benchmark_db})
		SET_TESTS_PROPERTIES(optimizer_benchmark_warmup_${_warmup} PROPERTIES
			PASS_REGULAR_EXPRESSION "arael +solve_ms=[0-9.]+ +iters=6 +cost=123\\.0414 ")
	ENDFOREACH()
	# --ref-strategy: arael scored against its own solution is at distance 0,
	# and bundle adjustment refuses the option
	ADD_TEST(NAME optimizer_benchmark_ref_strategy
		COMMAND optimizer_benchmark --strategy 5 --reps 1 --ref-strategy 5 ${_benchmark_db})
	SET_TESTS_PROPERTIES(optimizer_benchmark_ref_strategy PROPERTIES
		PASS_REGULAR_EXPRESSION "arael +solve_ms=[0-9.]+ +iters=6 +cost=123\\.0414 +ref_rmse_m=0\\.0000 ")
	# --converge: the convergence search finds the iterations arael needs and
	# times it there, at the same optimum
	ADD_TEST(NAME optimizer_benchmark_converge
		COMMAND optimizer_benchmark --strategy 5 --reps 1 --converge ${_benchmark_db})
	SET_TESTS_PROPERTIES(optimizer_benchmark_converge PROPERTIES
		PASS_REGULAR_EXPRESSION "arael +solve_ms=[0-9.]+ +iters=5 +cost=123\\.0414 +converge_iters=5 converged=1 ")
	ADD_TEST(NAME optimizer_benchmark_ref_strategy_ba_refused
		COMMAND optimizer_benchmark --ba --ref-strategy 5 ${_benchmark_db})
	SET_TESTS_PROPERTIES(optimizer_benchmark_ref_strategy_ba_refused PROPERTIES
		PASS_REGULAR_EXPRESSION "--ref-strategy needs a pose graph")
ENDIF()
