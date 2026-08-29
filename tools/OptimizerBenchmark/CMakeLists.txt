ADD_EXECUTABLE(optimizer_benchmark main.cpp)

TARGET_LINK_LIBRARIES(optimizer_benchmark rtabmap_core)

SET_TARGET_PROPERTIES( optimizer_benchmark
	PROPERTIES OUTPUT_NAME ${PROJECT_PREFIX}-optimizerBenchmark)

INSTALL(TARGETS optimizer_benchmark
		RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
		BUNDLE DESTINATION "${CMAKE_BUNDLE_LOCATION}" COMPONENT runtime)
