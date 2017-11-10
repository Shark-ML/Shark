cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(Shark_Benchmarks)

find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})

macro( SHARK_ADD_BENCHMARK SRC NAME )

	add_executable(${NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SRC})
	target_link_libraries(${NAME} ${SHARK_LIBRARIES})
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 11)
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
endmacro()

SHARK_ADD_BENCHMARK(nearest_neighbours.cpp NearestNeighbours)
SHARK_ADD_BENCHMARK(random_forrest.cpp Random_Forrest)
SHARK_ADD_BENCHMARK(kernel_csvm.cpp Kernel_CSvm)
SHARK_ADD_BENCHMARK(linear_csvm.cpp Linear_CSvm)
SHARK_ADD_BENCHMARK(linear_regression.cpp Linear_Regression)
SHARK_ADD_BENCHMARK(ridge_regression.cpp Ridge_Regression)
SHARK_ADD_BENCHMARK(logistic_regression_LBFGS.cpp Logistic_Regression_LBFGS)
SHARK_ADD_BENCHMARK(logistic_regression_SAG.cpp Logistic_Regression_SAG)
#SHARK_ADD_BENCHMARK(hypervolume_algorithms.cpp HypervolumeAlgorithms)
