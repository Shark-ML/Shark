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

SHARK_ADD_BENCHMARK(conv2d.cpp Conv2D)
SHARK_ADD_BENCHMARK(gemm.cpp Gemm)
SHARK_ADD_BENCHMARK(syrk.cpp Syrk)
SHARK_ADD_BENCHMARK(trmm.cpp Trmm)
SHARK_ADD_BENCHMARK(potrf.cpp Potrf)