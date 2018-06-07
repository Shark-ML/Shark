#####################################################################
#       Shark Machine Learning Library
#       Top-Level CMake driver file
#       Optionally included sub-probjects:
#         * Test/CMakeLists.txt
#         * examples/CMakeLists.txt
#         * doc/CMakeLists.txt
#####################################################################
project( shark )
cmake_minimum_required( VERSION 3.1 FATAL_ERROR)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
INCLUDE (CheckFunctionExists)


cmake_policy(SET CMP0003 NEW)
if(POLICY CMP0042)
	cmake_policy(SET CMP0042 NEW)
endif()
if(POLICY CMP0053)
	cmake_policy(SET CMP0053 NEW)
endif()
#=========================================================
# Output directories.
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${shark_BINARY_DIR}/bin")
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
	if(UNIX)
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${shark_BINARY_DIR}/lib")
	else()
		set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${shark_BINARY_DIR}/bin")
	endif()
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${shark_BINARY_DIR}/lib")
endif()
mark_as_advanced( 
	CMAKE_RUNTIME_OUTPUT_DIRECTORY
	CMAKE_LIBRARY_OUTPUT_DIRECTORY
	CMAKE_ARCHIVE_OUTPUT_DIRECTORY
)

#####################################################################
# Static/Shared libraries
#####################################################################
option(BUILD_SHARED_LIBS "Compile shark into a dynamic library instead of a static one." OFF)
if(BUILD_SHARED_LIBS)
	set(SHARK_USE_DYNLIB 1)
endif()

#####################################################################
# Version information
#####################################################################
option(BUILD_OFFICIAL_RELEASE "Is this an official Shark release." OFF )
mark_as_advanced( BUILD_OFFICIAL_RELEASE )

set(SHARK_VERSION_MAJOR 4)
set(SHARK_VERSION_MINOR 0)
set(SHARK_VERSION_PATCH 0)
set(SHARK_VERSION ${SHARK_VERSION_MAJOR}.${SHARK_VERSION_MINOR}.${SHARK_VERSION_PATCH})
set(SHARK_SOVERSION 0)


#####################################################################
#	Adjustments for cpack and deb package generation
#####################################################################
set(CPACK_GENERATOR TGZ DEB)
set(CPACK_PACKAGE_NAME "libshark")
set(CPACK_BUNDLE_NAME "libshark")
set(CPACK_PACKAGE_VENDOR "Institut fur Neuroinformatik, Ruhr-Universitaet Bochum")
set(CPACK_PACKAGE_VERSION_MAJOR ${SHARK_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${SHARK_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${SHARK_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${SHARK_VERSION_MAJOR}.${SHARK_VERSION_MINOR}.${SHARK_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/COPYING.LESSER)

#Debian Compatible naming by default
if(CMAKE_SIZEOF_VOID_P MATCHES "8")
	set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
else()
	set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386)
endif()
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE})
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libboost-all-dev (>=1.54)")

set(CPACK_DEBIAN_PACKAGE_DESCRIPTION
	"SHARK is a modular C++ library for the design and optimization of adaptive systems. It provides methods for linear and nonlinear optimization, in particular evolutionary and gradient-based algorithms, kernel-based learning algorithms and neural networks, and various other machine learning techniques. SHARK serves as a toolbox to support real world applications as well as research indifferent domains of computational intelligence and machine learning. The sources are compatible with the following platforms: Windows, Solaris, MacOS X, and Linux."
)
set( CPACK_DEBIAN_PACKAGE_MAINTAINER "Christian Igel <c.igel@ieee.org>" )

#####################################################################
#    Adjust include, lib, example and doc paths for installation
#####################################################################
if( UNIX )
	include(GNUInstallDirs)
	set( SHARK_INSTALL_BIN_DIR ${CMAKE_INSTALL_BINDIR} CACHE STRING "" )
	set( SHARK_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE STRING "" )
	set( SHARK_INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE STRING "" )
	set( SHARK_INSTALL_CONTRIB_DIR ${CMAKE_INSTALL_DATADIR}/shark/contrib CACHE STRING "" )
	set( SHARK_INSTALL_EXAMPLE_DIR ${CMAKE_INSTALL_DATADIR}/shark/examples CACHE STRING "" )
	set( SHARK_INSTALL_DOC_DIR ${CMAKE_INSTALL_DATADIR}/shark/doc CACHE STRING "" )
else()
	set( SHARK_INSTALL_BIN_DIR bin )
	set( SHARK_INSTALL_INCLUDE_DIR include/shark/ )
	set( SHARK_INSTALL_LIB_DIR lib/ )
	set( SHARK_INSTALL_CONTRIB_DIR contrib/ )
	set( SHARK_INSTALL_EXAMPLE_DIR examples/ )
	set( SHARK_INSTALL_DOC_DIR doc )
endif()

#####################################################################
#           Enable installer and package generation
#####################################################################
 include( CPack )

#####################################################################
#           Explicit macro setup for debug configuration
#####################################################################
# enable or disable debugging, default is Release
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

if (UNIX)
    add_compile_options("$<$<CONFIG:DEBUG>:-Wall>")
endif()

list(APPEND COMPILE_DEFINITIONS_RELEASE NDEBUG)
message(STATUS "Will build: " ${CMAKE_BUILD_TYPE})

#####################################################################
#           Boost configuration
#####################################################################
set(Boost_USE_STATIC_LIBS OFF CACHE BOOL "use static libraries from Boost")
set(Boost_USE_MULTITHREADED ON)
add_definitions(-DBOOST_PARAMETER_MAX_ARITY=15)
add_definitions(-DBOOST_FILESYSTEM_VERSION=3)
add_definitions(-DBOOST_RESULT_OF_USE_DECLTYPE)

# Should we link the boost test dynamically
if(NOT Boost_USE_STATIC_LIBS)
	add_definitions(-DBOOST_TEST_DYN_LINK)
	add_definitions(-DBOOST_ALL_DYN_LINK)
endif()

find_package( 
	Boost 1.48.0 REQUIRED COMPONENTS
	serialization
	filesystem system
)

if(NOT Boost_FOUND)
	message(FATAL_ERROR "Please make sure Boost 1.48.0 is installed on your system")
endif()

if (WIN32)
	# disable autolinking in boost
	add_definitions( -DBOOST_ALL_NO_LIB )
endif()

include_directories(SYSTEM ${Boost_INCLUDE_DIR} )
link_directories( ${Boost_LIBRARY_DIR} )

# Set the libraries needed by Shark
list(APPEND LINK_LIBRARIES ${Boost_LIBRARIES})

message( STATUS "Using boost from " ${Boost_LIBRARY_DIR} )

#####################################################################
#		OpenMP
#####################################################################
option( ENABLE_OPENMP "Enable OpenMP" ON )
if( ENABLE_OPENMP )
	find_package( OpenMP QUIET )
	if( OPENMP_FOUND )
		message( STATUS "OpenMP found" )
		set(SHARK_USE_OPENMP 1)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

		set(SHARK_REQUIRED_C_FLAGS "${OpenMP_C_FLAGS}")
		set(SHARK_REQUIRED_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
		set(SHARK_REQUIRED_EXE_LINKER_FLAGS "${OpenMP_EXE_LINKER_FLAGS}")

	else()
		message( STATUS "OpenMP not found" )
	endif()
else()
	message( STATUS "Building without OpenMP as requested." )
endif()

#####################################################################
#           HDF5 configuration
#####################################################################
find_package(HDF5 COMPONENTS C CXX HL QUIET)
mark_as_advanced(HDF5_DIR)
if(HDF5_FOUND)
	if(HDF5_C_COMPILER_EXECUTABLE AND HDF5_CXX_COMPILER_EXECUTABLE)
		message(STATUS "Checking HDF5 installation: HDF5 installation seems ok.")
		include_directories( ${HDF5_INCLUDE_DIRS} )
		list(APPEND LINK_LIBRARIES ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
	else()
		message(STATUS "Checking HDF5 installation:HDF5 package might be broken.")
		if(NOT( HDF5_C_COMPILER_EXECUTABLE))
			message(STATUS "  C Compiler Extension not found.")
		endif()
		if(NOT( HDF5_CXX_COMPILER_EXECUTABLE))
			message(STATUS "  CXX Compiler Extension not found.")
		endif()
			message(STATUS "Disabling HDF5.")
		unset( HDF5_FOUND )
	endif()
else()
	message(STATUS "HDF5 not found, skip")
endif()

#####################################################################
#           BLAS configuration
#####################################################################

option( ENABLE_SIMD "Use experimental SIMD feature in linear algebra" OFF )
if( ENABLE_SIMD )
	set( SHARK_USE_SIMD 1)
endif()

option( ENABLE_CBLAS "Use Installed Linear Algebra Library" ON )
set(CBLAS_LIBRARY_PATH "/usr/lib64/" CACHE PATH "Linear algebra library path")
set(CBLAS_INCLUDE_PATH "/usr/include/" CACHE PATH "Linear algebra include path")
if( ENABLE_CBLAS )
	set(CBLAS_VENDOR FALSE)
	if( APPLE )
		set(CBLAS_VENDOR "Accelerate")
		set(CBLAS_INCLUDES "")
		set(CBLAS_LIBRARIES "-framework Accelerate" )
	else()
		#todo: do a propper vendor check
		find_library(OPENBLAS_LIBRARY openblas
			HINTS ${CBLAS_LIBRARY_PATH}
		)
		find_library(CBLAS_LIBRARY cblas
			HINTS ${CBLAS_LIBRARY_PATH}
		)
		find_library(CLAPACK_LIBRARY lapack
			HINTS ${CBLAS_LIBRARY_PATH}
		)
		find_library(ATLAS_LIBRARY atlas
			HINTS ${CBLAS_LIBRARY_PATH}
		)
		mark_as_advanced(
			OPENBLAS_LIBRARY
			CBLAS_LIBRARY
			CLAPACK_LIBRARY
			ATLAS_LIBRARY
		)
		
		find_file(CBLAS_INCLUDES cblas.h 
			PATHS ${CBLAS_INCLUDE_PATH}
		)

		if( OPENBLAS_LIBRARY AND CBLAS_INCLUDES)
			set(CBLAS_VENDOR "OpenBLAS")
			set(CBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
		elseif( CBLAS_LIBRARY AND CLAPACK_LIBRARY AND ATLAS_LIBRARY AND CBLAS_INCLUDES)
			set(CBLAS_VENDOR "ATLAS")
			set(CBLAS_LIBRARIES ${CLAPACK_LIBRARY} ${CBLAS_LIBRARY} ${ATLAS_LIBRARY})
		elseif( CBLAS_LIBRARY AND CBLAS_INCLUDES)
			#check that we can compile a basic program with the libraries we have found
			#vendor versions might come with additional libraries which would be bad.
			try_compile(CBLAS_COMPILE 
				"${PROJECT_BINARY_DIR}/cBlasCheck"
				"${CMAKE_SOURCE_DIR}/cBlasCheck.cpp"
				CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CBLAS_INCLUDE_PATH}"
				LINK_LIBRARIES ${CBLAS_LIBRARY}
			)
			if(CBLAS_COMPILE)
				set(CBLAS_VENDOR "GENERIC")
				set(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
			else()
				message(WARNING "Can not compile basic program with cblas library")
			endif()
		endif()
	endif()
	
	if(CBLAS_VENDOR)
		message(STATUS "CBLAS FOUND: " ${CBLAS_VENDOR})
		message(STATUS "CBLAS include file " ${CBLAS_INCLUDES} )
		message(STATUS "CBLAS library files " ${CBLAS_LIBRARIES} )
		set(SHARK_USE_CBLAS 1)
		list(APPEND EXTRA_INCLUDE_DIRECTORIES ${CBLAS_INCLUDE_PATH} )
		list(APPEND LINK_LIBRARIES ${CBLAS_LIBRARIES})
		include_directories ( ${CBLAS_INCLUDE_PATH} )
	else()
		message(STATUS "No usable CBLAS Library found. No fast linear Algebra used.")
	endif()
	
	#Special setup for ATLAS
	if( CBLAS_VENDOR MATCHES "ATLAS" )
		set( SHARK_USE_ATLAS_LAPACK 1) # ATLAS always contains some LAPACK methods that we can use
	endif()

	if(CLAPACK_LIBRARY)
		#check for full lapack
		set(CMAKE_REQUIRE_QUIET 1)
		set(CMAKE_REQUIRED_LIBRARIES ${CBLAS_LIBRARIES})
		check_function_exists(dsyev_ HAS_FULL_LAPACK)
		
		if( HAS_FULL_LAPACK )
			set( SHARK_USE_LAPACK 1)
			message(STATUS "Using Lapack!")
		endif()
	endif()
endif()

#####################################################################
#           BLAS OpenCL configuration
#####################################################################

option( ENABLE_OPENCL "Use OpenCL and boost.compute" OFF )
if(ENABLE_OPENCL)
	if(${Boost_MINOR_VERSION} LESS 61)
		message(FATAL_ERROR "Boost below 1.61 does not include Boost.Compute.")
	endif()
	find_package(OpenCL REQUIRED)
	include_directories(${OpenCL_INCLUDE_DIRS})
	list( APPEND LINK_LIBRARIES ${OpenCL_LIBRARY})
	set(SHARK_USE_OPENCL 1)
endif()

option( ENABLE_CLBLAST "Use CLBlast as GPU backend" OFF )
if(ENABLE_CLBLAST)
	find_package(CLBlast REQUIRED)
	set(SHARK_USE_CLBLAST 1)
	list( APPEND LINK_LIBRARIES clblast)
endif()

#####################################################################
#                       Static Code Analysis
#####################################################################
option(USE_CPPCHECK "Use CPPCheck Static Code Analysis" OFF)
mark_as_advanced(USE_CPPCHECK)
if(USE_CPPCHECK)
	find_program(CPP_CHECK cppcheck)
	mark_as_advanced(CPP_CHECK)
if(CPP_CHECK)
	message(STATUS "cppcheck available for static code analysis." )
	add_custom_target(code_analysis
		${CPP_CHECK} --enable=all --xml --force -I${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR} 2> StaticAnalysis.xml)
	endif()
endif()

#####################################################################
#                       GCov Code Analysis
#####################################################################
option(USE_GCOV_CHECK  "GCov Coverage Check" OFF)

if(USE_GCOV_CHECK)
find_program(GCOV_CHECK gcov)
if(CMAKE_BUILD_TYPE MATCHES "Release")
	message( FATAL_ERROR 
		"\nYou specified a Release Build."
		"This does not make sense, if Coverage is enabled."
		"Please change Build type to Debug or turn off Coverage."
	)
endif()
	message(STATUS "GCov available for static code analysis. Configuring for Code Check." )
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
	set(CFLAGS "${CFLAGS} -g -O0 -Wall -W --coverage")
	set(LDFLAGS "${LDFLAGS} --coverage")
endif()


#####################################################################
#                Adjust compiler flags and settings for MSVC
#####################################################################
if( MSVC )
	#4250: inherit via dominance
	add_definitions(/wd4250)
	#4251: needs to have dll-interface
	add_definitions(/wd4251)
	#4275: non-dll interface used as base for dll-interface class
	add_definitions(/wd4275)
	#4308: Negative integral constant
	add_definitions(/wd4800)
	add_definitions(/wd4308)

	string( REPLACE "/O2" "/Od" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" )
	set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "compiler-flags" FORCE )

	add_definitions(-D_USE_MATH_DEFINES)
	add_definitions(-DNOMINMAX)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
	add_definitions(-D_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1)

	# Allow multi core compile on Visual Studio
	set(MULTICORE_COMPILE OFF CACHE BOOL "Compile with all available cores")
	if (${MULTICORE_COMPILE})
		add_definitions(${CMAKE_CXX_FLAGS} "/MP")
	endif(${MULTICORE_COMPILE})
endif()

#####################################################################
#                       General Path settings
#####################################################################
include_directories( ${shark_SOURCE_DIR}/include )
include_directories( ${shark_BINARY_DIR}/include )
add_subdirectory( include )
add_subdirectory( src )

#####################################################################
#                       Include Examples
#####################################################################
option( BUILD_EXAMPLES "Build example programs." ON )
add_subdirectory( examples )

#####################################################################
#                       Include Unit Tests
#####################################################################
# Enable Testing
include(CTest)
if(BUILD_TESTING)
	enable_testing()
	add_subdirectory( Test )
endif()

#####################################################################
#                       Include Documentation
#####################################################################
option(BUILD_DOCUMENTATION "Build documentation." OFF)
if(BUILD_DOCUMENTATION)
	add_subdirectory(doc)
endif()

###################################################################
#                       CPACK PACKAGING
###################################################################
set(INSTALL_CMAKE_DIR lib/cmake/Shark/)

# Add all targets to the build-tree export set
export(TARGETS shark SharkVersion FILE "${shark_BINARY_DIR}/SharkTargets.cmake")

# Create the SharkConfig.cmake and SharkConfigVersion files
set(SHARK_USE_FILE "${PROJECT_SOURCE_DIR}/UseShark.cmake")
set(SHARK_TARGETS_FILE "${PROJECT_BINARY_DIR}/SharkTargets.cmake")
set(SHARK_LIBRARIES ${LINK_LIBRARIES} shark)
set(SHARK_LIBRARY_DIRS "${PROJECT_BINARY_DIR}/lib")
set(SHARK_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include" "${Boost_INCLUDE_DIR}" ${EXTRA_INCLUDE_DIRECTORIES})

# Configure the files to be exported
configure_file(SharkConfig.cmake.in "${PROJECT_BINARY_DIR}/SharkConfig.cmake")
configure_file(SharkConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/SharkConfigVersion.cmake")

# Configure the SharkConfig.cmake for the install tree
set(SHARK_CONFIG_CODE "
	# Compute the installation prefix from this SharkConfig.cmake file location.
	get_filename_component(SHARK_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")

# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${INSTALL_CMAKE_DIR}")
foreach(p ${_count})
	set(SHARK_CONFIG_CODE "${SHARK_CONFIG_CODE}
		get_filename_component(SHARK_INSTALL_PREFIX \"\${SHARK_INSTALL_PREFIX}\" DIRECTORY)")
endforeach()
set(SHARK_CONFIG_CODE "${SHARK_CONFIG_CODE}")

set(SHARK_USE_FILE "\${SHARK_INSTALL_PREFIX}/${INSTALL_CMAKE_DIR}UseShark.cmake")
set(SHARK_TARGETS_FILE "\${SHARK_INSTALL_PREFIX}/${INSTALL_CMAKE_DIR}SharkTargets.cmake")
set(SHARK_LIBRARIES ${LINK_LIBRARIES} shark)
set(SHARK_LIBRARY_DIRS "\${SHARK_INSTALL_PREFIX}/lib")
set(SHARK_INCLUDE_DIRS "\${SHARK_INSTALL_PREFIX}/${SHARK_INSTALL_INCLUDE_DIR}" "${Boost_INCLUDE_DIR}" ${EXTRA_INCLUDE_DIRECTORIES})
configure_file(SharkConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SharkConfig.cmake")

# Install the SharkConfig.cmake and SharkConfigVersion.cmake
install(FILES
	"${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SharkConfig.cmake"
	"${PROJECT_BINARY_DIR}/SharkConfigVersion.cmake"
	"${PROJECT_SOURCE_DIR}/UseShark.cmake"
	DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev
)

# Install the export set for use with the install-tree
install(EXPORT SharkTargets DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

#####################################################################
#                  Configure the Shark.h file
#####################################################################
configure_file (
	"${CMAKE_SOURCE_DIR}/include/shark/Core/Shark.h.in"
	"${CMAKE_BINARY_DIR}/include/shark/Core/Shark.h"
)
install(FILES
	"${CMAKE_BINARY_DIR}/include/shark/Core/Shark.h"
	DESTINATION "${SHARK_INSTALL_INCLUDE_DIR}/shark/Core/" COMPONENT deV
)

###################################################################
#                           UNINSTALL                             #
###################################################################
# refer to
# http://www.cmake.org/Wiki/CMake_FAQ#Can_I_do_.22make_uninstall.22_with_CMake.3F
# for details
configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
	IMMEDIATE @ONLY)

add_custom_target(uninstall
	COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
)
