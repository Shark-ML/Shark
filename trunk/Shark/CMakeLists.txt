#####################################################################
#       Shark Machine Learning Library   
#       Top-Level CMake driver file        
#       Optionally included sub-probjects:
#         * Test/CMakeLists.txt                
#         * examples/CMakeLists.txt                                
#         * doc/CMakeLists.txt                           
#####################################################################
PROJECT( shark CXX )

CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
CMAKE_POLICY(SET CMP0003 NEW)
ENABLE_TESTING()

SET( EXECUTABLE_OUTPUT_PATH bin )
SET( LIBRARY_OUTPUT_PATH lib )

ENABLE_LANGUAGE(C) # Fix small issue with OpenMP on some platforms/versions

#####################################################################
#                       Dynamic or static 
#####################################################################

OPTION( OPT_DYNAMIC_LIBRARY
	"Compile shark into a dynamic library instead of a static one."
	OFF
)

IF( OPT_DYNAMIC_LIBRARY )
    set(Boost_USE_STATIC_LIBS OFF)
    add_definitions(-DBOOST_TEST_DYN_LINK)
ELSE( OPT_DYNAMIC_LIBRARY )
    set(Boost_USE_STATIC_LIBS ON)
    SET( LIBRARY_FLAGS STATIC )
ENDIF( OPT_DYNAMIC_LIBRARY )

#####################################################################
#		Official Releases are tagged
#####################################################################
OPTION( OPT_OFFICIAL_RELEASE "Is this an official Shark release." OFF )
mark_as_advanced( OPT_OFFICIAL_RELEASE ) 

IF( OPT_OFFICIAL_RELEASE )
	ADD_DEFINITIONS( -DSHARK_OFFICIAL_RELEASE )
ENDIF( OPT_OFFICIAL_RELEASE )

#####################################################################
#		Marks the official version
#####################################################################
SET( SHARK_VERSION_MAJOR 3 )
SET( SHARK_VERSION_MINOR 0 )
SET( SHARK_VERSION_PATCH 0 )
set(SHARK_VERSION
  ${SHARK_VERSION_MAJOR}.${SHARK_VERSION_MINOR}.${SHARK_VERSION_PATCH})

ADD_DEFINITIONS(
 -DSHARK_VERSION_MAJOR=${SHARK_VERSION_MAJOR}
 -DSHARK_VERSION_MINOR=${SHARK_VERSION_MINOR}
 -DSHARK_VERSION_PATCH=${SHARK_VERSION_PATCH}
)

CONFIGURE_FILE (
  "${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Core/Shark.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Core/Shark.h"
  )

#####################################################################
#	Adjustments for cpack and deb package generation
#####################################################################
SET( CPACK_PACKAGE_NAME "libshark-i386" )
SET( CPACK_BUNDLE_NAME "libshark" )
SET( CPACK_PACKAGE_VENDOR "Institut fur Neuroinformatik, Ruhr-Universitaet Bochum" )
SET( CPACK_PACKAGE_VERSION_MAJOR ${SHARK_VERSION_MAJOR} )
SET( CPACK_PACKAGE_VERSION_MINOR ${SHARK_VERSION_MINOR} )
SET( CPACK_PACKAGE_VERSION_PATCH ${SHARK_VERSION_PATCH} )
SET( CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/gpl-3.0.txt )

if (UNIX AND NOT WIN32)
 if (CMAKE_SIZEOF_VOID_P MATCHES "8")
   SET( CPACK_PACKAGE_NAME "libshark-x86_64" )
   SET( CPACK_DEBIAN_PACKAGE_NAME libshark-x86-64 )
   SET( CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64 )
 else (CMAKE_SIZEOF_VOID_P MATCHES "8")
   SET( CPACK_PACKAGE_NAME "libshark-i386" )
   SET( CPACK_DEBIAN_PACKAGE_NAME libshark-i386 )
   SET( CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386 )
 endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif (UNIX AND NOT WIN32)

SET( CPACK_DEBIAN_PACKAGE_VERSION ${SHARK_VERSION_MAJOR}.${SHARK_VERSION_MINOR}.${SHARK_VERSION_PATCH} )

SET( CPACK_DEBIAN_PACKAGE_DESCRIPTION
     "SHARK is a modular C++ library for the design and optimization of adaptive systems. It provides methods for linear and nonlinear optimization, in particular evolutionary and gradient-based algorithms, kernel-based learning algorithms and neural networks, and various other machine learning techniques. SHARK serves as a toolbox to support real world applications as well as research indifferent domains of computational intelligence and machine learning. The sources are compatible with the following platforms: Windows, Solaris, MacOS X, and Linux."
)
# SET( CPACK_DEBIAN_PACKAGE_DEPENDS ... )
# SET( DEBIAN_PACKAGE_BUILDS_DEPENDS ... )
SET( CPACK_DEBIAN_PACKAGE_MAINTAINER "Christian Igel <c.igel@ieee.org>" )

#####################################################################
#    Adjust include, lib, example and doc paths for installation
#####################################################################
IF( UNIX )
	SET( SHARK_INSTALL_INCLUDE_DIR include/ )
	SET( SHARK_INSTALL_LIB_DIR lib/ )
	SET( SHARK_INSTALL_CONTRIB_DIR share/shark/contrib/ )
	SET( SHARK_INSTALL_EXAMPLE_DIR share/shark/examples/ )
	SET( SHARK_INSTALL_DOC_DIR share/shark/doc/ )
ELSE( UNIX )
	SET( SHARK_INSTALL_INCLUDE_DIR include/shark/ )
	SET( SHARK_INSTALL_LIB_DIR lib/ )
	SET( SHARK_INSTALL_CONTRIB_DIR contrib/ )
	SET( SHARK_INSTALL_EXAMPLE_DIR examples/ )
	SET( SHARK_INSTALL_DOC_DIR doc )
ENDIF( UNIX )

#####################################################################
#           Enable installer and package generation
#####################################################################
INCLUDE( CPack )

#####################################################################
#           Explicit macro setup for debug configuration
#####################################################################
# enable or disable debugging, default is Release
IF(NOT CMAKE_BUILD_TYPE)
       SET(CMAKE_BUILD_TYPE "Release")
ENDIF(NOT CMAKE_BUILD_TYPE)
IF((CMAKE_BUILD_TYPE MATCHES "debug") OR (CMAKE_BUILD_TYPE MATCHES "DEBUG"))
	SET(CMAKE_BUILD_TYPE "Debug")
ENDIF((CMAKE_BUILD_TYPE MATCHES "debug") OR (CMAKE_BUILD_TYPE MATCHES "DEBUG"))
IF((CMAKE_BUILD_TYPE MATCHES "release") OR (CMAKE_BUILD_TYPE MATCHES "RELEASE"))
	SET(CMAKE_BUILD_TYPE "Release")
ENDIF((CMAKE_BUILD_TYPE MATCHES "release") OR (CMAKE_BUILD_TYPE MATCHES "RELEASE"))

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
	ADD_DEFINITIONS(-DDEBUG )
	ADD_DEFINITIONS("-DBOOST_UBLAS_TYPE_CHECK=0" )
	#we want to be warned in debug mode
	IF(UNIX)
		IF(CMAKE_COMPILER_IS_GNUCC)
			ADD_DEFINITIONS(-Wall)
		ENDIF(CMAKE_COMPILER_IS_GNUCC)
	ENDIF(UNIX)
ENDIF (CMAKE_BUILD_TYPE MATCHES "Debug")
IF(CMAKE_BUILD_TYPE MATCHES "Release")
	ADD_DEFINITIONS(-DNDEBUG )
	ADD_DEFINITIONS(-DBOOST_UBLAS_NDEBUG )
ENDIF (CMAKE_BUILD_TYPE MATCHES "Release")

#####################################################################
#           Report compilers
#####################################################################
message(STATUS "C++ compiler: " ${CMAKE_CXX_COMPILER})
message(STATUS "C compiler: " ${CMAKE_C_COMPILER})

#####################################################################
#           Boost configuration
#####################################################################
SET(Boost_USE_MULTITHREADED ON)
ADD_DEFINITIONS( -DBOOST_PARAMETER_MAX_ARITY=15 -DBOOST_FILESYSTEM_VERSION=3 )
FIND_PACKAGE( Boost 1.45.0 COMPONENTS system date_time filesystem program_options signals serialization thread unit_test_framework REQUIRED )
include_directories ( ${Boost_INCLUDE_DIR} ) # CI: changed Boost_INCLUDE_DIRS to Boost_INCLUDE_DIR ... hope that this is always works
IF( WIN32 )
	link_directories( ${Boost_LIBRARY_DIR} )
ELSE( WIN32 )
	FIND_LIBRARY( PTHREAD_LIBRARY pthread )
	SET( LINK_LIBRARIES
		${Boost_LIBRARIES}
		${PTHREAD_LIBRARY}
	)
ENDIF( WIN32 )
mark_as_advanced( 
		  Boost_DIR 
		  PTHREAD_LIBRARY
)
mark_as_advanced( CLEAR
                  Boost_INCLUDE_DIR
                  Boost_LIBRARY_DIRS
) 
message( STATUS "Using boost from " ${Boost_LIBRARY_DIRS} )

#####################################################################
#		Optionally enable OpenMP 
#####################################################################
OPTION( OPT_ENABLE_OPENMP "Enable OpenMP if available." OFF )

IF( OPT_ENABLE_OPENMP )
	FIND_PACKAGE( OpenMP QUIET )

	IF( OPENMP_FOUND )
		MESSAGE( STATUS "OpenMP...found" )
		ADD_DEFINITIONS( -DSHARK_USE_OPENMP )
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	ELSE( OPENMP_FOUND )
		MESSAGE( WARNING "OpenMP...not found" )
	ENDIF( OPENMP_FOUND )
ENDIF( OPT_ENABLE_OPENMP )

#####################################################################
#           HDF5 configuration 
#####################################################################
FIND_PACKAGE(HDF5 COMPONENTS C CXX HL QUIET)
mark_as_advanced( HDF5_DIR ) 
# mark_as_advanced( CLEAR HDF5_LIBRARY_DIRS DF5_INCLUDE_DIRS ) 
IF( HDF5_FOUND )
    MESSAGE(STATUS "HDF5...found")
    INCLUDE_DIRECTORIES( ${HDF5_INCLUDE_DIRS} )
    IF( WIN32 )
        LINK_DIRECTORIES( ${HDF5_LIBRARY_DIRS} )
    ELSE( WIN32 )
        SET(LINK_LIBRARIES ${LINK_LIBRARIES} ${HDF5_LIBRARIES} )
    ENDIF( WIN32 )
ELSE( HDF5_FOUND )
    MESSAGE(STATUS "HDF5 not found, skip")
ENDIF( HDF5_FOUND )

#####################################################################
#           ATLAS configuration
#####################################################################
OPTION( OPT_ENABLE_ATLAS "Use ATLAS as Linear Algebra Library" OFF )
IF ( OPT_ENABLE_ATLAS )
	find_library(CBLAS_LIBRARY
		cblas
		HINTS
		${ATLAS_ROOT}/lib /opt/local/lib
	)
	find_library(CLAPACK_LIBRARY
		lapack
		HINTS
		${ATLAS_ROOT}/lib /opt/local/lib
	)
	find_library(ATLAS_LIBRARY
		atlas
		HINTS
		${ATLAS_ROOT}/lib /opt/local/lib
	)

	get_filename_component(ATLAS_PATH @CBLAS_LIBRARY@ PATH )
	SET (ATLAS_PATH ${ATLAS_PATH}/.. )
	mark_as_advanced(
		CBLAS_LIBRARY
		CLAPACK_LIBRARY
		ATLAS_LIBRARY
	)
	IF ( CBLAS_LIBRARY )
		message(STATUS "ATLAS found....true")
		SET(LINK_LIBRARIES ${LINK_LIBRARIES} ${CLAPACK_LIBRARY} )
		SET(LINK_LIBRARIES ${LINK_LIBRARIES} ${CBLAS_LIBRARY} )
		SET(LINK_LIBRARIES ${LINK_LIBRARIES} ${ATLAS_LIBRARY} )
		ADD_DEFINITIONS( -DSHARK_USE_ATLAS )
		IF ( ATLAS_ROOT )
				message(STATUS "ATLAS root: " ${ATLAS_ROOT})
				include_directories ( ${ATLAS_ROOT}/include/ ${ATLAS_ROOT}/include/atlas/ )
		ELSE( ATLAS_ROOT )
				message(STATUS "ATLAS path: " ${ATLAS_PATH})
				include_directories ( ${ATLAS_PATH}/include/ ${ATLAS_PATH}/include/atlas/ )
		ENDIF(ATLAS_ROOT)
	ELSE ( CBLAS_LIBRARY )
		message(STATUS "ATLAS found....false")
	ENDIF ( CBLAS_LIBRARY )
ENDIF ( OPT_ENABLE_ATLAS )

#####################################################################
#                       Static Code Analysis
#####################################################################
mark_as_advanced( CPP_CHECK )
IF( UNIX )
    FIND_PROGRAM( CPP_CHECK cppcheck )
    IF( CPP_CHECK )
    	MESSAGE(STATUS "cppcheck available for static code analysis." )
    	ADD_CUSTOM_TARGET(
		code_analysis
		${CPP_CHECK} --enable=all --xml --force -I${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR} 2> StaticAnalysis.xml
	)
    ENDIF( CPP_CHECK )
ENDIF( UNIX )

#####################################################################
#                Adjust compiler flags and settings for MSVC
#####################################################################
IF( MSVC )
    message(STATUS "Win32")
	SET( DISABLE_WARNINGS "/wd4250 /wd4251 /wd4275 /wd4800 /wd4308" )
	#4250: inherit via dominance
	#4251: needs to have dll-interface
	#4275: non-dll interface used as base for dll-interface class
	#4308: Negative integral constant
	SET( ADDITIONAL_COMPILER_FLAGS "${DISABLE_WARNINGS}" )

	STRING( REPLACE "/O2" "/Od" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" )

	SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${ADDITIONAL_COMPILER_FLAGS}" )
	SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ADDITIONAL_COMPILER_FLAGS}" )
	SET( CMAKE_CXX_FLAGS_MINSIZERELEASE "${CMAKE_CXX_FLAGS_MINSIZERELEASE} ${ADDITIONAL_COMPILER_FLAGS}" CACHE STRING "compiler-flags" FORCE )
	SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${ADDITIONAL_COMPILER_FLAGS}" CACHE STRING "compiler-flags" FORCE )

	SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "compiler-flags" FORCE )
	SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "compiler-flags" FORCE )
	SET( CMAKE_CXX_FLAGS_MINSIZERELEASE "${CMAKE_CXX_FLAGS_MINSIZERELEASE}" CACHE STRING "compiler-flags" FORCE )
	SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "compiler-flags" FORCE )

	ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
	ADD_DEFINITIONS(-DNOMINMAX)
	ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
	ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE) 
	ADD_DEFINITIONS(-D_SCL_SECURE_NO_WARNINGS)
	ADD_DEFINITIONS(-D_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1)
ENDIF( MSVC )


IF( UNIX )
	IF( APPLE )
	  	SET(CMAKE_CXX_CREATE_STATIC_LIBRARY "libtool -static -o <TARGET> <LINK_FLAGS> <OBJECTS>" )
	ENDIF( APPLE )
ENDIF( UNIX )

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
INCLUDE_DIRECTORIES( include/ )
ADD_SUBDIRECTORY( include )
ADD_SUBDIRECTORY( src )

#####################################################################
#                       Include Examples
#####################################################################
OPTION( OPT_COMPILE_EXAMPLES "Compile example programs." ON )

#IF( OPT_COMPILE_EXAMPLES )
	ADD_SUBDIRECTORY( examples )
#ENDIF( OPT_COMPILE_EXAMPLES )

#####################################################################
#                       Include Unit Tests
#####################################################################
OPTION( OPT_MAKE_TESTS "Compile unit tests." ON )

IF( OPT_MAKE_TESTS)
	ADD_SUBDIRECTORY( Test )
ENDIF( OPT_MAKE_TESTS )

#####################################################################
#                       Include Documentation
#####################################################################
OPTION( OPT_COMPILE_DOCUMENTATION "Compile documentation." OFF )

IF( OPT_COMPILE_DOCUMENTATION )
	ADD_SUBDIRECTORY( doc )
ENDIF( OPT_COMPILE_DOCUMENTATION )


#TARGET_LINK_LIBRARIES( Statistics shark ${LINK_LIBRARIES} )



###################################################################
#                       CMAKE PACKAGING
###################################################################


if(WIN32 AND NOT CYGWIN)
  set(INSTALL_CMAKE_DIR CMake)
else()
  set(INSTALL_CMAKE_DIR lib/CMake/Shark/)
endif()

# Add all targets to the build-tree export set
export(TARGETS shark Version FILE "${PROJECT_BINARY_DIR}/SharkTargets.cmake")
 
# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
#export(PACKAGE Shark)

# Create the SharkConfig.cmake and SharkConfigVersion files
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include/" "${Boost_INCLUDE_DIRS}")
set(CONF_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_file(SharkConfig.cmake.in "${PROJECT_BINARY_DIR}/SharkConfig.cmake")
set(CONF_TARGET_DIR "${CMAKE_INSTALL_PREFIX}/${INSTALL_CMAKE_DIR}")
set(CONF_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/" "${Boost_INCLUDE_DIRS}")
configure_file(SharkConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SharkConfig.cmake")
configure_file(SharkConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/SharkConfigVersion.cmake")

# Install the SharkConfig.cmake and SharkConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SharkConfig.cmake"
  "${PROJECT_BINARY_DIR}/SharkConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
 
# Install the export set for use with the install-tree
install(EXPORT SharkTargets DESTINATION
  "${INSTALL_CMAKE_DIR}" COMPONENT dev)



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
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
