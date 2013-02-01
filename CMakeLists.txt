#####################################################################
#       Shark Machine Learning Library                              #
#       Top-Level CMake driver file                                 #
#       Optionally included sub-probjects:                          #
#         * Test/CMakeLists.txt                                     #
#         * examples/CMakeLists.txt                                 #
#         * doc/CMakeLists.txt                                      #
#####################################################################
PROJECT( shark CXX )

CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
CMAKE_POLICY(SET CMP0003 NEW)

SET( EXECUTABLE_OUTPUT_PATH bin )
SET( LIBRARY_OUTPUT_PATH lib )

ENABLE_LANGUAGE(C) # Fix small issue with OpenMP on some platforms/versions

#####################################################################
#                       Dynamic or static                           #
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
#		Optionally enable OpenMP                            #
#####################################################################
OPTION( OPT_ENABLE_OPENMP "Enable OpenMP if available." OFF )

IF( OPT_ENABLE_OPENMP )
    FIND_PACKAGE( OpenMP QUIET )
ENDIF( OPT_ENABLE_OPENMP )

#####################################################################
#		Enable CTest integration                            #
#####################################################################
ENABLE_TESTING()


#####################################################################
#		Official Releases are tagged.                       #
#####################################################################
OPTION( OPT_OFFICIAL_RELEASE "Is this an official Shark release." OFF )

IF( OPT_OFFICIAL_RELEASE )
	ADD_DEFINITIONS( -DSHARK_OFFICIAL_RELEASE )
ENDIF( OPT_OFFICIAL_RELEASE )

#####################################################################
#		Marks the official version                          #
#####################################################################
SET( SHARK_VERSION_MAJOR 3 )
SET( SHARK_VERSION_MINOR 0 )
SET( SHARK_VERSION_PATCH 0 )

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
#	Pull in cpp-netlib                                          #
#####################################################################
OPTION( OPT_ENABLE_NETWORKING "Make the shark library support http and RESTful APIs." OFF )
IF( OPT_ENABLE_NETWORKING )
    include_directories ( ${CMAKE_CURRENT_SOURCE_DIR}/contrib/cpp-netlib-0.9.0 )
    ADD_DEFINITIONS( -DBOOST_NETWORK_NO_LIB )
ENDIF( OPT_ENABLE_NETWORKING )

#####################################################################
#	Adjustments for cpack and deb package generation            #
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
#    Adjust include, lib, example and doc paths for installation    #
#####################################################################
IF( UNIX )
	SET( SHARK_INSTALL_INCLUDE_DIR include/shark/ )
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
#           Enable installer and package generation	            #
#####################################################################
INCLUDE( CPack )

#####################################################################
#           Explicit macro setup for debug configuration	    #
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
#           Boost configuration	                                    #
#####################################################################
SET(Boost_USE_MULTITHREADED ON)
ADD_DEFINITIONS( -DBOOST_PARAMETER_MAX_ARITY=15 -DBOOST_FILESYSTEM_VERSION=3 )
FIND_PACKAGE( Boost 1.45.0 COMPONENTS system date_time filesystem program_options regex signals serialization thread unit_test_framework REQUIRED )

include_directories ( ${Boost_INCLUDE_DIRS} )
IF( WIN32 )
	link_directories( ${Boost_LIBRARY_DIRS} )
ELSE( WIN32 )
	FIND_LIBRARY( PTHREAD_LIBRARY pthread )
	SET( LINK_LIBRARIES
		${Boost_LIBRARIES}
		${PTHREAD_LIBRARY}
	)
ENDIF( WIN32 )

#####################################################################
#           HDF5 configuration                                      #
#####################################################################
FIND_PACKAGE(HDF5 COMPONENTS C CXX HL QUIET)
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
#           ATLAS configuration	                                    #
#####################################################################
OPTION( OPT_ENABLE_ATLAS "Use ATLAS as Linear Algebra Library" OFF )
IF ( OPT_ENABLE_ATLAS )
	find_library(CBLAS_LIBRARY
		cblas
		HINTS
		${ATLAS_ROOT}/lib
	)
	find_library(CLAPACK_LIBRARY
		lapack
		HINTS
		${ATLAS_ROOT}/lib
	)
	find_library(ATLAS_LIBRARY
		atlas
		HINTS
		${ATLAS_ROOT}/lib
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
		include_directories ( ${ATLAS_ROOT}/include/ )
	ELSE ( CBLAS_LIBRARY )
		message(STATUS "ATLAS found....false")
	ENDIF ( CBLAS_LIBRARY )
ENDIF ( OPT_ENABLE_ATLAS )
#####################################################################
# Enable math constants and prevent windows from generating	    #
# min/max macros polluting the global namespace.		    #
# Also disable VC++ Security warnings              #
#####################################################################
ADD_DEFINITIONS( -D_USE_MATH_DEFINES -DNOMINMAX )
IF( MSVC )
	ADD_DEFINITIONS( -D_SCL_SECURE_NO_WARNINGS )
ENDIF( MSVC )

INCLUDE_DIRECTORIES( include )

#####################################################################
#           RNG Module                                              #
#####################################################################
SET( RNG_HEADERS
    include/shark/Rng/AbstractDistribution.h
    include/shark/Rng/Bernoulli.h
    include/shark/Rng/Binomial.h
    include/shark/Rng/Cauchy.h
    include/shark/Rng/DiffGeometric.h
    include/shark/Rng/Dirichlet.h
    include/shark/Rng/DiscreteUniform.h
    include/shark/Rng/Erlang.h
    include/shark/Rng/Gamma.h
    include/shark/Rng/Geometric.h
    include/shark/Rng/GlobalRng.h
    include/shark/Rng/HyperGeometric.h
    include/shark/Rng/LogNormal.h
    include/shark/Rng/NegExponential.h
    include/shark/Rng/Normal.h
    include/shark/Rng/Poisson.h
    include/shark/Rng/Uniform.h
    include/shark/Rng/Weibull.h
)

SET( RNG_SRCS
)

SOURCE_GROUP( RNG\\include FILES ${RNG_HEADERS} )
SOURCE_GROUP( RNG\\src FILES ${RNG_SRCS} )

#####################################################################
#           LinAlg Module                                           #
#####################################################################
SET( LINALG_HEADERS
	include/shark/LinAlg/eigenvalues.h
	include/shark/LinAlg/Cholesky.h
	include/shark/LinAlg/RQ.h
	include/shark/LinAlg/svd.h
	include/shark/LinAlg/solveTriangular.h
	include/shark/LinAlg/Base.h
	include/shark/LinAlg/Impl/svd.inl
	include/shark/LinAlg/Impl/pivotingRQ.inl
	include/shark/LinAlg/Impl/eigensymmJacobi2.inl
	include/shark/LinAlg/Impl/g_inverse.inl
	include/shark/LinAlg/Impl/eigensort.inl
	include/shark/LinAlg/Impl/Cholesky.inl
	include/shark/LinAlg/Impl/eigen.inl
	include/shark/LinAlg/Impl/eigensymm.inl
	include/shark/LinAlg/Impl/eigenerr.inl
	include/shark/LinAlg/Impl/rankDecomp.inl
	include/shark/LinAlg/Impl/solveTriangular.inl
	include/shark/LinAlg/Impl/VectorStatistics.inl
	include/shark/LinAlg/Impl/solveSystem.inl
	include/shark/LinAlg/Impl/detsymm.inl
	include/shark/LinAlg/Impl/invert.inl
	include/shark/LinAlg/Impl/svdrank.inl
	include/shark/LinAlg/Impl/rank.inl
	include/shark/LinAlg/Impl/linmin.inl
	include/shark/LinAlg/Impl/svdsort.inl
	include/shark/LinAlg/Impl/fft.inl
	include/shark/LinAlg/Impl/dlinmin.inl
	include/shark/LinAlg/Impl/lnsrch.inl
	include/shark/LinAlg/Impl/eigensymmJacobi.inl
	include/shark/LinAlg/VectorStatistics.h
	include/shark/LinAlg/arrayoptimize.h
	include/shark/LinAlg/Inverse.h
	include/shark/LinAlg/solveSystem.h
	include/shark/LinAlg/BLAS/VectorTransformations.h
	include/shark/LinAlg/BLAS/VectorMatrixType.h
	include/shark/LinAlg/BLAS/Permutation.h
	include/shark/LinAlg/BLAS/Initialize.h
	include/shark/LinAlg/BLAS/MatrixVectorOperation.h
	include/shark/LinAlg/BLAS/Proxy.h
	include/shark/LinAlg/BLAS/ublas.h
	include/shark/LinAlg/BLAS/Impl/BlockMatrixVectorOperation.inl
	include/shark/LinAlg/BLAS/Impl/fast_prod_matrix.inl
	include/shark/LinAlg/BLAS/Impl/sumMatrix.inl
	include/shark/LinAlg/BLAS/Impl/numeric_bindings
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/gotoblas
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/gotoblas/cblas_inc.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/gotoblas/syrk.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/gotoblas/gemm.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/gotoblas/trsv.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/gotoblas/trsm.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/gotoblas/gemv.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/gemv.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/trsm.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/trsv.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/potrf.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/potrs.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/trsv.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/cblas_inc.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/syrk.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/gemm.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/gemv.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/trsm.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/potri.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/gemm.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/default
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/default/trsm.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/default/gemv.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/default/syrk.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/default/ublasTags.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/default/gemm.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/default/trsv.h
	include/shark/LinAlg/BLAS/Impl/numeric_bindings/syrk.h
	include/shark/LinAlg/BLAS/Impl/UnaryExpressions.h
	include/shark/LinAlg/BLAS/Impl/fast_prod_vector.inl
	include/shark/LinAlg/BLAS/Impl/repmat.h
	include/shark/LinAlg/BLAS/Impl/BinaryExpressions.h
	include/shark/LinAlg/BLAS/Impl/Initialize.h
	include/shark/LinAlg/BLAS/Impl/symmRankKUpdate.inl
	include/shark/LinAlg/BLAS/BlockProducts.h
	include/shark/LinAlg/BLAS/traits/MetaBlas.h
	include/shark/LinAlg/BLAS/traits/sparse.hpp
	include/shark/LinAlg/BLAS/traits/matrix_raw.hpp
	include/shark/LinAlg/BLAS/traits/matrixContainerMeta.h
	include/shark/LinAlg/BLAS/traits/metafunctions.h
	include/shark/LinAlg/BLAS/traits/vector_raw.hpp
	include/shark/LinAlg/BLAS/traits/vectorContainerMeta.h
	include/shark/LinAlg/BLAS/fastOperations.h
	include/shark/LinAlg/BLAS/StorageAdaptors.h
	include/shark/LinAlg/BLAS/Tools.h
	include/shark/LinAlg/BLAS/Metrics.h
	include/shark/LinAlg/rotations.h
	include/shark/LinAlg/fft.h
)
SET( LINALG_SRCS
)
SOURCE_GROUP( LinAlg\\include FILES ${LINALG_HEADERS} )
SOURCE_GROUP( LinAlg\\src FILES ${LINALG_SRCS} )

#####################################################################
#           Models Module                                           #
#####################################################################
SET( MODELS_HEADERS
	include/shark/Models/ConcatenatedModel.h
	include/shark/Models/NearestNeighborRegression.h
	include/shark/Models/Softmax.h
	include/shark/Models/LinearNorm.h
	include/shark/Models/LinearClassifier.h
	include/shark/Models/RecurrentStructure.h
	include/shark/Models/CMAC.h
	include/shark/Models/AbstractModel.h
	include/shark/Models/SigmoidModel.h
	include/shark/Models/NBClassifier.h
	include/shark/Models/LinearModel.h
	include/shark/Models/Normalizer.h
	include/shark/Models/KalmanFilter.h
	include/shark/Models/OnlineRNNet.h
	include/shark/Models/OneVersusOneClassifier.h
	include/shark/Models/NearestNeighborClassifier.h
	include/shark/Models/FFNet.h
	include/shark/Models/Clustering/SoftClusteringModel.h
	include/shark/Models/Clustering/HierarchicalClustering.h
	include/shark/Models/Clustering/HardClusteringModel.h
	include/shark/Models/Clustering/AbstractClustering.h
	include/shark/Models/Clustering/ClusteringModel.h
	include/shark/Models/Clustering/Centroids.h
	include/shark/Models/Converter.h
	include/shark/Models/RNNet.h
	include/shark/Models/Kernels/ScaledKernel.h
	include/shark/Models/Kernels/MonomialKernel.h
	include/shark/Models/Kernels/ProductKernel.h
	include/shark/Models/Kernels/DiscreteKernel.h
	include/shark/Models/Kernels/MultiTaskKernel.h
	include/shark/Models/Kernels/KernelExpansion.h
	include/shark/Models/Kernels/NormalizedKernel.h
	include/shark/Models/Kernels/AbstractKernelFunction.h
	include/shark/Models/Kernels/PolynomialKernel.h
	include/shark/Models/Kernels/ArdKernel.h
	include/shark/Models/Kernels/GaussianRbfKernel.h
	include/shark/Models/Kernels/CSvmDerivative.h
	include/shark/Models/Kernels/MklKernel.h
	include/shark/Models/Kernels/LinearKernel.h
	include/shark/Models/Kernels/WeightedSumKernel.h
	include/shark/Models/Impl/Neurons.h
	include/shark/Models/Impl/LinearModel.inl
	include/shark/Models/RBFNet.h
	include/shark/Models/SoftNearestNeighborClassifier.h
	include/shark/Models/Trees/CARTClassifier.h
	include/shark/Models/Trees/BinaryTree.h
	include/shark/Models/Trees/RFClassifier.h
	include/shark/Models/Trees/KHCTree.h
	include/shark/Models/Trees/LCTree.h
	include/shark/Models/Trees/KDTree.h
	include/shark/Models/Neurons.h
)

SOURCE_GROUP( Models\\include FILES ${MODELS_HEADERS} )
SOURCE_GROUP( Models\\src FILES ${MODELS_SRCS} )

SET( MODELS_SRCS
	src/Models/CMAC.cpp
	src/Models/Converter.cpp
	src/Models/KalmanFilter.cpp
	src/Models/LinearClassifier.cpp
	src/Models/LinearNorm.cpp
	src/Models/OnlineRNNet.cpp
	src/Models/RNNet.cpp
	src/Models/RecurrentStructure.cpp
	src/Models/SigmoidModel.cpp
	src/Models/Softmax.cpp
	src/Models/Centroids.cpp
)

#####################################################################
#           Objective Functions Module                              #
#####################################################################
SET( OBJECTIVEFUNCTIONS_HEADERS
	include/shark/ObjectiveFunctions/AbstractObjectiveFunction.h
	include/shark/ObjectiveFunctions/AbstractMultiObjectiveFunction.h
	include/shark/ObjectiveFunctions/AbstractCost.h
	include/shark/ObjectiveFunctions/DataObjectiveFunction.h
	include/shark/ObjectiveFunctions/CombinedObjectiveFunction.h
	include/shark/ObjectiveFunctions/Regularizer.h
	include/shark/ObjectiveFunctions/ErrorFunction.h
	include/shark/ObjectiveFunctions/NoisyErrorFunction.h
	include/shark/ObjectiveFunctions/CrossValidationError.h
 	include/shark/ObjectiveFunctions/ROC.h
	include/shark/ObjectiveFunctions/RadiusMarginQuotient.h
	include/shark/ObjectiveFunctions/LooError.h
	include/shark/ObjectiveFunctions/LooErrorCSvm.h
	include/shark/ObjectiveFunctions/SpanBoundCSvm.h
	include/shark/ObjectiveFunctions/NegativeGaussianProcessEvidence.h
	include/shark/ObjectiveFunctions/NegativeAUC.h
	include/shark/ObjectiveFunctions/SvmLogisticInterpretation.h
	include/shark/ObjectiveFunctions/Loss/AbstractLoss.h
	include/shark/ObjectiveFunctions/Loss/SquaredLoss.h
	include/shark/ObjectiveFunctions/Loss/AbsoluteLoss.h
	include/shark/ObjectiveFunctions/Loss/CrossEntropy.h
	include/shark/ObjectiveFunctions/Loss/CrossEntropyIndependent.h
	include/shark/ObjectiveFunctions/Loss/NegativeClassificationLogLikelihood.h
	include/shark/ObjectiveFunctions/Loss/ZeroOneLoss.h
	include/shark/ObjectiveFunctions/Loss/DiscreteLoss.h
	include/shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h
	include/shark/ObjectiveFunctions/Benchmarks/Sphere.h
	include/shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h
	include/shark/ObjectiveFunctions/Benchmarks/Cigar.h
	include/shark/ObjectiveFunctions/Benchmarks/CigarDiscus.h
	include/shark/ObjectiveFunctions/Benchmarks/Discus.h
	include/shark/ObjectiveFunctions/Benchmarks/DiffPowers.h
	include/shark/ObjectiveFunctions/Benchmarks/CIGTAB1.h
	include/shark/ObjectiveFunctions/Benchmarks/CIGTAB2.h
	include/shark/ObjectiveFunctions/Benchmarks/DTLZ1.h
	include/shark/ObjectiveFunctions/Benchmarks/DTLZ2.h
	include/shark/ObjectiveFunctions/Benchmarks/DTLZ3.h
	include/shark/ObjectiveFunctions/Benchmarks/DTLZ4.h
	include/shark/ObjectiveFunctions/Benchmarks/DTLZ5.h
	include/shark/ObjectiveFunctions/Benchmarks/DTLZ6.h
	include/shark/ObjectiveFunctions/Benchmarks/DTLZ7.h
	include/shark/ObjectiveFunctions/Benchmarks/ELLI1.h
	include/shark/ObjectiveFunctions/Benchmarks/ELLI2.h
	include/shark/ObjectiveFunctions/Benchmarks/Fonseca.h
	include/shark/ObjectiveFunctions/Benchmarks/GSP.h
	include/shark/ObjectiveFunctions/Benchmarks/IHR1.h
	include/shark/ObjectiveFunctions/Benchmarks/IHR2.h
	include/shark/ObjectiveFunctions/Benchmarks/IHR3.h
	include/shark/ObjectiveFunctions/Benchmarks/IHR4.h
	include/shark/ObjectiveFunctions/Benchmarks/IHR6.h
	include/shark/ObjectiveFunctions/Benchmarks/LZ1.h
	include/shark/ObjectiveFunctions/Benchmarks/LZ2.h
	include/shark/ObjectiveFunctions/Benchmarks/LZ3.h
	include/shark/ObjectiveFunctions/Benchmarks/LZ4.h
	include/shark/ObjectiveFunctions/Benchmarks/LZ5.h
	include/shark/ObjectiveFunctions/Benchmarks/LZ6.h
	include/shark/ObjectiveFunctions/Benchmarks/LZ7.h
	include/shark/ObjectiveFunctions/Benchmarks/LZ8.h
	include/shark/ObjectiveFunctions/Benchmarks/LZ9.h
	include/shark/ObjectiveFunctions/Benchmarks/ZDT1.h
	include/shark/ObjectiveFunctions/Benchmarks/ZDT2.h
	include/shark/ObjectiveFunctions/Benchmarks/ZDT3.h
	include/shark/ObjectiveFunctions/Benchmarks/ZDT4.h
	include/shark/ObjectiveFunctions/Benchmarks/ZDT6.h

)

SET( OBJECTIVEFUNCTIONS_SRCS
	src/ObjectiveFunctions/ROC.cpp
	src/ObjectiveFunctions/DiscreteLoss.cpp
	src/ObjectiveFunctions/Loss/NegativeClassificationLogLikelihood.cpp
)

SOURCE_GROUP( ObjectiveFunctions\\include FILES ${OBJECTIVEFUNCTIONS_HEADERS} )
SOURCE_GROUP( ObjectiveFunctions\\src FILES ${OBJECTIVEFUNCTIONS_SRCS} )

#####################################################################
#           Algorithms Module                                       #
#####################################################################
SET( ALGORITHMS_HEADERS
	include/shark/Algorithms/AbstractOptimizer.h
	include/shark/Algorithms/AbstractMultiObjectiveOptimizer.h
	include/shark/Algorithms/AbstractSingleObjectiveOptimizer.h
 	include/shark/Algorithms/JaakkolaHeuristic.h
	include/shark/Algorithms/KMeans.h
	include/shark/Algorithms/GradientDescent/CG.h
	include/shark/Algorithms/GradientDescent/BFGS.h
	include/shark/Algorithms/GradientDescent/Rprop.h
	include/shark/Algorithms/GradientDescent/NoisyRprop.h
	include/shark/Algorithms/GradientDescent/SteepestDescent.h
	include/shark/Algorithms/GradientDescent/Quickprop.h
	include/shark/Algorithms/GradientDescent/LineSearch.h
	include/shark/Algorithms/Trainers/AbstractTrainer.h
	include/shark/Algorithms/Trainers/Distribution/DistTrainerContainer.h
	include/shark/Algorithms/Trainers/Distribution/GenericDistTrainer.h
	include/shark/Algorithms/Trainers/Distribution/NormalTrainer.h
	include/shark/Algorithms/Trainers/AbstractSvmTrainer.h
	include/shark/Algorithms/Trainers/CSvmTrainer.h
	include/shark/Algorithms/Trainers/EpsilonSvmTrainer.h
	include/shark/Algorithms/Trainers/OneClassSvmTrainer.h
	include/shark/Algorithms/Trainers/RegularizationNetworkTrainer.h
	include/shark/Algorithms/Trainers/McSvmOVATrainer.h
	include/shark/Algorithms/Trainers/McSvmMMRTrainer.h
	include/shark/Algorithms/Trainers/McSvmWWTrainer.h
	include/shark/Algorithms/Trainers/McSvmCSTrainer.h
	include/shark/Algorithms/Trainers/McSvmLLWTrainer.h
	include/shark/Algorithms/Trainers/McSvmATSTrainer.h
	include/shark/Algorithms/Trainers/McSvmATMTrainer.h
	include/shark/Algorithms/Trainers/McSvmADMTrainer.h
	include/shark/Algorithms/Trainers/LinearRegression.h
	include/shark/Algorithms/Trainers/LDA.h
	include/shark/Algorithms/Trainers/FisherLDA.h
	include/shark/Algorithms/Trainers/Perceptron.h
	include/shark/Algorithms/Trainers/KernelMeanClassifier.h
	include/shark/Algorithms/Trainers/NBClassifierTrainer.h
	include/shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h
	include/shark/Algorithms/Trainers/NormalizeComponentsUnitInterval.h
	include/shark/Algorithms/Trainers/NormalizeKernelUnitVariance.h
	include/shark/Algorithms/Trainers/PCA.h
	include/shark/Algorithms/Trainers/SigmoidFit.h
	include/shark/Algorithms/Trainers/RFTrainer.h
	include/shark/Algorithms/Trainers/CARTTrainer.h
	include/shark/Algorithms/LP/GLPK.h
	include/shark/Algorithms/QP/QuadraticProgram.h
	#include/shark/Algorithms/QP/OnlineQuadraticProgram.h
	include/shark/Algorithms/QP/QpSvmDecomp.h
	include/shark/Algorithms/QP/QpBoxDecomp.h
	include/shark/Algorithms/QP/QpMcDecomp.h
	#include/shark/Algorithms/QP/QpOnlineMcDecomp.h
	include/shark/Algorithms/QP/QpSparseArray.h
	include/shark/Algorithms/NearestNeighbors/AbstractNearestNeighbors.h
	include/shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h
	include/shark/Algorithms/NearestNeighbors/SimpleNearestNeighbors.h
	include/shark/Algorithms/DirectSearch/AGE.h
	include/shark/Algorithms/DirectSearch/AGE2.h
	include/shark/Algorithms/DirectSearch/BoundingBoxCalculator.h
	include/shark/Algorithms/DirectSearch/CMA/Chromosome.h
	include/shark/Algorithms/DirectSearch/CMA.h
	include/shark/Algorithms/DirectSearch/CMSA.h
	include/shark/Algorithms/DirectSearch/EA.h
	include/shark/Algorithms/DirectSearch/ElitistCMA.h
	include/shark/Algorithms/DirectSearch/Experiments/EvaluationCountStoppingCondition.h
	include/shark/Algorithms/DirectSearch/Experiments/Experiment.h
	include/shark/Algorithms/DirectSearch/Experiments/FrontStore.h
	include/shark/Algorithms/DirectSearch/FastNonDominatedSort.h
	include/shark/Algorithms/DirectSearch/FitnessComparator.h
	include/shark/Algorithms/DirectSearch/FitnessExtractor.h
	include/shark/Algorithms/DirectSearch/Grid.h
	include/shark/Algorithms/DirectSearch/GridSearch.h
	include/shark/Algorithms/DirectSearch/HypervolumeApproximator.h
	include/shark/Algorithms/DirectSearch/HypervolumeCalculator.h
	include/shark/Algorithms/DirectSearch/HypervolumeIndicator.h
	include/shark/Algorithms/DirectSearch/Indicators/AdditiveEpsilonIndicator.h
	include/shark/Algorithms/DirectSearch/Indicators/InvertedGenerationalDistance.h
	include/shark/Algorithms/DirectSearch/Indicators/MultiplicativeEpsilonIndicator.h
	include/shark/Algorithms/DirectSearch/InterruptibleAlgorithmRunner.h
	include/shark/Algorithms/DirectSearch/MOCMA.h
	include/shark/Algorithms/DirectSearch/OnePlusOneES.h
	include/shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h
	include/shark/Algorithms/DirectSearch/Operators/Initializers/CMA/Initializer.h
	include/shark/Algorithms/DirectSearch/Operators/Initializers/CovarianceMatrixInitializer.h
	include/shark/Algorithms/DirectSearch/Operators/Mutation/CMA/Mutator.h
	include/shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h
	include/shark/Algorithms/DirectSearch/Operators/ParameterUpdate/CMA/Updater.h
	include/shark/Algorithms/DirectSearch/Operators/Recombination/GlobalIntermediateRecombination.h
	include/shark/Algorithms/DirectSearch/Operators/Recombination/OnePointCrossover.h
	include/shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h
	include/shark/Algorithms/DirectSearch/Operators/Recombination/UniformCrossover.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/ApproximatedHypervolumeSelection.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/BinaryTournamentSelection.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/EPTournamentSelection.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/LinearRanking.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/MuKommaLambdaSelection.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/MuPlusLambdaSelection.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/RouletteWheelSelection.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/SteadyStateIndicatorBasedSelection.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h
	include/shark/Algorithms/DirectSearch/Operators/Selection/UniformRanking.h
	include/shark/Algorithms/DirectSearch/PAES.h
	include/shark/Algorithms/DirectSearch/ParetoDominanceComparator.h
	include/shark/Algorithms/DirectSearch/Population.h
	include/shark/Algorithms/DirectSearch/RankShareComparator.h
	include/shark/Algorithms/DirectSearch/RealCodedNSGAII.h
	include/shark/Algorithms/DirectSearch/SMS-EMOA.h
	include/shark/Algorithms/DirectSearch/SteadyStateMOCMA.h
	include/shark/Algorithms/DirectSearch/Traits/FitnessTraits.h
	include/shark/Algorithms/DirectSearch/Traits/QualityIndicatorTraits.h
	include/shark/Algorithms/DirectSearch/TypedIndividual.h
	include/shark/Algorithms/DirectSearch/XNES.h
)

SET( ALGORITHMS_SRCS
	src/Algorithms/FisherLDA.cpp
	src/Algorithms/GLPK.cpp
	src/Algorithms/KMeans.cpp
	src/Algorithms/LDA.cpp
	src/Algorithms/LinearRegression.cpp
	src/Algorithms/NoisyRprop.cpp
	src/Algorithms/PCA.cpp
	src/Algorithms/SigmoidFit.cpp
	src/Algorithms/RFTrainer.cpp
	src/Algorithms/CARTTrainer.cpp
	src/Algorithms/GradientDescent/Rprop.cpp
	src/Algorithms/GradientDescent/Quickprop.cpp
	src/Algorithms/GradientDescent/IRLS.cpp
	src/Algorithms/GradientDescent/CG.cpp
	src/Algorithms/GradientDescent/BFGS.cpp
	src/Algorithms/DirectSearch/CMA.cpp
	src/Algorithms/DirectSearch/CMSA.cpp
)

SOURCE_GROUP( Algorithms\\include FILES ${ALGORITHMS_HEADERS} )
SOURCE_GROUP( Algorithms\\src FILES ${ALGORITHMS_SRCS} )

#####################################################################
#           Data Handling Module                                    #
#####################################################################
SET( DATA_HEADERS
 	include/shark/Data/Csv.h
 	include/shark/Data/Libsvm.h
 	include/shark/Data/PrecomputedMatrix.h
	include/shark/Data/CVDatasetTools.h
	include/shark/Data/Dataset.h
	include/shark/Data/DataDistribution.h
	include/shark/Data/HDF5.h
)

SET( DATA_SRCS
)

SOURCE_GROUP( Data\\include FILES ${DATA_HEADERS} )
SOURCE_GROUP( Data\\src FILES ${DATA_SRCS} )

#####################################################################
#           Core Module                                             #
#####################################################################
SET( CORE_HEADERS
	include/shark/Core/utility/ScopedHandle.h

	include/shark/Core/AbstractBoxConstraintsProvider.h
	include/shark/Core/AbstractFeasibleRegion.h
	
	include/shark/Core/Traits/MultiObjectiveFunctionTraits.h
	include/shark/Core/Traits/ObjectiveFunctionTraits.h
	include/shark/Core/Traits/OptimizerTraits.h
	
	include/shark/Core/Derivative.h
	include/shark/Core/Exception.h
	
	include/shark/Core/Flags.h
	
	include/shark/Core/IConfigurable.h
	include/shark/Core/INameable.h
	include/shark/Core/IParameterizable.h
	include/shark/Core/ISerializable.h
	
	include/shark/Core/LogFormatters/PrintfLogFormatter.h
	include/shark/Core/Logger.h
	include/shark/Core/LogHandlers/StreamHandler.h
	
	include/shark/Core/Math.h
	
	include/shark/Core/Factory.h
	include/shark/Core/Probe.h
	
	include/shark/Core/Chart.h
	include/shark/Core/Renderers/HighchartRenderer.h
	
	include/shark/Core/SearchSpaces/VectorSpace.h
	
	include/shark/Core/Shark.h
	include/shark/Core/SignalTrap.h
	include/shark/Core/Singleton.h
	include/shark/Core/Timer.h
	
	include/shark/Core/ResultSets.h
)

SET( CORE_SRCS )

# SOURCE_GROUP( ML\\include FILES ${CORE_HEADERS} )

#####################################################################
#           Fuzzy Module                                            #
#####################################################################
SET( FUZZY_HEADERS
include/shark/Fuzzy/SugenoRule.h
include/shark/Fuzzy/CustomIM.h
include/shark/Fuzzy/FCL/FuzzyControlLanguageParser.h
include/shark/Fuzzy/FCL/FuzzyControlLanguageParserFusionBindings.h
include/shark/Fuzzy/FuzzyRelation.h
include/shark/Fuzzy/FuzzySet.h
include/shark/Fuzzy/Implication.h
include/shark/Fuzzy/InferenceMachine.h
include/shark/Fuzzy/LinguisticTerm.h
include/shark/Fuzzy/LinguisticVariable.h
include/shark/Fuzzy/MamdaniIM.h
include/shark/Fuzzy/Operators.h
include/shark/Fuzzy/Rule.h
include/shark/Fuzzy/RuleBase.h
include/shark/Fuzzy/RuleBaseGenerator.h
include/shark/Fuzzy/SugenoIM.h

include/shark/Fuzzy/LinguisticTerms/BellLT.h
include/shark/Fuzzy/LinguisticTerms/ComposedLT.h
include/shark/Fuzzy/LinguisticTerms/ConstantLT.h
include/shark/Fuzzy/LinguisticTerms/CustomizedLT.h
include/shark/Fuzzy/LinguisticTerms/GeneralizedBellLT.h
include/shark/Fuzzy/LinguisticTerms/InfinityLT.h
include/shark/Fuzzy/LinguisticTerms/SigmoidalLT.h
include/shark/Fuzzy/LinguisticTerms/SingletonLT.h
include/shark/Fuzzy/LinguisticTerms/TrapezoidLT.h
include/shark/Fuzzy/LinguisticTerms/TriangularLT.h

include/shark/Fuzzy/FuzzySets/TriangularFS.h
include/shark/Fuzzy/FuzzySets/BellFS.h
include/shark/Fuzzy/FuzzySets/ComposedFS.h
include/shark/Fuzzy/FuzzySets/ComposedNDimFS.h
include/shark/Fuzzy/FuzzySets/ConstantFS.h
include/shark/Fuzzy/FuzzySets/CustomizedFS.h
include/shark/Fuzzy/FuzzySets/GeneralizedBellFS.h
include/shark/Fuzzy/FuzzySets/HomogenousNDimFS.h
include/shark/Fuzzy/FuzzySets/InfinityFS.h
include/shark/Fuzzy/FuzzySets/NDimFS.h
include/shark/Fuzzy/FuzzySets/SigmoidalFS.h
include/shark/Fuzzy/FuzzySets/SingletonFS.h
include/shark/Fuzzy/FuzzySets/TrapezoidFS.h
)



SET( FUZZY_SRCS
	src/Fuzzy/BellFS.cpp
	src/Fuzzy/BellLT.cpp
	src/Fuzzy/ComposedFS.cpp
	src/Fuzzy/ComposedLT.cpp
	src/Fuzzy/ComposedNDimFS.cpp
	src/Fuzzy/ConstantLT.cpp
	src/Fuzzy/CustomIM.cpp
	src/Fuzzy/CustomizedLT.cpp
	src/Fuzzy/FuzzyException.cpp
	src/Fuzzy/FuzzySet.cpp
	src/Fuzzy/GeneralizedBellFS.cpp
	src/Fuzzy/GeneralizedBellLT.cpp
	src/Fuzzy/HomogenousNDimFS.cpp
	src/Fuzzy/Implication.cpp
	src/Fuzzy/InferenceMachine.cpp
	src/Fuzzy/InfinityFS.cpp
	src/Fuzzy/InfinityLT.cpp
	src/Fuzzy/LinguisticTerm.cpp
	src/Fuzzy/LinguisticVariable.cpp
	src/Fuzzy/MamdaniIM.cpp
	src/Fuzzy/NDimFS.cpp
	src/Fuzzy/Operators.cpp
	src/Fuzzy/RCObject.cpp
	src/Fuzzy/Rule.cpp
	src/Fuzzy/RuleBase.cpp
	src/Fuzzy/RuleBaseGenerator.cpp
	src/Fuzzy/SigmoidalFS.cpp
	src/Fuzzy/SigmoidalLT.cpp
	src/Fuzzy/SingletonFS.cpp
	src/Fuzzy/SingletonLT.cpp
	src/Fuzzy/SugenoIM.cpp
	src/Fuzzy/SugenoRule.cpp
	src/Fuzzy/TrapezoidFS.cpp
	src/Fuzzy/TrapezoidLT.cpp
	src/Fuzzy/TriangularFS.cpp
	src/Fuzzy/TriangularLT.cpp
	src/Fuzzy/xmlParser.cpp
)

SOURCE_GROUP( Fuzzy\\include FILES ${FUZZY_HEADERS} )
SOURCE_GROUP( Fuzzy\\src FILES ${FUZZY_SRCS} )

#####################################################################
#           Network Module                                          #
#####################################################################

IF( OPT_ENABLE_NETWORKING )
    SET( NETWORK_HEADERS
    	 include/shark/Network/Handlers/AboutHandler.h
	 include/shark/Network/Handlers/FileHandler.h
	 include/shark/Network/Handlers/LoggerPoolRestHandler.h
	 include/shark/Network/Handlers/RestHandler.h
	 include/shark/Network/HttpServer.h
    )
ENDIF()

#####################################################################
#           RBM Module                                              #
#####################################################################
SET( RBM_HEADERS
	include/shark/Unsupervised/RBM/Impl/analytics.h
	include/shark/Unsupervised/RBM/Impl/EnergyInteractionTerm.h
	include/shark/Unsupervised/RBM/Impl/AverageEnergyGradient1x1.h
	include/shark/Unsupervised/RBM/Impl/RBMStructure.h
	include/shark/Unsupervised/RBM/BinaryRBM.h
	include/shark/Unsupervised/RBM/TruncExpBinaryRBM.h
	include/shark/Unsupervised/RBM/Problems/DistantModes.h
	include/shark/Unsupervised/RBM/Problems/MNIST.h
	include/shark/Unsupervised/RBM/Problems/BarsAndStripes.h
	include/shark/Unsupervised/RBM/Problems/Shifter.h
	include/shark/Unsupervised/RBM/Neuronlayers/TruncatedExponentialLayer.h
	include/shark/Unsupervised/RBM/Neuronlayers/GaussianLayer.h
	include/shark/Unsupervised/RBM/Neuronlayers/BinaryLayer.h
	include/shark/Unsupervised/RBM/StateSpaces/TwoStateSpace.h
	include/shark/Unsupervised/RBM/StateSpaces/RealSpace.h
	include/shark/Unsupervised/RBM/GaussianBinaryRBM.h
	include/shark/Unsupervised/RBM/Tags.h
	include/shark/Unsupervised/RBM/Energy.h
	include/shark/Unsupervised/RBM/GradientApproximations/ContrastiveDivergence.h
	include/shark/Unsupervised/RBM/GradientApproximations/MultiChainApproximator.h
	include/shark/Unsupervised/RBM/GradientApproximations/ExactGradient.h
	include/shark/Unsupervised/RBM/GradientApproximations/Impl/DataEvaluator.h
	include/shark/Unsupervised/RBM/GradientApproximations/SingleChainApproximator.h
	include/shark/Unsupervised/RBM/RBM.h
	include/shark/Unsupervised/RBM/Sampling/MarkovChain.h
	include/shark/Unsupervised/RBM/Sampling/Impl/SampleTypes.h
	include/shark/Unsupervised/RBM/Sampling/TemperedMarkovChain.h
	include/shark/Unsupervised/RBM/Sampling/GibbsOperator.h
	include/shark/Unsupervised/RBM/analytics.h

)

SET( RBM_SRCS
)

SOURCE_GROUP( RBM\\include FILES ${RBM_HEADERS} )
SOURCE_GROUP( RBM\\src FILES ${RBM_SRCS} )

SET( SRCS
   ${CORE_SRCS}
    ${CORE_HEADERS}
    ${RNG_SRCS}
    ${RNG_HEADERS}
    ${LINALG_SRCS}
    ${LINALG_HEADERS}
    ${FUZZY_HEADERS}
    ${MODELS_SRCS}
    ${MODELS_HEADERS}
    ${OBJECTIVEFUNCTIONS_SRCS}
    ${OBJECTIVEFUNCTIONS_HEADERS}
    ${ALGORITHMS_SRCS}
    ${ALGORITHMS_HEADERS}
    ${DATA_SRCS}
    ${DATA_HEADERS}
    ${NETWORK_HEADERS}
    ${RBM_HEADERS}
    ${RBM_SRCS}
)

SET( PUBLIC_HEADERS
    ../SharkDefs.h
    ${CORE_HEADERS}
    ${RNG_HEADERS}
    ${LINALG_HEADERS}
#    ${FUZZY_HEADERS}
    ${MODELS_HEADERS}
    ${OBJECTIVEFUNCTIONS_HEADERS}
    ${ALGORITHMS_HEADERS}
    ${DATA_HEADERS}
    ${NETWORK_HEADERS}
    ${RBM_HEADERS}
)



#####################################################################
#                       Library setup                               #
#####################################################################

IF( WIN32 )
    ADD_LIBRARY( shark ${SRCS} )
    SET_TARGET_PROPERTIES( shark PROPERTIES DEBUG_PREFIX ../ )
    SET_TARGET_PROPERTIES( shark PROPERTIES DEBUG_POSTFIX _debug )
ELSE( WIN32 )
      ADD_LIBRARY( shark ${LIBRARY_FLAGS} ${SRCS} )
      SET_TARGET_PROPERTIES( shark PROPERTIES DEBUG_POSTFIX _debug )
ENDIF( WIN32 )

TARGET_LINK_LIBRARIES( shark ${LINK_LIBRARIES} )

ADD_EXECUTABLE( Version src/Core/Version.cpp )
TARGET_LINK_LIBRARIES( Version ${LINK_LIBRARIES} )

#TARGET_LINK_LIBRARIES( Statistics shark ${LINK_LIBRARIES} )


#####################################################################
#                       Static Code Analysis                        #
#####################################################################
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
#                       Include Examples                            #
#####################################################################
OPTION( OPT_COMPILE_EXAMPLES "Compile example programs." ON )

IF( OPT_COMPILE_EXAMPLES )
	ADD_SUBDIRECTORY( examples )
ENDIF( OPT_COMPILE_EXAMPLES )

#####################################################################
#                       Include Unit Tests                          #
#####################################################################
IF( OPT_DYNAMIC_LIBRARY )
    OPTION( OPT_MAKE_TESTS "Test the library after Building." OFF )
ELSE( OPT_DYNAMIC_LIBRARY )
    OPTION( OPT_MAKE_TESTS "Test the library after Building." ON )
ENDIF( OPT_DYNAMIC_LIBRARY )

IF( OPT_MAKE_TESTS)
	ADD_SUBDIRECTORY( Test )
ENDIF( OPT_MAKE_TESTS )

#####################################################################
#                       Include Documentation                       #
#####################################################################
OPTION( OPT_COMPILE_DOCUMENTATION "Compile documentation." OFF )

IF( OPT_COMPILE_DOCUMENTATION )
	ADD_SUBDIRECTORY( doc )
ENDIF( OPT_COMPILE_DOCUMENTATION )

#####################################################################
#                       Install the library                         #
#####################################################################

INSTALL( TARGETS shark
	 RUNTIME DESTINATION bin
	 LIBRARY DESTINATION lib
	 ARCHIVE DESTINATION lib
)

#####################################################################
#                  Install the version executable                   #
#####################################################################
INSTALL(
	TARGETS Version
	RUNTIME DESTINATION bin
)

#####################################################################
#                  Adjust library version for debian                #
#####################################################################
SET_TARGET_PROPERTIES( shark PROPERTIES
	VERSION 3.0.0
	SOVERSION 3.0.0
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Algorithms
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Core
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Data
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Fuzzy
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/LinAlg
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Models
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Network
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/ObjectiveFunctions
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Rng
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Statistics
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Unsupervised
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( FILES
include/shark/SharkDefs.h
DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
)

IF( OPT_ENABLE_NETWORKING )

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/include/shark/Network
	 DESTINATION DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
	 PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY
	 ${CMAKE_CURRENT_SOURCE_DIR}/contrib/cpp-netlib-0.9.0
	 DESTINATION ${SHARK_INSTALL_CONTRIB_DIR}
	 PATTERN ".*" EXCLUDE
)
ENDIF( OPT_ENABLE_NETWORKING )
#####################################################################
#                Adjust compiler flags and settings								#
#####################################################################
IF( WIN32 )
	SET( DISABLE_WARNINGS "/wd4250 /wd4251 /wd4275 /wd4800 /wd4308" )
	#4250: inherit via dominance
	#4251: needs to have dll-interface
	#4275: non-dll interface used as base for dll-interface class
	#4308: Negative integral constant
	SET( ADDITIONAL_COMPILER_FLAGS "${DISABLE_WARNINGS}" )

	IF( OPENMP_FOUND )
		MESSAGE( "OpenMP found" )
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp" )
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp" )
	ENDIF( OPENMP_FOUND )

	#SET( ADDITIONAL_LINKER_FLAGS "/DEBUG /LARGEADDRESSAWARE" )

	#disable optimization for RelWithDebInfo target to allow debugging
	STRING( REPLACE "/O2" "/Od" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" )

	SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${ADDITIONAL_COMPILER_FLAGS}" )
	SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ADDITIONAL_COMPILER_FLAGS}" )
	SET( CMAKE_CXX_FLAGS_MINSIZERELEASE "${CMAKE_CXX_FLAGS_MINSIZERELEASE} ${ADDITIONAL_COMPILER_FLAGS}" CACHE STRING "compiler-flags" FORCE )
	SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${ADDITIONAL_COMPILER_FLAGS}" CACHE STRING "compiler-flags" FORCE )

	SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "compiler-flags" FORCE )
	SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "compiler-flags" FORCE )
	SET( CMAKE_CXX_FLAGS_MINSIZERELEASE "${CMAKE_CXX_FLAGS_MINSIZERELEASE}" CACHE STRING "compiler-flags" FORCE )
	SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "compiler-flags" FORCE )

	ADD_DEFINITIONS( -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1)

ENDIF( WIN32 )

IF( UNIX )
	IF( APPLE )
	  	SET(CMAKE_CXX_CREATE_STATIC_LIBRARY "libtool -static -o <TARGET> <LINK_FLAGS> <OBJECTS>" )
	ENDIF( APPLE )
ENDIF( UNIX )
