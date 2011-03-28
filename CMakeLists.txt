PROJECT( shark )

CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

FIND_PACKAGE( OpenMP QUIET )

#CMAKE_POLICY( SET CMP0003 OLD )

ENABLE_TESTING()

SET( CPACK_PACKAGE_NAME "libshark-i386" )
SET( CPACK_BUNDLE_NAME "libshark" )
SET( CPACK_PACKAGE_VENDOR "Institut fur Neuroinformatik, Ruhr-Universität Bochum" )
SET( CPACK_PACKAGE_VERSION_MAJOR 2 )
SET( CPACK_PACKAGE_VERSION_MINOR 3 )
SET( CPACK_PACKAGE_VERSION_PATCH 4 )
SET( CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/gpl-2.0.txt )

SET( CPACK_DEBIAN_PACKAGE_NAME libshark-i386 )
SET( CPACK_DEBIAN_PACKAGE_VERSION 2.3.4 )
SET( CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386 )
SET( CPACK_DEBIAN_PACKAGE_DESCRIPTION "SHARK is a modular C++ library for the design and optimization of adaptive systems. It provides methods for linear and nonlinear optimization, in particular evolutionary and gradient-based algorithms, kernel-based learning algorithms and neural networks, and various other machine learning techniques. SHARK serves as a toolbox to support real world applications as well as research in different domains of computational intelligence and machine learning. The sources are compatible with the following platforms: Windows, Solaris, MacOS X, and Linux." )
# SET( CPACK_DEBIAN_PACKAGE_DEPENDS ... )
# SET( DEBIAN_PACKAGE_BUILDS_DEPENDS ... )
SET( CPACK_DEBIAN_PACKAGE_MAINTAINER "c.igel@ieee.org" )

IF( UNIX )
  SET( SHARK_INSTALL_INCLUDE_DIR include/ )
  SET( SHARK_INSTALL_LIB_DIR lib/shark/ )
  SET( SHARK_INSTALL_EXAMPLE_DIR share/shark/examples/ )
  SET( SHARK_INSTALL_DOC_DIR share/shark/doc/ )
ELSE( UNIX )
  SET( SHARK_INSTALL_INCLUDE_DIR include/ )
  SET( SHARK_INSTALL_LIB_DIR lib/ )
  SET( SHARK_INSTALL_EXAMPLE_DIR examples/ )
  SET( SHARK_INSTALL_DOC_DIR doc/ )
ENDIF( UNIX )

INCLUDE( CPack )

ADD_DEFINITIONS( -D_CRT_SECURE_NO_WARNINGS )

INCLUDE_DIRECTORIES( include )

SET( ARRAY_HEADERS 
  include/Array/Array.h
  include/Array/Array2D.h
  include/Array/ArrayIo.h
  include/Array/ArrayOp.h
  include/Array/ArraySort.h
  include/Array/ArrayTable.h
  )

SET( ARRAY_SRCS
  src/Array/ReadArray.cpp
  )

SOURCE_GROUP( Array\\include FILES ${ARRAY_HEADERS} )
SOURCE_GROUP( Array\\src FILES ${ARRAY_SRCS} )

SET( RNG_HEADERS
  include/Rng/Bernoulli.h
  include/Rng/Binomial.h
  include/Rng/Cauchy.h
  include/Rng/DiffGeometric.h
  include/Rng/Dirichlet.h
  include/Rng/DiscreteUniform.h
  include/Rng/Erlang.h
  include/Rng/Gamma.h
  include/Rng/Geometric.h
  include/Rng/GlobalRng.h
  include/Rng/HyperGeometric.h
  include/Rng/LogNormal.h
  include/Rng/NegExponential.h
  include/Rng/Normal.h
  include/Rng/Poisson.h
  include/Rng/RNG.h
  include/Rng/RandomVar.h
  include/Rng/Uniform.h
  include/Rng/Weibull.h
  )

SET( RNG_SRCS 
  src/Rng/Bernoulli.cpp
  src/Rng/Binomial.cpp
  src/Rng/Cauchy.cpp
  src/Rng/DiffGeometric.cpp
  src/Rng/Dirichlet.cpp
  src/Rng/DiscreteUniform.cpp
  src/Rng/Erlang.cpp
  src/Rng/Gamma.cpp
  src/Rng/Geometric.cpp
  src/Rng/GlobalRng.cpp
  src/Rng/HyperGeometric.cpp
  src/Rng/LogNormal.cpp
  src/Rng/NegExponential.cpp
  src/Rng/Normal.cpp
  src/Rng/Poisson.cpp
  src/Rng/RNG.cpp
  src/Rng/Uniform.cpp
  src/Rng/Weibull.cpp

  )

SOURCE_GROUP( RNG\\include FILES ${RNG_HEADERS} )
SOURCE_GROUP( RNG\\src FILES ${RNG_SRCS} )

SET( LINALG_HEADERS
  include/LinAlg/LinAlg.h
  include/LinAlg/VecMat.h
  include/LinAlg/arrayoptimize.h
  include/LinAlg/fft.h
  )

SET( LINALG_SRCS 
  src/LinAlg/LinAlg.cpp
  src/LinAlg/VecMat.cpp
  src/LinAlg/cblnsrch.cpp
  src/LinAlg/detsymm.cpp
  src/LinAlg/dlinmin.cpp
  src/LinAlg/eigen.cpp
  src/LinAlg/eigenerr.cpp
  src/LinAlg/eigensort.cpp
  src/LinAlg/eigensymm.cpp
  src/LinAlg/eigensymmJacobi.cpp
  src/LinAlg/eigensymmJacobi2.cpp
  src/LinAlg/fft.cpp
  src/LinAlg/g_inverse.cpp
  src/LinAlg/invert.cpp
  src/LinAlg/linmin.cpp
  src/LinAlg/lnsrch.cpp
  src/LinAlg/rank.cpp
  src/LinAlg/rankDecomp.cpp
  src/LinAlg/svd.cpp
  src/LinAlg/svdrank.cpp
  src/LinAlg/svdsort.cpp
  )

SOURCE_GROUP( LinAlg\\include FILES ${LINALG_HEADERS} )
SOURCE_GROUP( LinAlg\\src FILES ${LINALG_SRCS} )

SET( FILEUTIL_HEADERS
  include/FileUtil/FileUtil.h
  include/FileUtil/IOTools.h
  include/FileUtil/Params.h
  )

SET( FILEUTIL_SRCS
  src/FileUtil/FileUtil.cpp
  src/FileUtil/Params.cpp
  )

SOURCE_GROUP( FileUtil\\include FILES ${FILEUTIL_HEADERS} )
SOURCE_GROUP( FileUtil\\src FILES ${FILEUTIL_SRCS} )

SET( EALIB_HEADERS
  include/EALib/CMA.h
  include/EALib/CMSA.h
  include/EALib/Chromosome.h
  include/EALib/ChromosomeCMA.h
  include/EALib/ChromosomeCMACholesky.h
  include/EALib/ChromosomeFactory.h
  include/EALib/ChromosomeT.h
  include/EALib/ElitistCMA.h
  include/EALib/Individual.h
  include/EALib/IndividualT.h
  include/EALib/Interval.h
  include/EALib/MultiObjectiveFunctions.h
  include/EALib/ObjectiveFunction.h
  include/EALib/ObjectiveFunctions.h
  include/EALib/OnePlusOneES.h
  include/EALib/PVMinterface.h
  include/EALib/Population.h
  include/EALib/PopulationT.h
  include/EALib/SearchAlgorithm.h
  include/EALib/UncertaintyQuantification.h
  )

SET( EALIB_SRCS
  src/EALib/CMA.cpp
  src/EALib/CMSA.cpp
  src/EALib/Chromosome.cpp
  src/EALib/ChromosomeFactory.cpp
  src/EALib/ChromosomeT_bool.cpp
  src/EALib/ChromosomeT_char.cpp
  src/EALib/ChromosomeT_double.cpp
  src/EALib/ChromosomeT_int.cpp
  src/EALib/ElitistCMA.cpp
  src/EALib/Individual.cpp
  src/EALib/MultiObjectiveFunctions.cpp
  src/EALib/ObjectiveFunction.cpp
  src/EALib/ObjectiveFunctions.cpp
  src/EALib/OnePlusOneES.cpp
  src/EALib/PVMinterface.cpp
  src/EALib/Population.cpp
  src/EALib/UncertaintyQuantification.cpp
  )

SOURCE_GROUP( EALib\\include FILES ${EALIB_HEADERS} )
SOURCE_GROUP( EALib\\src FILES ${EALIB_SRCS} )

SET( MOOEALIB_HEADERS
  include/MOO-EALib/ArchiveMOO.h
  include/MOO-EALib/Hypervolume.h
  include/MOO-EALib/IndividualMOO.h
  include/MOO-EALib/MO-CMA.h
  include/MOO-EALib/MOOMeasures.h
  include/MOO-EALib/MOOTestFunctions.h
  include/MOO-EALib/NSGA2.h
  include/MOO-EALib/PopulationMOO.h
  include/MOO-EALib/README
  include/MOO-EALib/RMMEDA.h
  include/MOO-EALib/SelectionMOO-Impl.h
  include/MOO-EALib/SelectionMOO.h
  include/MOO-EALib/TestFunction.h
  )

SET( MOOEALIB_SRCS
  src/MOO-EALib/ArchiveMOO.cpp
  src/MOO-EALib/Hypervolume.cpp
  src/MOO-EALib/IndividualMOO.cpp
  src/MOO-EALib/MO-CMA.cpp
  src/MOO-EALib/NSGA2.cpp
  src/MOO-EALib/PopulationMOO.cpp
  src/MOO-EALib/RMMEDA.cpp
  src/MOO-EALib/SelectionMOO.cpp
  )

SOURCE_GROUP( MOO-EALib\\include FILES ${MOOEALIB_HEADERS} )
SOURCE_GROUP( MOO-EALib\\src FILES ${MOOEALIB_SRCS} )

SET( RECLAM_HEADERS
  include/ReClaM/AdpBP.h
  include/ReClaM/ArtificialDistributions.h
  include/ReClaM/BFGS.h
  include/ReClaM/BinaryCriterion.h
  include/ReClaM/CG.h
  include/ReClaM/CMAOptimizer.h
  include/ReClaM/ClassificationError.h
  include/ReClaM/CMAC.h
  include/ReClaM/ComponentWiseModel.h
  include/ReClaM/ConcatenatedModel.h
  include/ReClaM/CoverTree.h
  include/ReClaM/CrossEntropy.h
  include/ReClaM/CrossEntropyIndependent.h
  include/ReClaM/CrossValidation.h
  include/ReClaM/DF_CrossEntropy.h
  include/ReClaM/DF_CrossEntropyIndependent.h
  include/ReClaM/DF_MeanSquaredError.h
  include/ReClaM/Dataset.h
  include/ReClaM/EarlyStopping.h
  include/ReClaM/ErrorFunction.h
  include/ReClaM/ErrorPercentage.h
  include/ReClaM/FFNet.h
  include/ReClaM/FFNetSource.h
  include/ReClaM/GaussKernel.h
  include/ReClaM/GaussianProcess.h
  include/ReClaM/GridSearch.h
  include/ReClaM/InverseClassSeparability.h
  include/ReClaM/JaakkolaHeuristic.h
  include/ReClaM/KTA.h
  include/ReClaM/KalmanFilter.h
  include/ReClaM/KernelFunction.h
  include/ReClaM/KernelKMeans.h
  include/ReClaM/KernelMeanClassifier.h
  include/ReClaM/KernelNearestNeighbor.h
  include/ReClaM/KernelNearestNeighborRegression.h
  include/ReClaM/LDA.h
  include/ReClaM/LMSEFFNet.h
  include/ReClaM/LOO.h
  include/ReClaM/LinOutFFNet.h
  include/ReClaM/LinOutMSEBFFNet.h
  include/ReClaM/LinearEquation.h
  include/ReClaM/LinearModel.h
  include/ReClaM/LinearRegression.h
  include/ReClaM/MSEFFNet.h
  include/ReClaM/MSERBFNet.h
  include/ReClaM/MSERNNet.h
  include/ReClaM/McSvmApproximation.h
  include/ReClaM/MeanSquaredError.h
  include/ReClaM/Model.h
  include/ReClaM/ModelInterface.h
  include/ReClaM/ModelWithErrorFunction.h
  include/ReClaM/NegativeLogLikelihood.h
  include/ReClaM/NegativePolarization.h
  include/ReClaM/NetParams.h
  include/ReClaM/NoisyRprop.h
  include/ReClaM/NoisySvmLikelihood.h
  include/ReClaM/NormedModels.h
  include/ReClaM/Optimizer.h
  include/ReClaM/PCA.h
  include/ReClaM/FisherLDA.h
  include/ReClaM/Paraboloid.h
  include/ReClaM/Perceptron.h
  include/ReClaM/ProbenBNet.h
  include/ReClaM/ProbenNet.h
  include/ReClaM/QuadraticProgram.h
  include/ReClaM/KernelMatrices.h
  include/ReClaM/Utilities.h
  include/ReClaM/QpSvmDecomp.h
  include/ReClaM/QpBoxDecomp.h
  include/ReClaM/QpMcDecomp.h
  include/ReClaM/QpMcStzDecomp.h
  include/ReClaM/QpEbCsDecomp.h
  include/ReClaM/Quickprop.h
  include/ReClaM/RBFNet.h
  include/ReClaM/ROC.h
  include/ReClaM/RadiusMargin.h
  include/ReClaM/Rprop.h
  include/ReClaM/SigmoidFit.h
  include/ReClaM/SigmoidModel.h
  # include/ReClaM/SpanBound.h
  include/ReClaM/Softmax.h
  include/ReClaM/SpanBound1.h
  include/ReClaM/SpanBound2.h
  include/ReClaM/SpanBoundA.h
  include/ReClaM/SquaredError.h
  include/ReClaM/SteepestDescent.h
  include/ReClaM/StochasticGradientDescent.h
  include/ReClaM/Svm.h
  include/ReClaM/SvmApproximation.h
  include/ReClaM/TanhNet.h
  include/ReClaM/ValidationError.h
  include/ReClaM/Variance.h
  include/ReClaM/VarianceEstimator.h
  include/ReClaM/WTA.h
  include/ReClaM/createConnectionMatrix.h
  )

SET( RECLAM_SRCS
  src/ReClaM/ArtificialDistributions.cpp
  src/ReClaM/BFGS.cpp
  src/ReClaM/CG.cpp
  src/ReClaM/CMAOptimizer.cpp
  src/ReClaM/ClassificationError.cpp
  src/ReClaM/CMAC.cpp
  src/ReClaM/ComponentWiseModel.cpp
  src/ReClaM/ConcatenatedModel.cpp
  src/ReClaM/CoverTree.cpp
  src/ReClaM/CrossValidation.cpp
  src/ReClaM/Dataset.cpp
  src/ReClaM/EarlyStopping.cpp
  src/ReClaM/ErrorFunction.cpp
  src/ReClaM/FFNet.cpp
  src/ReClaM/GaussKernel.cpp
  src/ReClaM/GaussianProcess.cpp
  src/ReClaM/GridSearch.cpp
  src/ReClaM/InverseClassSeparability.cpp
  src/ReClaM/JaakkolaHeuristic.cpp
  src/ReClaM/KTA.cpp
  src/ReClaM/KalmanFilter.cpp
  src/ReClaM/KernelFunction.cpp
  src/ReClaM/KernelKMeans.cpp
  src/ReClaM/KernelMeanClassifier.cpp
  src/ReClaM/KernelNearestNeighbor.cpp
  src/ReClaM/LDA.cpp
  src/ReClaM/LOO.cpp
  src/ReClaM/LinearEquation.cpp
  src/ReClaM/LinearModel.cpp
  src/ReClaM/LinearRegression.cpp
  src/ReClaM/MSEFFNet.cpp
  src/ReClaM/MSERNNet.cpp
  src/ReClaM/McSvmApproximation.cpp
  src/ReClaM/Model.cpp
  src/ReClaM/NegativeLogLikelihood.cpp
  src/ReClaM/NegativePolarization.cpp
  src/ReClaM/NetParams.cpp
  src/ReClaM/NoisyRprop.cpp
  src/ReClaM/NoisySvmLikelihood.cpp
  src/ReClaM/Optimizer.cpp
  src/ReClaM/PCA.cpp
  src/ReClaM/FisherLDA.cpp
  src/ReClaM/Perceptron.cpp
  src/ReClaM/KernelMatrices.cpp
  src/ReClaM/Utilities.cpp
  src/ReClaM/QpSvmDecomp.cpp
  src/ReClaM/QpBoxDecomp.cpp
  src/ReClaM/QpMcDecomp.cpp
  src/ReClaM/QpMcStzDecomp.cpp
  src/ReClaM/QpEbCsDecomp.cpp
  src/ReClaM/ROC.cpp
  src/ReClaM/RadiusMargin.cpp
  src/ReClaM/SigmoidFit.cpp
  src/ReClaM/SigmoidModel.cpp
  src/ReClaM/Softmax.cpp
  #	src/ReClaM/SpanBound.cpp
  src/ReClaM/SpanBound1.cpp
  src/ReClaM/SpanBound2.cpp
  src/ReClaM/SpanBoundA.cpp
  src/ReClaM/Svm.cpp
  src/ReClaM/SvmApproximation.cpp
  src/ReClaM/ValidationError.cpp
  src/ReClaM/VarianceEstimator.cpp
  )

SOURCE_GROUP( ReClaM\\include FILES ${RECLAM_HEADERS} )
SOURCE_GROUP( ReClaM\\src FILES ${RECLAM_SRCS} )

SET( MIXTURE_HEADERS
  include/Mixture/CodeBook.h
  include/Mixture/KernelDensity.h
  include/Mixture/LocalLinearRegression.h
  include/Mixture/MixtureLinearRegression.h
  include/Mixture/MixtureModel.h
  include/Mixture/MixtureOfGaussians.h
  include/Mixture/RBFN-PTO.h
  include/Mixture/RBFN.h
  include/Mixture/RandomVector.h
  )

SET( MIXTURE_SRCS
  src/Mixture/CodeBook.cpp
  src/Mixture/KernelDensity.cpp
  src/Mixture/MixtureLinearRegression.cpp
  src/Mixture/LocalLinearRegression.cpp
  src/Mixture/MixtureOfGaussians.cpp
  src/Mixture/RBFN.cpp
  )

SOURCE_GROUP( Mixture\\include FILES ${MIXTURE_HEADERS} )
SOURCE_GROUP( Mixture\\src FILES ${MIXTURE_SRCS} )

SET( TIMESERIES_HEADERS
  include/TimeSeries/BimodalBrownianProcess.h
  include/TimeSeries/Counter.h
  include/TimeSeries/DiscreteMackeyGlass.h
  include/TimeSeries/Embedding.h
  include/TimeSeries/Generator.h
  include/TimeSeries/IOGenerator.h
  include/TimeSeries/IOSamples.h
  include/TimeSeries/Lorenz63.h
  include/TimeSeries/Lorenz84.h
  include/TimeSeries/MackeyGlass.h
  include/TimeSeries/NoisyIOSamples.h
  include/TimeSeries/NoisyMackeyGlass.h
  include/TimeSeries/RK4-1D.h
  include/TimeSeries/RK4.h
  include/TimeSeries/SelectComponent.h
  )

SET( TIMESERIES_SRCS
  src/TimeSeries/BimodalBrownianProcess.cpp
  src/TimeSeries/DiscreteMackeyGlass.cpp
  src/TimeSeries/Lorenz63.cpp
  src/TimeSeries/Lorenz84.cpp
  src/TimeSeries/MackeyGlass.cpp
  src/TimeSeries/NoisyIOSamples.cpp
  src/TimeSeries/NoisyMackeyGlass.cpp
  src/TimeSeries/RK4-1D.cpp
  src/TimeSeries/RK4.cpp
  )

SOURCE_GROUP( TimeSeries\\include FILES ${TIMESERIES_HEADERS} )
SOURCE_GROUP( TimeSeries\\src FILES ${TIMESERIES_SRCS} )

SET( FUZZY_HEADERS
  include/Fuzzy/BellFS.h
  include/Fuzzy/BellLT.h
  include/Fuzzy/ComposedFS.h
  include/Fuzzy/ComposedLT.h
  include/Fuzzy/ComposedNDimFS.h
  include/Fuzzy/ConstantFS.h
  include/Fuzzy/ConstantLT.h
  include/Fuzzy/CustomIM.h
  include/Fuzzy/CustomizedFS.h
  include/Fuzzy/CustomizedLT.h
  include/Fuzzy/FuzzyException.h
  include/Fuzzy/FuzzyIncludes.h
  include/Fuzzy/FuzzyRelation.h
  include/Fuzzy/FuzzySet.h
  include/Fuzzy/GeneralizedBellFS.h
  include/Fuzzy/GeneralizedBellLT.h
  include/Fuzzy/HomogenousNDimFS.h
  include/Fuzzy/Implication.h
  include/Fuzzy/InferenceMachine.h
  include/Fuzzy/InfinityFS.h
  include/Fuzzy/InfinityLT.h
  include/Fuzzy/LinguisticTerm.h
  include/Fuzzy/LinguisticVariable.h
  include/Fuzzy/MamdaniIM.h
  include/Fuzzy/NDimFS.h
  include/Fuzzy/Operators.h
  include/Fuzzy/RCObject.h
  include/Fuzzy/RCPtr.h
  include/Fuzzy/Rule.h
  include/Fuzzy/RuleBase.h
  include/Fuzzy/RuleBaseGenerator.h
  include/Fuzzy/SigmoidalFS.h
  include/Fuzzy/SigmoidalLT.h
  include/Fuzzy/SingletonFS.h
  include/Fuzzy/SingletonLT.h
  include/Fuzzy/SugenoIM.h
  include/Fuzzy/SugenoRule.h
  include/Fuzzy/TrapezoidFS.h
  include/Fuzzy/TrapezoidLT.h
  include/Fuzzy/TriangularFS.h
  include/Fuzzy/TriangularLT.h
  # include/Fuzzy/xmlParser.h
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
  #	src/Fuzzy/xmlParser.cpp
  )

SOURCE_GROUP( Fuzzy\\include FILES ${FUZZY_HEADERS} )
SOURCE_GROUP( Fuzzy\\src FILES ${FUZZY_SRCS} )

SET( SRCS
  src/SharkDefs.cpp
  ${ARRAY_SRCS}
  ${ARRAY_HEADERS}
  ${RNG_SRCS}
  ${RNG_HEADERS}
  ${LINALG_SRCS}
  ${LINALG_HEADERS}
  ${FILEUTIL_SRCS}
  ${FILEUTIL_HEADERS}
  ${EALIB_SRCS}
  ${EALIB_HEADERS}
  ${MOOEALIB_SRCS}
  ${MOOEALIB_HEADERS}
  ${RECLAM_SRCS}
  ${RECLAM_HEADERS}
  ${MIXTURE_SRCS}
  ${MIXTURE_HEADERS}
  ${TIMESERIES_SRCS}
  ${TIMESERIES_HEADERS}
  ${FUZZY_SRCS}
  ${FUZZY_HEADERS}	
  )

SET( PUBLIC_HEADERS
  SharkDefs.h
  ${ARRAY_HEADERS}
  ${RNG_HEADERS}
  ${LINALG_HEADERS}
  ${FILEUTIL_HEADERS}
  ${EALIB_HEADERS}
  ${MOOEALIB_HEADERS}
  ${RECLAM_HEADERS}
  ${MIXTURE_HEADERS}
  ${TIMESERIES_HEADERS}
  ${FUZZY_HEADERS}	
  )

IF( WIN32 )
  ADD_LIBRARY( shark ${SRCS} )
  #SET_TARGET_PROPERTIES( shark PROPERTIES DEBUG_PREFIX  )
  #SET_TARGET_PROPERTIES( shark PROPERTIES DEBUG_POSTFIX _debug )
  SET( LINK_LIBRARIES )
ELSE( WIN32 )
  ADD_LIBRARY( shark SHARED ${SRCS} )
  SET_TARGET_PROPERTIES( shark PROPERTIES DEBUG_POSTFIX _debug )
ENDIF( WIN32 )

IF( UNIX AND NOT APPLE )
  SET( LINK_LIBRARIES rt )
ENDIF( UNIX AND NOT APPLE )

TARGET_LINK_LIBRARIES( shark ${LINK_LIBRARIES} )

OPTION( SHARK_OPT_COMPILE_EXAMPLES "Compile example programs." ON )

IF( SHARK_OPT_COMPILE_EXAMPLES )
  ADD_SUBDIRECTORY( examples )

 

ENDIF( SHARK_OPT_COMPILE_EXAMPLES )

OPTION( SHARK_OPT_COMPILE_DOCUMENTATION "Compile documentation." OFF )

IF( SHARK_OPT_COMPILE_DOCUMENTATION )
  ADD_SUBDIRECTORY( doc doc )
ENDIF( SHARK_OPT_COMPILE_DOCUMENTATION )

INSTALL( TARGETS shark 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

SET_TARGET_PROPERTIES( shark PROPERTIES
  VERSION 2.3.0
  SOVERSION 2.3.0
  )

INSTALL( FILES 
  include/SharkDefs.h
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}
  )

INSTALL( FILES 
  ${ARRAY_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/Array
  )

INSTALL( FILES 
  ${RNG_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/Rng
  )

INSTALL( FILES 
  ${LINALG_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/LinAlg
  )

INSTALL( FILES 
  ${FILEUTIL_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/FileUtil
  )

INSTALL( FILES 
  ${EALIB_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/EALib
  )

INSTALL( FILES 
  ${MOOEALIB_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/MOO-EALib
  )

INSTALL( FILES 
  ${RECLAM_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/ReClaM
  )

INSTALL( FILES 
  ${MIXTURE_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/Mixture
  )

INSTALL( FILES 
  ${TIMESERIES_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/TimeSeries
  )

INSTALL( FILES 
  ${FUZZY_HEADERS}
  DESTINATION ${SHARK_INSTALL_INCLUDE_DIR}/Fuzzy
  )



