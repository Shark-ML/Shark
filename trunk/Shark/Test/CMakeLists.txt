#####################################################################
#       Shark Machine Learning Library                              #
#       Setup for unit testing                                      #
#       Test invocation: CTest                                      #
#       Test implementation: Boost UTF                              #
#####################################################################


#####################################################################
#       Configure logging of test restuls to XML                    #
#####################################################################
option( LOG_TEST_OUTPUT "Log test output to XML files." OFF )

#####################################################################
#   Adds a unit test for the shark library                          #
#   Param: SRC Source files for compilation                         #
#   Param: NAME Target name for the resulting executable            #
#   Output: Executable in ${SHARK}/Test/bin                         #
#                                                                   #
#       If OPT_LOG_TEST_OUTPUT is enabled, test log is written      #
#   to ${NAME_Log.xml} in ${SHARK}/Test/bin.                        #
#####################################################################
macro(SHARK_ADD_TEST SRC NAME)
   
  if( OPT_LOG_TEST_OUTPUT )
    set( XML_LOGGING_COMMAND_LINE_ARGS "--log_level=test_suite --log_format=XML --log_sink=${NAME}_Log.xml --report_level=no" )
  endif( OPT_LOG_TEST_OUTPUT )

  # Create the test executable
  add_executable( ${NAME} ${SRC} Models/derivativeTestHelper.h )
  target_link_libraries( ${NAME} shark)

  if(GCOV_CHECK)
    target_link_libraries( ${NAME} gcov)
  endif (GCOV_CHECK)

  # Add the test
  add_test( ${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME} ${XML_LOGGING_COMMAND_LINE_ARGS} )

  if(GCOV_CHECK)
    add_Test( ${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME} ${XML_LOGGING_COMMAND_LINE_ARGS} COMMAND Coverage)
  endif (GCOV_CHECK)
endmacro()

# BLAS Tests
shark_add_test( LinAlg/BLAS/iterators.cpp BLAS_Iterators)
shark_add_test( LinAlg/BLAS/vector_assign.cpp BLAS_Vector_Assign)
shark_add_test( LinAlg/BLAS/matrix_assign.cpp BLAS_Matrix_Assign)
shark_add_test( LinAlg/BLAS/compressed_vector.cpp BLAS_Compressed_Vector)
shark_add_test( LinAlg/BLAS/triangular_matrix.cpp BLAS_Triangular_Matrix)
shark_add_test( LinAlg/BLAS/compressed_matrix.cpp BLAS_Compressed_Matrix)
shark_add_test( LinAlg/BLAS/vector_proxy.cpp BLAS_Vector_Proxy)
shark_add_test( LinAlg/BLAS/matrix_proxy.cpp BLAS_Matrix_Proxy)
shark_add_test( LinAlg/BLAS/vector_expression.cpp BLAS_Vector_Expression)
shark_add_test( LinAlg/BLAS/axpy_prod.cpp BLAS_Axpy_Prod)
shark_add_test( LinAlg/BLAS/triangular_prod.cpp BLAS_Triangular_Prod)

# LinAlg Tests
shark_add_test( LinAlg/DiagonalMatrix.cpp LinAlg_DiagonalMatrix)
shark_add_test( LinAlg/sumRows.cpp LinAlg_SumRows)
shark_add_test( LinAlg/Proxy.cpp LinAlg_Proxy )
shark_add_test( LinAlg/repeat.cpp LinAlg_Repeat)
shark_add_test( LinAlg/rotations.cpp LinAlg_rotations )
shark_add_test( LinAlg/permute.cpp LinAlg_Permutations )
shark_add_test( LinAlg/KernelMatrix.cpp LinAlg_KernelMatrix )
shark_add_test( LinAlg/Metrics.cpp LinAlg_Metrics)
shark_add_test( LinAlg/RQ.cpp LinAlg_RQ )
shark_add_test( LinAlg/eigensymm.cpp LinAlg_eigensymm )
shark_add_test( LinAlg/svd.cpp LinAlg_svd )
shark_add_test( LinAlg/choleskyDecomposition.cpp LinAlg_choleskyDecomposition)
shark_add_test( LinAlg/solve.cpp LinAlg_solve)
shark_add_test( LinAlg/BLAS/transformations.cpp BLAS_Transformations)
shark_add_test( LinAlg/Initialize.cpp LinAlg_Initialize)
shark_add_test( LinAlg/LRUCache.cpp LinAlg_LRUCache )
shark_add_test( LinAlg/PartlyPrecomputedMatrix.cpp LinAlg_PartlyPrecomputedMatrix )

#Algorithms tests 
#Direct Search
shark_add_test( Algorithms/DirectSearch/CMA.cpp DirectSearch_CMA )
shark_add_test( Algorithms/DirectSearch/CMSA.cpp DirectSearch_CMSA )
shark_add_test( Algorithms/DirectSearch/ElitistCMA.cpp DirectSearch_ElitistCMA )
shark_add_test( Algorithms/DirectSearch/VDCMA.cpp DirectSearch_VDCMA )
shark_add_test( Algorithms/DirectSearch/MOCMA.cpp DirectSearch_MOCMA )
shark_add_test( Algorithms/DirectSearch/SteadyStateMOCMA.cpp DirectSearch_SteadyStateMOCMA )
shark_add_test( Algorithms/DirectSearch/RealCodedNSGAII.cpp DirectSearch_RealCodedNSGAII )
shark_add_test( Algorithms/DirectSearch/FastNonDominatedSort.cpp DirectSearch_FastNonDominatedSort )
shark_add_test( Algorithms/DirectSearch/ParetoDominanceComparator.cpp DirectSearch_ParetoDominanceComparator )

# Direct Search Operator tests
shark_add_test( Algorithms/DirectSearch/Operators/Selection/Selection.cpp DirectSearch_Selection )
shark_add_test( Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.cpp DirectSearch_IndicatorBasedSelection )
shark_add_test( Algorithms/DirectSearch/Operators/Mutation/BitflipMutation.cpp DirectSearch_BitflipMutation )
shark_add_test( Algorithms/DirectSearch/Operators/PenalizingEvaluator.cpp DirectSearch_PenalizingEvaluator )

# Direct Search Indicator tests
shark_add_test( Algorithms/DirectSearch/Indicators/HypervolumeIndicator.cpp DirectSearch_HypervolumeIndicator )

# GradientDescent
shark_add_test( Algorithms/GradientDescent/LineSearch.cpp GradDesc_LineSearch )
shark_add_test( Algorithms/GradientDescent/BFGS.cpp GradDesc_BFGS )
shark_add_test( Algorithms/GradientDescent/LBFGS.cpp GradDesc_LBFGS )
shark_add_test( Algorithms/GradientDescent/CG.cpp GradDesc_CG )
shark_add_test( Algorithms/GradientDescent/Rprop.cpp GradDesc_Rprop )
shark_add_test( Algorithms/GradientDescent/SteepestDescent.cpp GradDesc_SteepestDescent )


# Trainers
shark_add_test( Algorithms/Trainers/CSvmTrainer.cpp Trainers_CSvmTrainer )
shark_add_test( Algorithms/Trainers/FisherLDA.cpp Trainers_FisherLDA )
shark_add_test( Algorithms/Trainers/KernelMeanClassifier.cpp Trainers_KernelMeanClassifier )
shark_add_test( Algorithms/Trainers/EpsilonSvmTrainer.cpp Trainers_EpsilonSvmTrainer )
shark_add_test( Algorithms/Trainers/OneClassSvmTrainer.cpp Trainers_OneClassSvmTrainer )
shark_add_test( Algorithms/Trainers/RegularizationNetworkTrainer.cpp Trainers_RegularizationNetworkTrainer )
shark_add_test( Algorithms/Trainers/LDA.cpp Trainers_LDA )
shark_add_test( Algorithms/Trainers/LinearRegression.cpp Trainers_LinearRegression )
shark_add_test( Algorithms/Trainers/McSvmTrainer.cpp Trainers_McSvmTrainer )
shark_add_test( Algorithms/Trainers/LinearSvmTrainer.cpp Trainers_LinearSvmTrainer )
shark_add_test( Algorithms/Trainers/NBClassifierTrainerTests.cpp Trainers_NBClassifier )
shark_add_test( Algorithms/Trainers/Normalization.cpp Trainers_Normalization )
shark_add_test( Algorithms/Trainers/KernelNormalization.cpp Trainers_KernelNormalization )
shark_add_test( Algorithms/Trainers/SigmoidFit.cpp Trainers_SigmoidFit )
shark_add_test( Algorithms/Trainers/PCA.cpp Trainers_PCA )
shark_add_test( Algorithms/Trainers/Perceptron.cpp Trainers_Perceptron )
shark_add_test( Algorithms/Trainers/MissingFeatureSvmTrainerTests.cpp Trainers_MissingFeatureSvmTrainer )
shark_add_test( Algorithms/Trainers/Budgeted/AbstractBudgetMaintenanceStrategy_Test.cpp Trainers_AbstractBudgetMaintenanceStrategy )
shark_add_test( Algorithms/Trainers/Budgeted/MergeBudgetMaintenanceStrategy_Test.cpp MergeBudgetMaintenanceStrategy )
shark_add_test( Algorithms/Trainers/Budgeted/RemoveBudgetMaintenanceStrategy_Test.cpp RemoveBudgetMaintenanceStrategy )
shark_add_test( Algorithms/Trainers/Budgeted/KernelBudgetedSGDTrainer_Test.cpp KernelBudgetedSGDTrainer )

# Misc algorithms
shark_add_test( Algorithms/GridSearch.cpp Algorithms_GridSearch )
shark_add_test( Algorithms/Hypervolume.cpp Algorithms_Hypervolume )
shark_add_test( Algorithms/nearestneighbors.cpp Algorithms_NearestNeighbor )
shark_add_test( Algorithms/KMeans.cpp Algorithms_KMeans )
shark_add_test( Algorithms/JaakkolaHeuristic.cpp Algorithms_JaakkolaHeuristic )

# Models
shark_add_test( Models/ConcatenatedModel.cpp Models_ConcatenatedModel )
shark_add_test( Models/FFNet.cpp Models_FFNet )
shark_add_test( Models/Autoencoder.cpp Models_Autoencoder )
shark_add_test( Models/TiedAutoencoder.cpp Models_TiedAutoencoder )
shark_add_test( Models/LinearModel.cpp Models_LinearModel )
shark_add_test( Models/LinearNorm.cpp Models_LinearNorm )
shark_add_test( Models/ConvexCombination.cpp Models_ConvexCombination )
shark_add_test( Models/NBClassifierTests.cpp Models_NBClassifier )
#shark_add_test( Models/OnlineRNNet.cpp Models_OnlineRNNet )
shark_add_test( Models/RBFLayer.cpp Models_RBFLayer ) 
shark_add_test( Models/RNNet.cpp Models_RNNet ) 
shark_add_test( Models/CMAC.cpp Models_CMAC )
shark_add_test( Models/MeanModel.cpp Models_MeanModel )

shark_add_test( Models/SigmoidModel.cpp Models_SigmoidModel )
shark_add_test( Models/Softmax.cpp Models_Softmax )
shark_add_test( Models/SoftNearestNeighborClassifier.cpp Models_SoftNearestNeighborClassifier )
shark_add_test( Models/Kernels/KernelExpansion.cpp Models_KernelExpansion )
shark_add_test( Models/NearestNeighborRegression.cpp Models_NearestNeighborRegression )
shark_add_test( Models/OneVersusOneClassifier.cpp Models_OneVersusOneClassifier )

# Kernels
shark_add_test( Models/Kernels/GaussianRbfKernel.cpp Models_GaussianKernel )
shark_add_test( Models/Kernels/LinearKernel.cpp Models_LinearKernel )
shark_add_test( Models/Kernels/NormalizedKernel.cpp Models_NormalizedKernel )
shark_add_test( Models/Kernels/MonomialKernel.cpp Models_MonomialKernel )
shark_add_test( Models/Kernels/PolynomialKernel.cpp Models_PolynomialKernel )
shark_add_test( Models/Kernels/ScaledKernel.cpp Models_ScaledKernel )
shark_add_test( Models/Kernels/WeightedSumKernel.cpp Models_WeightedSumKernel )
shark_add_test( Models/Kernels/ProductKernel.cpp Models_ProductKernel )
shark_add_test( Models/Kernels/ArdKernel.cpp Models_ArdKernel )
shark_add_test( Models/Kernels/MklKernel.cpp Models_MklKernel )
shark_add_test( Models/Kernels/SubrangeKernel.cpp Models_SubrangeKernel )
shark_add_test( Models/Kernels/DiscreteKernel.cpp Models_DiscreteKernel )
shark_add_test( Models/Kernels/MultiTaskKernel.cpp Models_MultiTaskKernel )
shark_add_test( Models/Kernels/ModelKernel.cpp Models_ModelKernel )

# KernelMethods
shark_add_test( Models/Kernels/KernelHelpers.cpp Models_KernelHelpers )
shark_add_test( Models/Kernels/KernelNearestNeighborClassifier.cpp Models_KernelNearestNeighborClassifier )
shark_add_test( Models/Kernels/KernelNearestNeighborRegression.cpp Models_KernelNearestNeighborRegression )
shark_add_test( Models/Kernels/EvalSkipMissingFeaturesTests.cpp Models_EvalSkipMissingFeatures )
shark_add_test( Models/Kernels/MissingFeaturesKernelExpansionTests.cpp Models_MissingFeaturesKernelExpansion )
shark_add_test( Models/Kernels/CSvmDerivative.cpp Models_CSvmDerivative )

# Trees
shark_add_test( Models/RFClassifier.cpp Models_RFClassifier )
shark_add_test( Models/CARTClassifier.cpp Models_CARTClassifier )

# Core tests
#shark_add_test( Core/ScopedHandleTests.cpp Core_ScopedHandleTests )
shark_add_test( Core/Iterators.cpp Core_Iterators )
shark_add_test( Core/Math.cpp Core_Math )

# Data Tests
shark_add_test( Data/Csv.cpp Data_Csv )
shark_add_test( Data/Bootstrap.cpp Data_Bootstrap )
shark_add_test( Data/CVDatasetTools.cpp Data_CVDatasetTools )
shark_add_test( Data/Dataset.cpp Data_Dataset )
shark_add_test( Data/DataView.cpp Data_DataView )
shark_add_test( Data/LabelOrder_Test.cpp Data_LabelOrder )
shark_add_test( Data/Statistics.cpp Data_Statistics )
if(HDF5_FOUND)
  shark_add_test( Data/HDF5Tests.cpp Data_HDF5 )
endif()
shark_add_test( Data/SparseData.cpp Data_SparseData )
shark_add_test( Data/ExportKernelMatrix.cpp Data_ExportKernelMatrix )

#Objective Functions
shark_add_test( ObjectiveFunctions/ErrorFunction.cpp ObjFunct_ErrorFunction )
shark_add_test( ObjectiveFunctions/SparseAutoencoderError.cpp ObjFunct_SparseAutoencoderError )
shark_add_test( ObjectiveFunctions/NoisyErrorFunction.cpp ObjFunct_NoisyErrorFunction )
shark_add_test( ObjectiveFunctions/CrossValidation.cpp ObjFunct_CrossValidation )
shark_add_test( ObjectiveFunctions/Benchmarks.cpp ObjFunct_Benchmarks )
shark_add_test( ObjectiveFunctions/KernelTargetAlignment.cpp ObjFunct_KernelTargetAlignment )
shark_add_test( ObjectiveFunctions/KernelBasisDistance.cpp ObjFunct_KernelBasisDistance )
shark_add_test( ObjectiveFunctions/RadiusMarginQuotient.cpp ObjFunct_RadiusMarginQuotient )
shark_add_test( ObjectiveFunctions/LooError.cpp ObjFunct_LooError )
shark_add_test( ObjectiveFunctions/LooErrorCSvm.cpp ObjFunct_LooErrorCSvm )
shark_add_test( ObjectiveFunctions/NegativeLogLikelihood.cpp ObjFunct_NegativeLogLikelihood )
shark_add_test( ObjectiveFunctions/SvmLogisticInterpretation.cpp ObjFunct_SvmLogisticInterpretation )
shark_add_test( ObjectiveFunctions/BoxConstraintHandler.cpp ObjFunct_BoxConstraintHandler )

#Objective Functions/Loss
shark_add_test( ObjectiveFunctions/CrossEntropy.cpp ObjFunct_CrossEntropy )
shark_add_test( ObjectiveFunctions/SquaredLoss.cpp ObjFunct_SquaredLoss )
shark_add_test( ObjectiveFunctions/HingeLoss.cpp ObjFunct_HingeLoss )
shark_add_test( ObjectiveFunctions/SquaredHingeLoss.cpp ObjFunct_SquaredHingeLoss )
shark_add_test( ObjectiveFunctions/EpsilonHingeLoss.cpp ObjFunct_EpsilonHingeLoss )
shark_add_test( ObjectiveFunctions/SquaredEpsilonHingeLoss.cpp ObjFunct_SquaredEpsilonHingeLoss )
shark_add_test( ObjectiveFunctions/AbsoluteLoss.cpp ObjFunct_AbsoluteLoss )
shark_add_test( ObjectiveFunctions/HuberLoss.cpp ObjFunct_HuberLoss )
shark_add_test( ObjectiveFunctions/TukeyBiweightLoss.cpp ObjFunct_TukeyBiweightLoss )
shark_add_test( ObjectiveFunctions/AUC.cpp ObjFunct_AUC )
shark_add_test( ObjectiveFunctions/NegativeGaussianProcessEvidence.cpp ObjFunct_NegativeGaussianProcessEvidence )

#Rng
shark_add_test( Rng/Rng.cpp Rng_Distributions )
shark_add_test( Rng/MultiVariateNormal.cpp Rng_MultiVariateNormal )
shark_add_test( Rng/MultiNomial.cpp Rng_MultiNomialDistribution )

#RBM
shark_add_test( RBM/BinaryLayer.cpp RBM_BinaryLayer)
shark_add_test( RBM/BipolarLayer.cpp RBM_BipolarLayer)
shark_add_test( RBM/GaussianLayer.cpp RBM_GaussianLayer)
shark_add_test( RBM/TruncatedExponentialLayer.cpp RBM_TruncatedExponentialLayer)

shark_add_test( RBM/MarkovChain.cpp RBM_MarkovChain)
#shark_add_test( RBM/GibbsOperator.cpp RBM_GibbsOperator)//not compiling anymore needs rewrite

shark_add_test( RBM/Energy.cpp RBM_Energy)
shark_add_test( RBM/AverageEnergyGradient.cpp RBM_AverageEnergyGradient)
shark_add_test( RBM/Analytics.cpp RBM_Analytics)

shark_add_test( RBM/ExactGradient.cpp RBM_ExactGradient)
#shark_add_test( RBM/ContrastiveDivergence.cpp RBM_ContrastiveDivergence) #does not compile currently
shark_add_test( RBM/TemperedMarkovChain.cpp RBM_TemperedMarkovChain)

shark_add_test( RBM/ParallelTemperingTraining.cpp RBM_PTTraining)
shark_add_test( RBM/PCDTraining.cpp RBM_PCDTraining)
shark_add_test( RBM/ContrastiveDivergenceTraining.cpp RBM_ContrastiveDivergenceTraining)
shark_add_test( RBM/ExactGradientTraining.cpp RBM_ExactGradientTraining)

# Copy test file
if(HDF5_FOUND)
	add_custom_command(
	    TARGET Data_HDF5
	    POST_BUILD
	    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Test/test_data
	    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Test/test_data ${CMAKE_BINARY_DIR}/Test/test_data)	
endif()

# Create output dir
add_custom_command(
    TARGET Data_Csv
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Test/test_output
)

add_custom_command(
    TARGET Data_ExportKernelMatrix
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Test/test_output
)
