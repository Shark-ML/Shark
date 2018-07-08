#####################################################################
#       Shark Machine Learning Library                              #
#       Setup for unit testing                                      #
#       Test invocation: CTest                                      #
#       Test implementation: Boost UTF                              #
#####################################################################

#####################################################################
#       Get Boost Unit test                  #
#####################################################################
find_package( 
	Boost 1.48.0 REQUIRED COMPONENTS
	unit_test_framework
)

set( boost_unit_test_lib ${Boost_LIBRARIES})

#####################################################################
#       Configure logging of test restults to XML                    #
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
	target_link_libraries( ${NAME} shark ${boost_unit_test_lib})
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 11)
	set_property(TARGET ${NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

	if(GCOV_CHECK)
		target_link_libraries( ${NAME} gcov)
	endif (GCOV_CHECK)

	# Add the test
	add_test( ${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME} ${XML_LOGGING_COMMAND_LINE_ARGS} )

	set_target_properties(${NAME} PROPERTIES FOLDER "Tests")
	if(GCOV_CHECK)
		add_Test( ${NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME} ${XML_LOGGING_COMMAND_LINE_ARGS} COMMAND Coverage)
	endif (GCOV_CHECK)
endmacro()

# LinAlg Tests
shark_add_test( LinAlg/rotations.cpp LinAlg_rotations )
shark_add_test( LinAlg/permute.cpp LinAlg_Permutations )
shark_add_test( LinAlg/KernelMatrix.cpp LinAlg_KernelMatrix )
shark_add_test( LinAlg/Metrics.cpp LinAlg_Metrics)

shark_add_test( LinAlg/LRUCache.cpp LinAlg_LRUCache )
shark_add_test( LinAlg/PartlyPrecomputedMatrix.cpp LinAlg_PartlyPrecomputedMatrix )

#Algorithms tests 
#Direct Search
shark_add_test( Algorithms/DirectSearch/CMA.cpp DirectSearch_CMA )
shark_add_test( Algorithms/DirectSearch/CMSA.cpp DirectSearch_CMSA )
shark_add_test( Algorithms/DirectSearch/ElitistCMA.cpp DirectSearch_ElitistCMA )
shark_add_test( Algorithms/DirectSearch/CrossEntropyMethod.cpp DirectSearch_CrossEntropyMethod )
shark_add_test( Algorithms/DirectSearch/VDCMA.cpp DirectSearch_VDCMA )
shark_add_test( Algorithms/DirectSearch/MOCMA.cpp DirectSearch_MOCMA )
shark_add_test( Algorithms/DirectSearch/SteadyStateMOCMA.cpp DirectSearch_SteadyStateMOCMA )
shark_add_test( Algorithms/DirectSearch/RealCodedNSGAII.cpp DirectSearch_RealCodedNSGAII )
shark_add_test( Algorithms/DirectSearch/RealCodedNSGAIII.cpp DirectSearch_RealCodedNSGAIII )
shark_add_test( Algorithms/DirectSearch/SMS-EMOA.cpp DirectSearch_SMS-EMOA )
shark_add_test( Algorithms/DirectSearch/NonDominatedSort.cpp DirectSearch_NonDominatedSort )
shark_add_test( Algorithms/DirectSearch/ParetoDominance.cpp DirectSearch_ParetoDominance )
shark_add_test( Algorithms/DirectSearch/Operators/HypervolumeSubsetSelection.cpp DirectSearch_HypervolumeSubsetSelection )
shark_add_test( Algorithms/DirectSearch/Operators/HypervolumeContribution.cpp DirectSearch_HypervolumeContribution )
shark_add_test( Algorithms/DirectSearch/MOEAD.cpp DirectSearch_MOEAD )
shark_add_test( Algorithms/DirectSearch/RVEA.cpp DirectSearch_RVEA )

# Direct Search Operator tests
shark_add_test( Algorithms/DirectSearch/Operators/Selection/Selection.cpp DirectSearch_Selection )
shark_add_test( Algorithms/DirectSearch/Operators/Selection/IndicatorBasedSelection.cpp DirectSearch_IndicatorBasedSelection )
shark_add_test( Algorithms/DirectSearch/Operators/Mutation/BitflipMutation.cpp DirectSearch_BitflipMutation )
shark_add_test( Algorithms/DirectSearch/Operators/PenalizingEvaluator.cpp DirectSearch_PenalizingEvaluator )
shark_add_test( Algorithms/DirectSearch/Operators/Lattice.cpp DirectSearch_Operators_Lattice )
shark_add_test( Algorithms/DirectSearch/Operators/Selection/ReferenceVectorGuidedSelection.cpp DirectSearch_Operators_ReferenceVectorGuidedSelection )
shark_add_test( Algorithms/DirectSearch/Operators/ReferenceVectorAdaptation.cpp DirectSearch_Operators_ReferenceVectorAdaptation )

# Direct Search Indicator tests
shark_add_test( Algorithms/DirectSearch/Indicators/HypervolumeIndicator.cpp DirectSearch_HypervolumeIndicator )

# GradientDescent
shark_add_test( Algorithms/GradientDescent/LineSearch.cpp GradDesc_LineSearch )
shark_add_test( Algorithms/GradientDescent/BFGS.cpp GradDesc_BFGS )
shark_add_test( Algorithms/GradientDescent/LBFGS.cpp GradDesc_LBFGS )
shark_add_test( Algorithms/GradientDescent/CG.cpp GradDesc_CG )
shark_add_test( Algorithms/GradientDescent/Adam.cpp GradDesc_Adam )
shark_add_test( Algorithms/GradientDescent/Rprop.cpp GradDesc_Rprop )
shark_add_test( Algorithms/GradientDescent/SteepestDescent.cpp GradDesc_SteepestDescent )


# Trainers
shark_add_test( Algorithms/Trainers/CSvmTrainer.cpp Trainers_CSvmTrainer )
shark_add_test( Algorithms/Trainers/RankingSvmTrainer.cpp Trainers_RankingSvmTrainer )
shark_add_test( Algorithms/Trainers/FisherLDA.cpp Trainers_FisherLDA )
shark_add_test( Algorithms/Trainers/KernelMeanClassifier.cpp Trainers_KernelMeanClassifier )
shark_add_test( Algorithms/Trainers/EpsilonSvmTrainer.cpp Trainers_EpsilonSvmTrainer )
shark_add_test( Algorithms/Trainers/OneClassSvmTrainer.cpp Trainers_OneClassSvmTrainer )
shark_add_test( Algorithms/Trainers/RegularizationNetworkTrainer.cpp Trainers_RegularizationNetworkTrainer )
shark_add_test( Algorithms/Trainers/LDA.cpp Trainers_LDA )
shark_add_test( Algorithms/Trainers/LinearRegression.cpp Trainers_LinearRegression )
shark_add_test( Algorithms/Trainers/LinearSAGTrainer.cpp Trainers_LinearSAGTrainer )
shark_add_test( Algorithms/Trainers/LassoRegression.cpp Trainers_LassoRegression )
shark_add_test( Algorithms/Trainers/LogisticRegression.cpp Trainers_LogisticRegression )
shark_add_test( Algorithms/Trainers/McSvmTrainer.cpp Trainers_McSvmTrainer )
shark_add_test( Algorithms/Trainers/LinearSvmTrainer.cpp Trainers_LinearSvmTrainer )
shark_add_test( Algorithms/Trainers/Normalization.cpp Trainers_Normalization )
shark_add_test( Algorithms/Trainers/KernelNormalization.cpp Trainers_KernelNormalization )
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
shark_add_test( Algorithms/ApproximateKernelExpansion.cpp Algorithms_ApproximateKernelExpansion )
shark_add_test( Algorithms/JaakkolaHeuristic.cpp Algorithms_JaakkolaHeuristic )

# Models
shark_add_test( Models/ConcatenatedModel.cpp Models_ConcatenatedModel )
shark_add_test( Models/LinearModel.cpp Models_LinearModel )
shark_add_test( Models/ConvolutionalModel.cpp Models_ConvolutionalModel )
shark_add_test( Models/ResizeLayer.cpp Models_ResizeLayer )
shark_add_test( Models/PoolingLayer.cpp Models_PoolingLayer )
shark_add_test( Models/NearestNeighborModel.cpp Models_NearestNeighborModel )
shark_add_test( Models/RBFLayer.cpp Models_RBFLayer ) 
shark_add_test( Models/CMAC.cpp Models_CMAC )
shark_add_test( Models/Ensemble.cpp Models_Ensemble )
shark_add_test( Models/DropoutLayer.cpp Models_DropoutLayer )
shark_add_test( Models/NeuronLayer.cpp Models_NeuronLayer )

shark_add_test( Models/Kernels/KernelExpansion.cpp Models_KernelExpansion )
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
shark_add_test( Models/Kernels/EvalSkipMissingFeaturesTests.cpp Models_EvalSkipMissingFeatures )
shark_add_test( Models/Kernels/MissingFeaturesKernelExpansionTests.cpp Models_MissingFeaturesKernelExpansion )
shark_add_test( Models/Kernels/CSvmDerivative.cpp Models_CSvmDerivative )

# Trees
shark_add_test( Models/RFClassifier.cpp Models_RFClassifier )

# Core tests
#shark_add_test( Core/ScopedHandleTests.cpp Core_ScopedHandleTests )
shark_add_test( Core/Reorder.cpp Core_ImageReorder )

# Data Tests
shark_add_test( Data/Csv.cpp Data_Csv )
shark_add_test( Data/Download.cpp Data_Download )
shark_add_test( Data/Bootstrap.cpp Data_Bootstrap )
shark_add_test( Data/CVDatasetTools.cpp Data_CVDatasetTools )
shark_add_test( Data/Dataset.cpp Data_Dataset )
shark_add_test( Data/DataView.cpp Data_DataView )
shark_add_test( Data/LabelOrder_Test.cpp Data_LabelOrder )
shark_add_test( Data/Statistics.cpp Data_Statistics )
#~ if(HDF5_FOUND)
  #~ shark_add_test( Data/HDF5Tests.cpp Data_HDF5 )
#~ endif()
shark_add_test( Data/SparseData.cpp Data_SparseData )
shark_add_test( Data/ExportKernelMatrix.cpp Data_ExportKernelMatrix )

#Objective Functions
shark_add_test( ObjectiveFunctions/ErrorFunction.cpp ObjFunct_ErrorFunction )
shark_add_test( ObjectiveFunctions/VariationalAutoencoder.cpp ObjFunct_VariationalAutoencoder )
shark_add_test( ObjectiveFunctions/CrossValidation.cpp ObjFunct_CrossValidation )
shark_add_test( ObjectiveFunctions/Benchmarks.cpp ObjFunct_Benchmarks )
shark_add_test( ObjectiveFunctions/KernelTargetAlignment.cpp ObjFunct_KernelTargetAlignment )
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
shark_add_test( ObjectiveFunctions/AUC.cpp ObjFunct_AUC )
shark_add_test( ObjectiveFunctions/NegativeGaussianProcessEvidence.cpp ObjFunct_NegativeGaussianProcessEvidence )

#random
shark_add_test( Rng/MultiVariateNormal.cpp random_MultiVariateNormal )
shark_add_test( Rng/MultiNomial.cpp random_MultiNomialDistribution )

#RBM
shark_add_test( RBM/BinaryLayer.cpp RBM_BinaryLayer)
shark_add_test( RBM/BipolarLayer.cpp RBM_BipolarLayer)
shark_add_test( RBM/GaussianLayer.cpp RBM_GaussianLayer)

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

#marking tests as slow
set_tests_properties( DirectSearch_HypervolumeContribution PROPERTIES LABELS "slow" )
set_tests_properties( Models_CMAC PROPERTIES LABELS "slow" )
set_tests_properties( ObjFunct_SvmLogisticInterpretation PROPERTIES LABELS "slow" )
set_tests_properties( ObjFunct_NegativeLogLikelihood PROPERTIES LABELS "slow" )
set_tests_properties( ObjFunct_LooErrorCSvm PROPERTIES LABELS "slow" )
set_tests_properties( Trainers_LinearSvmTrainer PROPERTIES LABELS "slow" )


# Copy test file
#~ if(HDF5_FOUND)
    #~ add_custom_command(
        #~ TARGET Data_HDF5
        #~ POST_BUILD
        #~ COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test_data
            #~ ${CMAKE_CURRENT_BINARY_DIR}/test_data
    #~ )
#~ endif()

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
