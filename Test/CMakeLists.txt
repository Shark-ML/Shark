#####################################################################
#       Shark Machine Learning Library                              #
#       Setup for unit testing                                      #
#       Test invocation: CTest                                      #
#       Test implementation: Boost UTF                              #
#####################################################################

#####################################################################
#       Configure logging of test restuls to XML                    #
#####################################################################
OPTION( OPT_LOG_TEST_OUTPUT "Log test output to XML files." OFF )

#####################################################################
#   Adds a unit test for the shark library                          #
#   Param: SRC Source files for compilation                         #
#   Param: NAME Target name for the resulting executable            #
#   Output: Executable in ${SHARK}/Test/bin                         #
#                                                                   #
#       If OPT_LOG_TEST_OUTPUT is enabled, test log is written      #
#   to ${NAME_Log.xml} in ${SHARK}/Test/bin.                        #
#####################################################################
MACRO( SHARK_ADD_TEST SRC NAME)

    IF( OPT_LOG_TEST_OUTPUT )
        SET( XML_LOGGING_COMMAND_LINE_ARGS "--log_level=test_suite --log_format=XML --log_sink=${NAME}_Log.xml --report_level=no" )
    ENDIF( OPT_LOG_TEST_OUTPUT )

    ADD_EXECUTABLE( ${NAME}
        ${SRC}
        Models/derivativeTestHelper.h
    )

    SET( LINK_LIBRARIES
        shark
        ${Boost_LIBRARIES}
    )

    TARGET_LINK_LIBRARIES( ${NAME} ${LINK_LIBRARIES} )

    ADD_TEST( ${NAME} ${EXECUTABLE_OUTPUT_PATH}/${NAME} ${XML_LOGGING_COMMAND_LINE_ARGS} )
ENDMACRO()

#LinAlg Tests
SHARK_ADD_TEST( LinAlg/sumRows.cpp LinAlg_SumRows)
SHARK_ADD_TEST( LinAlg/Proxy.cpp LinAlg_Proxy )
SHARK_ADD_TEST( LinAlg/repeat.cpp LinAlg_Repeat)
SHARK_ADD_TEST( LinAlg/covar_corrcoef.cpp LinAlg_covar_corrcoef )
SHARK_ADD_TEST( LinAlg/rotations.cpp LinAlg_rotations )
SHARK_ADD_TEST( LinAlg/permute.cpp LinAlg_Permutations )
SHARK_ADD_TEST( LinAlg/Metrics.cpp LinAlg_Metrics)
SHARK_ADD_TEST( LinAlg/RQ.cpp LinAlg_RQ )
SHARK_ADD_TEST( LinAlg/detsymm.cpp LinAlg_detsymm )
SHARK_ADD_TEST( LinAlg/dlmin.cpp LinAlg_dlmin )
SHARK_ADD_TEST( LinAlg/eigenerr.cpp LinAlg_eigenerr )
SHARK_ADD_TEST( LinAlg/eigensort.cpp LinAlg_eigensort )
SHARK_ADD_TEST( LinAlg/eigensymm.cpp LinAlg_eigensymm )
SHARK_ADD_TEST( LinAlg/eigensymmJacobi2.cpp LinAlg_eigensymmJacobi2 )
SHARK_ADD_TEST( LinAlg/eigensymmjacobi.cpp LinAlg_eigensymmjacobi )
SHARK_ADD_TEST( LinAlg/fft.cpp LinAlg_fft )
SHARK_ADD_TEST( LinAlg/g_inverse.cpp LinAlg_g_inverse )#todo: more tests
SHARK_ADD_TEST( LinAlg/linmin.cpp LinAlg_linmin )
SHARK_ADD_TEST( LinAlg/lnrsrch.cpp LinAlg_lnrsrch )
SHARK_ADD_TEST( LinAlg/rank.cpp LinAlg_rank )
SHARK_ADD_TEST( LinAlg/rank_decomp.cpp LinAlg_rank_decomp )
SHARK_ADD_TEST( LinAlg/VectorStatistics.cpp LinAlg_VectorStatistics )
SHARK_ADD_TEST( LinAlg/svd.cpp LinAlg_svd )
SHARK_ADD_TEST( LinAlg/svdrank.cpp LinAlg_svdrank )
SHARK_ADD_TEST( LinAlg/svdsort.cpp LinAlg_svdsort )
SHARK_ADD_TEST( LinAlg/choleskyDecomposition.cpp LinAlg_choleskyDecomposition)
SHARK_ADD_TEST( LinAlg/invertSymmPositiveDefinite.cpp LinAlg_invertSymmPositiveDefinite)
SHARK_ADD_TEST( LinAlg/eigen.cpp LinAlg_eigen)
SHARK_ADD_TEST( LinAlg/solve.cpp LinAlg_solve)
SHARK_ADD_TEST( LinAlg/VectorTransformations.cpp LinAlg_VectorTransformations)
SHARK_ADD_TEST( LinAlg/Initialize.cpp LinAlg_Initialize)
SHARK_ADD_TEST( LinAlg/fast_prod.cpp LinAlg_FastProd)
SHARK_ADD_TEST( LinAlg/BlockOperations.cpp LinAlg_BlockOperations)
#Operator tests
SHARK_ADD_TEST( Algorithms/DirectSearch/Operators/Selection/Selection.cpp DirectSearch_Selection )
SHARK_ADD_TEST( Algorithms/DirectSearch/Operators/Recombination/Recombination.cpp DirectSearch_Recombination )
SHARK_ADD_TEST( Algorithms/DirectSearch/Operators/Mutation/BitflipMutation.cpp DirectSearch_BitflipMutation )

#Algorithms tests
#Direct Search
SHARK_ADD_TEST( Algorithms/DirectSearch/CMA.cpp DirectSearch_CMA )
SHARK_ADD_TEST( Algorithms/DirectSearch/CMSA.cpp DirectSearch_CMSA )
SHARK_ADD_TEST( Algorithms/DirectSearch/ElitistCMA.cpp DirectSearch_ElitistCMA )
SHARK_ADD_TEST( Algorithms/DirectSearch/OnePlusOneES.cpp DirectSearch_OnePlusOneES )
SHARK_ADD_TEST( Algorithms/DirectSearch/MOCMA.cpp DirectSearch_MOCMA )
SHARK_ADD_TEST( Algorithms/DirectSearch/SteadyStateMOCMA.cpp DirectSearch_SteadyStateMOCMA )
SHARK_ADD_TEST( Algorithms/DirectSearch/Algorithms.cpp Algorithms_DirectSearch )
#GradientDescent
SHARK_ADD_TEST( Algorithms/GradientDescent/BFGS.cpp GradDesc_BFGS )
SHARK_ADD_TEST( Algorithms/GradientDescent/CG.cpp GradDesc_CG )
SHARK_ADD_TEST( Algorithms/GradientDescent/IRLS.cpp GradDesc_IRLS )
SHARK_ADD_TEST( Algorithms/GradientDescent/Rprop.cpp GradDesc_Rprop )
SHARK_ADD_TEST( Algorithms/GradientDescent/NoisyRprop.cpp GradDesc_NoisyRprop )
SHARK_ADD_TEST( Algorithms/GradientDescent/Quickprop.cpp GradDesc_Quickprop )
SHARK_ADD_TEST( Algorithms/GradientDescent/SteepestDescent.cpp GradDesc_SteepestDescent )
#Trainers
SHARK_ADD_TEST( Algorithms/Trainers/CSvmTrainer.cpp Trainers_CSvmTrainer )
SHARK_ADD_TEST( Algorithms/Trainers/FisherLDA.cpp Trainers_FisherLDA )
SHARK_ADD_TEST( Algorithms/Trainers/KernelMeanClassifier.cpp Trainers_KernelMeanClassifier )
SHARK_ADD_TEST( Algorithms/Trainers/KernelRegression.cpp Trainers_KernelRegression )
SHARK_ADD_TEST( Algorithms/Trainers/LDA.cpp Trainers_LDA )
SHARK_ADD_TEST( Algorithms/Trainers/LinearRegression.cpp Trainers_LinearRegression )
SHARK_ADD_TEST( Algorithms/Trainers/McSvmTrainer.cpp Trainers_McSvmTrainer )
SHARK_ADD_TEST( Algorithms/Trainers/LinearSvmTrainer.cpp Trainers_LinearSvmTrainer )
SHARK_ADD_TEST( Algorithms/Trainers/NBClassifierTrainerTests.cpp Trainers_NBClassifier )
SHARK_ADD_TEST( Algorithms/Trainers/Normalization.cpp Trainers_Normalization )
SHARK_ADD_TEST( Algorithms/Trainers/KernelNormalization.cpp Trainers_KernelNormalization )
SHARK_ADD_TEST( Algorithms/Trainers/SigmoidFit.cpp Trainers_SigmoidFit )
SHARK_ADD_TEST( Algorithms/Trainers/PCA.cpp Trainers_PCA )
SHARK_ADD_TEST( Algorithms/Trainers/Perceptron.cpp Trainers_Perceptron )
SHARK_ADD_TEST( Algorithms/Trainers/MissingFeatureSvmTrainerTests.cpp Trainers_MissingFeatureSvmTrainer )

#LP - Linear Programs
SHARK_ADD_TEST( Algorithms/LP/LinearProgram.cpp LP_LinearProgram )
#misc algorithms
SHARK_ADD_TEST( Algorithms/GridSearch.cpp Algorithms_GridSearch )
SHARK_ADD_TEST( Algorithms/Hypervolume.cpp Algorithms_Hypervolume )
SHARK_ADD_TEST( Algorithms/nearestneighbors.cpp Algorithms_NearestNeighbor )
SHARK_ADD_TEST( Algorithms/KMeans.cpp Algorithms_KMeans )
SHARK_ADD_TEST( Algorithms/JaakkolaHeuristic.cpp Algorithms_JaakkolaHeuristic )

SHARK_ADD_TEST( Fuzzy/FCL.cpp Fuzzy_Control_Language_Parser )
SHARK_ADD_TEST( Fuzzy/FuzzySets.cpp Fuzzy_FuzzySets )
SHARK_ADD_TEST( Fuzzy/LinguisticTerms.cpp Fuzzy_LinguisticTerms )
SHARK_ADD_TEST( Fuzzy/Mamdani.cpp Fuzzy_Mamdani )
#Models
SHARK_ADD_TEST( Models/ConcatenatedModel.cpp Models_ConcatenatedModel )
SHARK_ADD_TEST( Models/FFNet.cpp Models_FFNet )
SHARK_ADD_TEST( Models/LinearModel.cpp Models_LinearModel )
SHARK_ADD_TEST( Models/LinearNorm.cpp Models_LinearNorm )
SHARK_ADD_TEST( Models/NBClassifierTests.cpp Models_NBClassifier )
#SHARK_ADD_TEST( Models/OnlineRNNet.cpp Models_OnlineRNNet )
SHARK_ADD_TEST( Models/RNNet.cpp Models_RNNet )
SHARK_ADD_TEST( Models/CMAC.cpp Models_CMAC )

SHARK_ADD_TEST( Models/SigmoidModel.cpp Models_SigmoidModel )
SHARK_ADD_TEST( Models/Softmax.cpp Models_Softmax )
SHARK_ADD_TEST( Models/SoftNearestNeighborClassifier.cpp Models_SoftNearestNeighborClassifier )
SHARK_ADD_TEST( Models/Kernels/KernelExpansion.cpp Models_KernelExpansion )
SHARK_ADD_TEST( Models/NearestNeighborRegression.cpp Models_NearestNeighborRegression )
SHARK_ADD_TEST( Models/OneVersusOneClassifier.cpp Models_OneVersusOneClassifier )

#Kernels


#SHARK_ADD_TEST( Models/Kernels/KernelFunction.cpp Models_KernelFunction )
SHARK_ADD_TEST( Models/Kernels/GaussianRbfKernel.cpp Models_GaussianKernel )
SHARK_ADD_TEST( Models/Kernels/LinearKernel.cpp Models_LinearKernel )
SHARK_ADD_TEST( Models/Kernels/MonomialKernel.cpp Models_MonomialKernel )
SHARK_ADD_TEST( Models/Kernels/PolynomialKernel.cpp Models_PolynomialKernel )
SHARK_ADD_TEST( Models/Kernels/ScaledKernel.cpp Models_ScaledKernel )
SHARK_ADD_TEST( Models/Kernels/WeightedSumKernel.cpp Models_WeightedSumKernel )
SHARK_ADD_TEST( Models/Kernels/ProductKernel.cpp Models_ProductKernel )
SHARK_ADD_TEST( Models/Kernels/ArdKernel.cpp Models_ArdKernel )
# SHARK_ADD_TEST( Models/Kernels/MklKernel.cpp Models_MklKernel )   # error with gcc 4.2.1
SHARK_ADD_TEST( Models/Kernels/SubrangeKernel.cpp Models_SubrangeKernel )
SHARK_ADD_TEST( Models/Kernels/DiscreteKernel.cpp Models_DiscreteKernel )
# SHARK_ADD_TEST( Models/Kernels/MultiTaskKernel.cpp Models_MultiTaskKernel )   # error with gcc 4.2.1

#KernelMethods
SHARK_ADD_TEST( Models/Kernels/KernelHelpers.cpp Models_KernelHelpers )
SHARK_ADD_TEST( Models/Kernels/KernelNearestNeighborClassifier.cpp Models_KernelNearestNeighborClassifier )
SHARK_ADD_TEST( Models/Kernels/KernelNearestNeighborRegression.cpp Models_KernelNearestNeighborRegression )
SHARK_ADD_TEST( Models/Kernels/EvalSkipMissingFeaturesTests.cpp Models_EvalSkipMissingFeatures )
SHARK_ADD_TEST( Models/Kernels/MissingFeaturesKernelExpansionTests.cpp Models_MissingFeaturesKernelExpansion )
SHARK_ADD_TEST( Models/Kernels/CSvmDerivative.cpp Models_CSvmDerivative )

#Trees
SHARK_ADD_TEST( Models/RFClassifier.cpp Models_RFClassifier )
SHARK_ADD_TEST( Models/CARTClassifier.cpp Models_CARTClassifier )

# Core tests
SHARK_ADD_TEST( Core/ScopedHandleTests.cpp ScopedHandleTests )

#Data Tests
SHARK_ADD_TEST( Data/Csv.cpp Data_Csv )
SHARK_ADD_TEST( Data/CVDatasetTools.cpp Data_CVDatasetTools )
#SHARK_ADD_TEST( Data/Dataset.cpp Data_Dataset )
SHARK_ADD_TEST( Data/DataView.cpp Data_DataView )
IF (HDF5_FOUND)
    SHARK_ADD_TEST( Data/HDF5Tests.cpp Data_HDF5 )
ENDIF(HDF5_FOUND)
SHARK_ADD_TEST( Data/Libsvm.cpp Data_Libsvm )
# SHARK_ADD_TEST( Data/MKLBatchInterface.cpp Data_MKLBatchInterface )   # failes with gcc 4.2.1
SHARK_ADD_TEST( Data/PrecomputedMatrix.cpp Data_PrecomputedMatrix )

#Objective Functions
SHARK_ADD_TEST( ObjectiveFunctions/ErrorFunction.cpp ObjFunct_ErrorFunction )
SHARK_ADD_TEST( ObjectiveFunctions/SparseFFNetError.cpp ObjFunct_SparseFFNetError )
SHARK_ADD_TEST( ObjectiveFunctions/NoisyErrorFunction.cpp ObjFunct_NoisyErrorFunction )
SHARK_ADD_TEST( ObjectiveFunctions/CrossValidation.cpp ObjFunct_CrossValidation )
SHARK_ADD_TEST( ObjectiveFunctions/Benchmarks.cpp ObjFunct_Benchmarks )
SHARK_ADD_TEST( ObjectiveFunctions/KernelTargetAlignment.cpp ObjFunct_KernelTargetAlignment )
SHARK_ADD_TEST( ObjectiveFunctions/RadiusMarginQuotient.cpp ObjFunct_RadiusMarginQuotient )
SHARK_ADD_TEST( ObjectiveFunctions/LooError.cpp ObjFunct_LooError )
SHARK_ADD_TEST( ObjectiveFunctions/LooErrorCSvm.cpp ObjFunct_LooErrorCSvm )
SHARK_ADD_TEST( ObjectiveFunctions/SvmLogisticInterpretation.cpp ObjFunct_SvmLogisticInterpretation )

#Objective Functions/Loss
SHARK_ADD_TEST( ObjectiveFunctions/CrossEntropy.cpp ObjFunct_CrossEntropy )
SHARK_ADD_TEST( ObjectiveFunctions/DenoisingAutoencoderError.cpp ObjFunct_DenoisingAutoencoderError )
SHARK_ADD_TEST( ObjectiveFunctions/SquaredLoss.cpp ObjFunct_SquaredLoss )
SHARK_ADD_TEST( ObjectiveFunctions/NegativeClassificationLogLikelihood.cpp ObjFunct_NegativeClassificationLogLikelihood )
SHARK_ADD_TEST( ObjectiveFunctions/AbsoluteLoss.cpp ObjFunct_AbsoluteLoss )
SHARK_ADD_TEST( ObjectiveFunctions/AUC.cpp ObjFunct_AUC )
SHARK_ADD_TEST( ObjectiveFunctions/NegativeGaussianProcessEvidence.cpp ObjFunct_NegativeGaussianProcessEvidence )

SHARK_ADD_TEST( Rng/Rng.cpp Rng_Distributions )


#RBM
SHARK_ADD_TEST( RBM/BinaryLayer.cpp RBM_BinaryLayer)
SHARK_ADD_TEST( RBM/GaussianLayer.cpp RBM_GaussianLayer)
SHARK_ADD_TEST( RBM/TruncatedExponentialLayer.cpp RBM_TruncatedExponentialLayer)

SHARK_ADD_TEST( RBM/MarkovChain.cpp RBM_MarkovChain)
SHARK_ADD_TEST( RBM/GibbsOperator.cpp RBM_GibbsOperator)

SHARK_ADD_TEST( RBM/Energy.cpp RBM_Energy)
SHARK_ADD_TEST( RBM/AverageEnergyGradient.cpp RBM_AverageEnergyGradient)
SHARK_ADD_TEST( RBM/Analytics.cpp RBM_Analytics)

SHARK_ADD_TEST( RBM/ExactGradient.cpp RBM_ExactGradient)
SHARK_ADD_TEST( RBM/ContrastiveDivergence.cpp RBM_ContrastiveDivergence)
# SHARK_ADD_TEST( RBM/TemperedMarkovChain.cpp RBM_TemperedMarkovChain)   # error with gcc 4.2.1

# SHARK_ADD_TEST( RBM/ParallelTemperingTraining.cpp RBM_PTTraining)   # error with gcc 4.2.1
SHARK_ADD_TEST( RBM/PCDTraining.cpp RBM_PCDTraining)
SHARK_ADD_TEST( RBM/ContrastiveDivergenceTraining.cpp RBM_ContrastiveDivergenceTraining)
SHARK_ADD_TEST( RBM/ExactGradientTraining.cpp RBM_ExactGradientTraining)

# Networking
IF( OPT_ENABLE_NETWORKING )
     SHARK_ADD_TEST( Network/HttpServer.cpp Network_Http_Server )
     SHARK_ADD_TEST( Network/HttpClient.cpp Network_Http_Client )
ENDIF( OPT_ENABLE_NETWORKING )

# Copy test file
ADD_CUSTOM_COMMAND(
    TARGET Data_HDF5
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Test/test_data
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Test/test_data ${CMAKE_BINARY_DIR}/Test/test_data
)
# Create output dir
ADD_CUSTOM_COMMAND(
    TARGET Data_Csv
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Test/test_output
)
ADD_CUSTOM_COMMAND(
    TARGET Data_PrecomputedMatrix
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Test/test_output
)