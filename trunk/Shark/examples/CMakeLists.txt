PROJECT( shark )



CMAKE_MINIMUM_REQUIRED( VERSION 2.4 )

OPTION( SHARK_OPT_ENABLE_QT_EXAMPLES "" OFF )

CMAKE_POLICY( SET CMP0003 OLD )

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../include )

LINK_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../lib )

SET( meanvar_SRC
	Array/mean-var.cpp
)
SET( ackleyES_SRC
	EALib/ackleyES.cpp
)
SET( paraboloidCMA_SRC
	EALib/paraboloidCMA.cpp
)
SET( paraboloidElitistCMA_SRC
	EALib/paraboloidElitistCMA.cpp
)
SET( countingOnes_SRC
	EALib/countingOnes.cpp
)
SET( integerES_SRC
	EALib/integerES.cpp
)
SET( sphereGA_SRC
	EALib/sphereGA.cpp
)
SET( steadyState_SRC
	EALib/steadyState.cpp
)
SET( pvmSphereGA_SRC
	EALib/pvmSphereGA.cpp
)
SET( TSPGA_SRC
	EALib/TSP_GA.cpp
)
SET( SchwefelEllipsoidCMSA_SRC
	EALib/SchwefelEllipsoidCMSA.cpp
)
SET( FileUtilSimple_SRC
	FileUtil/FileUtilSimple.cpp
)
SET( FileUtilClass_SRC
	FileUtil/FileUtilClass.cpp
)
SET( sugeno_SRC
	Fuzzy/sugeno.cpp
)
SET( ranktest_SRC
	LinAlg/rank_test.cpp
)
SET( rankdecomptest_SRC
	LinAlg/rank_decomp_test.cpp
)
SET( svdtest_SRC
	LinAlg/svd_test.cpp
)
SET( svdranktest_SRC
	LinAlg/svdrank_test.cpp
)
SET( svdsorttest_SRC
	LinAlg/svdsort_test.cpp
)
SET( eigensymmtest_SRC
	LinAlg/eigensymm_test.cpp
)
SET( eigensymmJacobitest_SRC
	LinAlg/eigensymmJacobi_test.cpp
)
SET( eigensymmJacobi2test_SRC
	LinAlg/eigensymmJacobi2_test.cpp
)
SET( eigensorttest_SRC
	LinAlg/eigensort_test.cpp
)
SET( eigenerrtest_SRC
	LinAlg/eigenerr_test.cpp
)
SET( ginversematrix_SRC
	LinAlg/g_inverse_matrix.cpp
)
SET( detsymmtest_SRC
	LinAlg/detsymm_test.cpp
)
SET( lnsrchtest_SRC
	LinAlg/lnsrch_test.cpp
)
SET( cblnsrchtest_SRC
	LinAlg/cblnsrch_test.cpp
)
SET( linmintest_SRC
	LinAlg/linmin_test.cpp
)
SET( dlinmintest_SRC
	LinAlg/dlinmin_test.cpp
)
SET( linalgsimpletest_SRC
	LinAlg/linalg_simple_test.cpp
)
SET( covarcorrcoeftest_SRC
	LinAlg/covar_corrcoef_test.cpp
)
SET( ffttest_SRC
	LinAlg/fft_test.cpp
)
SET( MOGA_SRC
	MOO-EALib/MOGA.cpp
)
SET( NSGA2_SRC
	MOO-EALib/NSGA2.cpp
)
SET( PAES_SRC
	MOO-EALib/PAES.cpp
)
SET( VEGA_SRC
	MOO-EALib/VEGA.cpp
)
SET( NSGA2example_SRC
	MOO-EALib/NSGA2example.cpp
)
SET( RMMEDAtest_SRC
	MOO-EALib/RMMEDAtest.cpp
)
SET( realcodedNSGA2_SRC
	MOO-EALib/realcoded-NSGA2.cpp
)
SET( fonMOCMA_SRC
	MOO-EALib/fonMO-CMA.cpp
)
SET( fonSteadyStateMOCMA_SRC
	MOO-EALib/fonSteadyStateMO-CMA.cpp
)
SET( HypervolumeBenchmark_SRC
	MOO-EALib/HypervolumeBenchmark.cpp
)
SET( mgexample_SRC
	Mixture/mg-example.cpp
)
SET( rbfnexample_SRC
	Mixture/rbfn-example.cpp
)
SET( simpleRBFNet_SRC
	ReClaM/simpleRBFNet.cpp
)
SET( simpleFFNet_SRC
	ReClaM/simpleFFNet.cpp
)
SET( simpleRNNet_SRC
	ReClaM/simpleRNNet.cpp
)
SET( simpleFFNetSource_SRC
	ReClaM/simpleFFNetSource.cpp
)
SET( simpleMSERNNet_SRC
	ReClaM/simpleMSERNNet.cpp
)
SET( SVMclassification_SRC
	ReClaM/SVMclassification.cpp
)

SET( SVMregression_SRC
	ReClaM/SVMregression.cpp
)

SET( SVMregressiongnuplot_SRC
	ReClaM/SVMregression-gnuplot.cpp
)

SET( SVMclassificationgnuplot_SRC
	ReClaM/SVMclassification-gnuplot.cpp
)

SET( KNN_SRC
	ReClaM/KNN.cpp
)
SET( KM_SRC
	ReClaM/KM.cpp
)
SET( CrossValidation_SRC
	ReClaM/CrossValidation.cpp
)
SET( KernelOptimization_SRC
	ReClaM/KernelOptimization.cpp
)
SET( SvmApproximationExample_SRC
	ReClaM/SvmApproximationExample.cpp
)
SET( LinearClassifierTest_SRC
	ReClaM/LinearClassifierTest.cpp
)
SET( PCAtest_SRC
	ReClaM/PCAtest.cpp
)
SET( FisherLDA_SRC
	ReClaM/FisherLDA.cpp
)
SET( LinearRegressionTest_SRC
	ReClaM/LinearRegressionTest.cpp
)
SET( KalmanFilterTest_SRC
	ReClaM/KalmanFilterTest.cpp
)
SET( McSvm_SRC
	ReClaM/McSvm.cpp
)
SET( RngTest_SRC
	Rng/RngTest.cpp
)
SET( RngTest2_SRC
	Rng/RngTest2.cpp
)
SET( RngTestSimple_SRC
	Rng/RngTestSimple.cpp
)
SET( embeddingTest_SRC
	TimeSeries/embeddingTest.cpp
)
SET( lorenz84_SRC
	TimeSeries/lorenz84.cpp
)

#ADD_EXECUTABLE( Array/mean-var ${meanvar_SRC} )
#TARGET_LINK_LIBRARIES( Array/mean-var shark )
#INSTALL( TARGETS Array/mean-var
#RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Array )

ADD_EXECUTABLE( EALib/ackleyES ${ackleyES_SRC} )
TARGET_LINK_LIBRARIES( EALib/ackleyES shark )
INSTALL( TARGETS EALib/ackleyES
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( EALib/paraboloidCMA ${paraboloidCMA_SRC} )
TARGET_LINK_LIBRARIES( EALib/paraboloidCMA shark )
INSTALL( TARGETS EALib/paraboloidCMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( EALib/paraboloidElitistCMA ${paraboloidElitistCMA_SRC} )
TARGET_LINK_LIBRARIES( EALib/paraboloidElitistCMA shark )
INSTALL( TARGETS EALib/paraboloidElitistCMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( EALib/countingOnes ${countingOnes_SRC} )
TARGET_LINK_LIBRARIES( EALib/countingOnes shark )
INSTALL( TARGETS EALib/countingOnes
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( EALib/integerES ${integerES_SRC} )
TARGET_LINK_LIBRARIES( EALib/integerES shark )
INSTALL( TARGETS EALib/integerES
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( EALib/sphereGA ${sphereGA_SRC} )
TARGET_LINK_LIBRARIES( EALib/sphereGA shark )
INSTALL( TARGETS EALib/sphereGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( EALib/steadyState ${steadyState_SRC} )
TARGET_LINK_LIBRARIES( EALib/steadyState shark )
INSTALL( TARGETS EALib/steadyState
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( EALib/pvmSphereGA ${pvmSphereGA_SRC} )
TARGET_LINK_LIBRARIES( EALib/pvmSphereGA shark )
INSTALL( TARGETS EALib/pvmSphereGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( EALib/TSP_GA ${TSPGA_SRC} )
TARGET_LINK_LIBRARIES( EALib/TSP_GA shark )
INSTALL( TARGETS EALib/TSP_GA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( EALib/SchwefelEllipsoidCMSA ${SchwefelEllipsoidCMSA_SRC} )
TARGET_LINK_LIBRARIES( EALib/SchwefelEllipsoidCMSA shark )
INSTALL( TARGETS EALib/SchwefelEllipsoidCMSA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( FileUtil/FileUtilSimple ${FileUtilSimple_SRC} )
TARGET_LINK_LIBRARIES( FileUtil/FileUtilSimple shark )
INSTALL( TARGETS FileUtil/FileUtilSimple
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/FileUtil )

ADD_EXECUTABLE( FileUtil/FileUtilClass ${FileUtilClass_SRC} )
TARGET_LINK_LIBRARIES( FileUtil/FileUtilClass shark )
INSTALL( TARGETS FileUtil/FileUtilClass
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/FileUtil )

ADD_EXECUTABLE( Fuzzy/sugeno ${sugeno_SRC} )
TARGET_LINK_LIBRARIES( Fuzzy/sugeno shark )
INSTALL( TARGETS Fuzzy/sugeno
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Fuzzy )

ADD_EXECUTABLE( LinAlg/rank_test ${ranktest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/rank_test shark )
INSTALL( TARGETS LinAlg/rank_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/rank_decomp_test ${rankdecomptest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/rank_decomp_test shark )
INSTALL( TARGETS LinAlg/rank_decomp_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/svd_test ${svdtest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/svd_test shark )
INSTALL( TARGETS LinAlg/svd_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/svdrank_test ${svdranktest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/svdrank_test shark )
INSTALL( TARGETS LinAlg/svdrank_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/svdsort_test ${svdsorttest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/svdsort_test shark )
INSTALL( TARGETS LinAlg/svdsort_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/eigensymm_test ${eigensymmtest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/eigensymm_test shark )
INSTALL( TARGETS LinAlg/eigensymm_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/eigensymmJacobi_test ${eigensymmJacobitest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/eigensymmJacobi_test shark )
INSTALL( TARGETS LinAlg/eigensymmJacobi_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/eigensymmJacobi2_test ${eigensymmJacobi2test_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/eigensymmJacobi2_test shark )
INSTALL( TARGETS LinAlg/eigensymmJacobi2_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/eigensort_test ${eigensorttest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/eigensort_test shark )
INSTALL( TARGETS LinAlg/eigensort_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/eigenerr_test ${eigenerrtest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/eigenerr_test shark )
INSTALL( TARGETS LinAlg/eigenerr_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/g_inverse_matrix ${ginversematrix_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/g_inverse_matrix shark )
INSTALL( TARGETS LinAlg/g_inverse_matrix
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/detsymm_test ${detsymmtest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/detsymm_test shark )
INSTALL( TARGETS LinAlg/detsymm_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/lnsrch_test ${lnsrchtest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/lnsrch_test shark )
INSTALL( TARGETS LinAlg/lnsrch_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/cblnsrch_test ${cblnsrchtest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/cblnsrch_test shark )
INSTALL( TARGETS LinAlg/cblnsrch_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/linmin_test ${linmintest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/linmin_test shark )
INSTALL( TARGETS LinAlg/linmin_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/dlinmin_test ${dlinmintest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/dlinmin_test shark )
INSTALL( TARGETS LinAlg/dlinmin_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/linalg_simple_test ${linalgsimpletest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/linalg_simple_test shark )
INSTALL( TARGETS LinAlg/linalg_simple_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/covar_corrcoef_test ${covarcorrcoeftest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/covar_corrcoef_test shark )
INSTALL( TARGETS LinAlg/covar_corrcoef_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( LinAlg/fft_test ${ffttest_SRC} )
TARGET_LINK_LIBRARIES( LinAlg/fft_test shark )
INSTALL( TARGETS LinAlg/fft_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( MOO-EALib/MOGA ${MOGA_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/MOGA shark )
INSTALL( TARGETS MOO-EALib/MOGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( MOO-EALib/NSGA2 ${NSGA2_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/NSGA2 shark )
INSTALL( TARGETS MOO-EALib/NSGA2
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( MOO-EALib/PAES ${PAES_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/PAES shark )
INSTALL( TARGETS MOO-EALib/PAES
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( MOO-EALib/VEGA ${VEGA_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/VEGA shark )
INSTALL( TARGETS MOO-EALib/VEGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( MOO-EALib/NSGA2example ${NSGA2example_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/NSGA2example shark )
INSTALL( TARGETS MOO-EALib/NSGA2example
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( MOO-EALib/RMMEDAtest ${RMMEDAtest_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/RMMEDAtest shark )
INSTALL( TARGETS MOO-EALib/RMMEDAtest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( MOO-EALib/realcoded-NSGA2 ${realcodedNSGA2_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/realcoded-NSGA2 shark )
INSTALL( TARGETS MOO-EALib/realcoded-NSGA2
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( MOO-EALib/fonMO-CMA ${fonMOCMA_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/fonMO-CMA shark )
INSTALL( TARGETS MOO-EALib/fonMO-CMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( MOO-EALib/fonSteadyStateMO-CMA ${fonSteadyStateMOCMA_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/fonSteadyStateMO-CMA shark )
INSTALL( TARGETS MOO-EALib/fonSteadyStateMO-CMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( MOO-EALib/HypervolumeBenchmark ${HypervolumeBenchmark_SRC} )
TARGET_LINK_LIBRARIES( MOO-EALib/HypervolumeBenchmark shark )
INSTALL( TARGETS MOO-EALib/HypervolumeBenchmark
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( Mixture/mg-example ${mgexample_SRC} )
TARGET_LINK_LIBRARIES( Mixture/mg-example shark )
INSTALL( TARGETS Mixture/mg-example
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Mixture )

ADD_EXECUTABLE( Mixture/rbfn-example ${rbfnexample_SRC} )
TARGET_LINK_LIBRARIES( Mixture/rbfn-example shark )
INSTALL( TARGETS Mixture/rbfn-example
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Mixture )

ADD_EXECUTABLE( ReClaM/simpleRBFNet ${simpleRBFNet_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/simpleRBFNet shark )
INSTALL( TARGETS ReClaM/simpleRBFNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/simpleFFNet ${simpleFFNet_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/simpleFFNet shark )
INSTALL( TARGETS ReClaM/simpleFFNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/simpleRNNet ${simpleRNNet_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/simpleRNNet shark )
INSTALL( TARGETS ReClaM/simpleRNNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/simpleFFNetSource ${simpleFFNetSource_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/simpleFFNetSource shark )
INSTALL( TARGETS ReClaM/simpleFFNetSource
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/simpleMSERNNet ${simpleMSERNNet_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/simpleMSERNNet shark )
INSTALL( TARGETS ReClaM/simpleMSERNNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/SVMclassification ${SVMclassification_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/SVMclassification shark )
INSTALL( TARGETS ReClaM/SVMclassification
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

IF( LINUX )
ADD_EXECUTABLE( ReClaM/SVMclassification-gnuplot ${SVMclassificationgnuplot_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/SVMclassification-gnuplot shark )
INSTALL( TARGETS ReClaM/SVMclassification-gnuplot
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )
ENDIF( LINUX )

ADD_EXECUTABLE( ReClaM/SVMregression ${SVMregression_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/SVMregression shark )
INSTALL( TARGETS ReClaM/SVMregression
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

IF( LINUX )
ADD_EXECUTABLE( ReClaM/SVMregression-gnuplot ${SVMregressiongnuplot_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/SVMregression-gnuplot shark )
INSTALL( TARGETS ReClaM/SVMregression-gnuplot
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )
ENDIF( LINUX )

ADD_EXECUTABLE( ReClaM/KNN ${KNN_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/KNN shark )
INSTALL( TARGETS ReClaM/KNN
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/KM ${KM_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/KM shark )
INSTALL( TARGETS ReClaM/KM
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/CrossValidation ${CrossValidation_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/CrossValidation shark )
INSTALL( TARGETS ReClaM/CrossValidation
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/KernelOptimization ${KernelOptimization_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/KernelOptimization shark )
INSTALL( TARGETS ReClaM/KernelOptimization
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/SvmApproximationExample ${SvmApproximationExample_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/SvmApproximationExample shark )
INSTALL( TARGETS ReClaM/SvmApproximationExample
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/LinearClassifierTest ${LinearClassifierTest_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/LinearClassifierTest shark )
INSTALL( TARGETS ReClaM/LinearClassifierTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/PCAtest ${PCAtest_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/PCAtest shark )
INSTALL( TARGETS ReClaM/PCAtest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/FisherLDA ${FisherLDA_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/FisherLDA shark )
INSTALL( TARGETS ReClaM/FisherLDA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/LinearRegressionTest ${LinearRegressionTest_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/LinearRegressionTest shark )
INSTALL( TARGETS ReClaM/LinearRegressionTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/KalmanFilterTest ${KalmanFilterTest_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/KalmanFilterTest shark )
INSTALL( TARGETS ReClaM/KalmanFilterTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( ReClaM/McSvm ${McSvm_SRC} )
TARGET_LINK_LIBRARIES( ReClaM/McSvm shark )
INSTALL( TARGETS ReClaM/McSvm
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( Rng/RngTest ${RngTest_SRC} )
TARGET_LINK_LIBRARIES( Rng/RngTest shark )
INSTALL( TARGETS Rng/RngTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Rng )

ADD_EXECUTABLE( Rng/RngTest2 ${RngTest2_SRC} )
TARGET_LINK_LIBRARIES( Rng/RngTest2 shark )
INSTALL( TARGETS Rng/RngTest2
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Rng )

ADD_EXECUTABLE( Rng/RngTestSimple ${RngTestSimple_SRC} )
TARGET_LINK_LIBRARIES( Rng/RngTestSimple shark )
INSTALL( TARGETS Rng/RngTestSimple
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Rng )

ADD_EXECUTABLE( TimeSeries/embeddingTest ${embeddingTest_SRC} )
TARGET_LINK_LIBRARIES( TimeSeries/embeddingTest shark )
INSTALL( TARGETS TimeSeries/embeddingTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/TimeSeries )

ADD_EXECUTABLE( TimeSeries/lorenz84 ${lorenz84_SRC} )
TARGET_LINK_LIBRARIES( TimeSeries/lorenz84 shark )
INSTALL( TARGETS TimeSeries/lorenz84
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/TimeSeries )

IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.6)
INSTALL( DIRECTORY ./Array
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE 
PATTERN "*CMake*" EXCLUDE
)

INSTALL( DIRECTORY ./EALib
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE
PATTERN "*CMake*" EXCLUDE
)

INSTALL( DIRECTORY ./FileUtil
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE 
PATTERN "*CMake*" EXCLUDE
)

INSTALL( DIRECTORY ./Fuzzy
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE 
PATTERN "*CMake*" EXCLUDE
)

INSTALL( DIRECTORY ./LinAlg
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE 
PATTERN "*CMake*" EXCLUDE
)

INSTALL( DIRECTORY ./MOO-EALib
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE 
PATTERN "*CMake*" EXCLUDE
)

INSTALL( DIRECTORY ./Mixture
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE 
PATTERN "*CMake*" EXCLUDE
)

INSTALL( DIRECTORY ./ReClaM
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE 
PATTERN "*CMake*" EXCLUDE
)

INSTALL( DIRECTORY ./Rng
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE 
PATTERN "*CMake*" EXCLUDE
)

INSTALL( DIRECTORY ./TimeSeries
DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}
FILES_MATCHING PATTERN "*.h" PATTERN "*.cpp"
PATTERN ".svn" EXCLUDE 
PATTERN "*CMake*" EXCLUDE
)
ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.6)

IF( SHARK_OPT_ENABLE_QT_EXAMPLES )

	FIND_PACKAGE( Qt4 REQUIRED )
	FIND_PATH( Qwt_INCLUDE_DIR qwt HINTS "/sw/lib/qt4-mac/include" )
	FIND_LIBRARY( Qwt_LIBRARY qwt HINTS "/sw/lib/qt4-mac/lib" )

	IF( Qwt_INCLUDE_DIR )
	INCLUDE_DIRECTORIES(
		../include

		${QT_INCLUDE_DIR}
		${QT_QTGUI_INCLUDE_DIR}
		${QT_QTCORE_INCLUDE_DIR}

		${Qwt_INCLUDE_DIR}	
	)
	ELSE()
	INCLUDE_DIRECTORIES(
		../include

		${QT_INCLUDE_DIR}
		${QT_QTGUI_INCLUDE_DIR}
		${QT_QTCORE_INCLUDE_DIR}
	)
	ENDIF( Qwt_INCLUDE_DIR )

	SET ( shark_QT_LIBS
		${QT_QTCORE_LIBRARY}
		${QT_QTGUI_LIBRARY}
	)

	qt4_wrap_cpp( Regression_MOCS ReClaM/QT/regression.h )
	SET( Regression_SRCS ReClaM/QT/regression.cpp ${Regression_MOCS} )
	ADD_EXECUTABLE( ReClaM/QT/Regression ${Regression_SRCS} )
	TARGET_LINK_LIBRARIES( ReClaM/QT/Regression shark ${shark_QT_LIBS})
	INSTALL( TARGETS ReClaM/QT/Regression
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM/QT )

	qt4_wrap_cpp( Classification_MOCS ReClaM/QT/classification.h )
	SET( Classification_SRCS ReClaM/QT/classification.cpp ${Classification_MOCS} )
	ADD_EXECUTABLE( ReClaM/QT/Classification ${Classification_SRCS} )
	TARGET_LINK_LIBRARIES( ReClaM/QT/Classification shark ${shark_QT_LIBS})
	INSTALL( TARGETS ReClaM/QT/Classification
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM/QT )

	IF( Qwt_LIBRARY )
	qt4_wrap_cpp( GaussianProcess_MOCS ReClaM/QT/gaussian_process.h )
	SET( GaussianProcess_SRCS ReClaM/QT/gaussian_process.cpp ${GaussianProcess_MOCS} )
	ADD_EXECUTABLE( ReClaM/QT/GaussianProcess ${GaussianProcess_SRCS} )
	TARGET_LINK_LIBRARIES( ReClaM/QT/GaussianProcess shark ${shark_QT_LIBS} ${Qwt_LIBRARY} )
	INSTALL( TARGETS ReClaM/QT/GaussianProcess
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM/QT )
	ENDIF( Qwt_LIBRARY )

	qt4_wrap_cpp( EA_MOCS
		EALib/QT/ComparisonView.h
		EALib/QT/ConfigCtrl.h
		EALib/QT/ControlWidget.h
		EALib/QT/Experiment.h
		EALib/QT/ExperimentPropertiesWidget.h
		EALib/QT/LandscapeView2D.h
		EALib/QT/LandscapeView3D.h
		EALib/QT/MainWidget.h
		EALib/QT/MoFitnessView.h
		EALib/QT/StatusWidget.h
		EALib/QT/TestWidget.h
		EALib/QT/ValueView.h
		EALib/QT/View.h
		EALib/QT/Wizard.h
	)
	SET( EA_SRCS 
		${EA_MOCS} 
		EALib/QT/Canyon.cpp
		EALib/QT/ComparisonView.cpp
		EALib/QT/config.cpp
		EALib/QT/ConfigCtrl.cpp
		EALib/QT/ControlWidget.cpp
		EALib/QT/demo.cpp
		EALib/QT/Experiment.cpp
		EALib/QT/ExperimentPropertiesWidget.cpp
		EALib/QT/LandscapeView2D.cpp
		EALib/QT/LandscapeView3D.cpp
		EALib/QT/MainWidget.cpp
		EALib/QT/MoFitnessView.cpp
		EALib/QT/Optimization.cpp
		EALib/QT/StatusWidget.cpp
		EALib/QT/TestWidget.cpp
		EALib/QT/UTest.cpp
		EALib/QT/ValueView.cpp
		EALib/QT/View.cpp
		EALib/QT/Wizard.cpp
	)
	ADD_EXECUTABLE( EALib/QT/EA ${EA_SRCS} )
	TARGET_LINK_LIBRARIES( EALib/QT/EA shark ${shark_QT_LIBS} )
	INSTALL( TARGETS EALib/QT/EA
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib/QT )

ENDIF( SHARK_OPT_ENABLE_QT_EXAMPLES )
