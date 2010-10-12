PROJECT( shark )

CMAKE_MINIMUM_REQUIRED( VERSION 2.4 )

OPTION( ENABLE_QT_EXAMPLES "" OFF )

INCLUDE_DIRECTORIES( ../include )

LINK_DIRECTORIES( ../lib )

SET( meanvar_SRC
	../examples/Array/mean-var.cpp
)
SET( ackleyES_SRC
	../examples/EALib/ackleyES.cpp
)
SET( paraboloidCMA_SRC
	../examples/EALib/paraboloidCMA.cpp
)
SET( paraboloidElitistCMA_SRC
	../examples/EALib/paraboloidElitistCMA.cpp
)
SET( countingOnes_SRC
	../examples/EALib/countingOnes.cpp
)
SET( integerES_SRC
	../examples/EALib/integerES.cpp
)
SET( sphereGA_SRC
	../examples/EALib/sphereGA.cpp
)
SET( steadyState_SRC
	../examples/EALib/steadyState.cpp
)
SET( pvmSphereGA_SRC
	../examples/EALib/pvmSphereGA.cpp
)
SET( TSPGA_SRC
	../examples/EALib/TSP_GA.cpp
)
SET( SchwefelEllipsoidCMSA_SRC
	../examples/EALib/SchwefelEllipsoidCMSA.cpp
)
SET( FileUtilSimple_SRC
	../examples/FileUtil/FileUtilSimple.cpp
)
SET( FileUtilClass_SRC
	../examples/FileUtil/FileUtilClass.cpp
)
SET( sugeno_SRC
	../examples/Fuzzy/sugeno.cpp
)
SET( ranktest_SRC
	../examples/LinAlg/rank_test.cpp
)
SET( rankdecomptest_SRC
	../examples/LinAlg/rank_decomp_test.cpp
)
SET( svdtest_SRC
	../examples/LinAlg/svd_test.cpp
)
SET( svdranktest_SRC
	../examples/LinAlg/svdrank_test.cpp
)
SET( svdsorttest_SRC
	../examples/LinAlg/svdsort_test.cpp
)
SET( eigensymmtest_SRC
	../examples/LinAlg/eigensymm_test.cpp
)
SET( eigensymmJacobitest_SRC
	../examples/LinAlg/eigensymmJacobi_test.cpp
)
SET( eigensymmJacobi2test_SRC
	../examples/LinAlg/eigensymmJacobi2_test.cpp
)
SET( eigensorttest_SRC
	../examples/LinAlg/eigensort_test.cpp
)
SET( eigenerrtest_SRC
	../examples/LinAlg/eigenerr_test.cpp
)
SET( ginversematrix_SRC
	../examples/LinAlg/g_inverse_matrix.cpp
)
SET( detsymmtest_SRC
	../examples/LinAlg/detsymm_test.cpp
)
SET( lnsrchtest_SRC
	../examples/LinAlg/lnsrch_test.cpp
)
SET( cblnsrchtest_SRC
	../examples/LinAlg/cblnsrch_test.cpp
)
SET( linmintest_SRC
	../examples/LinAlg/linmin_test.cpp
)
SET( dlinmintest_SRC
	../examples/LinAlg/dlinmin_test.cpp
)
SET( linalgsimpletest_SRC
	../examples/LinAlg/linalg_simple_test.cpp
)
SET( covarcorrcoeftest_SRC
	../examples/LinAlg/covar_corrcoef_test.cpp
)
SET( ffttest_SRC
	../examples/LinAlg/fft_test.cpp
)
SET( MOGA_SRC
	../examples/MOO-EALib/MOGA.cpp
)
SET( NSGA2_SRC
	../examples/MOO-EALib/NSGA2.cpp
)
SET( PAES_SRC
	../examples/MOO-EALib/PAES.cpp
)
SET( VEGA_SRC
	../examples/MOO-EALib/VEGA.cpp
)
SET( NSGA2example_SRC
	../examples/MOO-EALib/NSGA2example.cpp
)
SET( RMMEDAtest_SRC
	../examples/MOO-EALib/RMMEDAtest.cpp
)
SET( realcodedNSGA2_SRC
	../examples/MOO-EALib/realcoded-NSGA2.cpp
)
SET( fonMOCMA_SRC
	../examples/MOO-EALib/fonMO-CMA.cpp
)
SET( fonSteadyStateMOCMA_SRC
	../examples/MOO-EALib/fonSteadyStateMO-CMA.cpp
)
SET( HypervolumeBenchmark_SRC
	../examples/MOO-EALib/HypervolumeBenchmark.cpp
)
SET( mgexample_SRC
	../examples/Mixture/mg-example.cpp
)
SET( rbfnexample_SRC
	../examples/Mixture/rbfn-example.cpp
)
SET( simpleRBFNet_SRC
	../examples/ReClaM/simpleRBFNet.cpp
)
SET( simpleFFNet_SRC
	../examples/ReClaM/simpleFFNet.cpp
)
SET( simpleRNNet_SRC
	../examples/ReClaM/simpleRNNet.cpp
)
SET( simpleFFNetSource_SRC
	../examples/ReClaM/simpleFFNetSource.cpp
)
SET( simpleMSERNNet_SRC
	../examples/ReClaM/simpleMSERNNet.cpp
)
SET( SVMclassification_SRC
	../examples/ReClaM/SVMclassification.cpp
)

SET( SVMregression_SRC
	../examples/ReClaM/SVMregression.cpp
)

SET( SVMregressiongnuplot_SRC
	../examples/ReClaM/SVMregression-gnuplot.cpp
)

SET( SVMclassificationgnuplot_SRC
	../examples/ReClaM/SVMclassification-gnuplot.cpp
)

SET( KNN_SRC
	../examples/ReClaM/KNN.cpp
)
SET( KM_SRC
	../examples/ReClaM/KM.cpp
)
SET( CrossValidation_SRC
	../examples/ReClaM/CrossValidation.cpp
)
SET( KernelOptimization_SRC
	../examples/ReClaM/KernelOptimization.cpp
)
SET( SvmApproximationExample_SRC
	../examples/ReClaM/SvmApproximationExample.cpp
)
SET( LinearClassifierTest_SRC
	../examples/ReClaM/LinearClassifierTest.cpp
)
SET( PCAtest_SRC
	../examples/ReClaM/PCAtest.cpp
)
SET( FisherLDA_SRC
	../examples/ReClaM/FisherLDA.cpp
)
SET( LinearRegressionTest_SRC
	../examples/ReClaM/LinearRegressionTest.cpp
)
SET( KalmanFilterTest_SRC
	../examples/ReClaM/KalmanFilterTest.cpp
)
SET( McSvm_SRC
	../examples/ReClaM/McSvm.cpp
)
SET( RngTest_SRC
	../examples/Rng/RngTest.cpp
)
SET( RngTest2_SRC
	../examples/Rng/RngTest2.cpp
)
SET( RngTestSimple_SRC
	../examples/Rng/RngTestSimple.cpp
)
SET( embeddingTest_SRC
	../examples/TimeSeries/embeddingTest.cpp
)
SET( lorenz84_SRC
	../examples/TimeSeries/lorenz84.cpp
)

ADD_EXECUTABLE( .././examples/Array/mean-var ${meanvar_SRC} )
TARGET_LINK_LIBRARIES( .././examples/Array/mean-var shark )
INSTALL( TARGETS .././examples/Array/mean-var
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Array )

ADD_EXECUTABLE( .././examples/EALib/ackleyES ${ackleyES_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/ackleyES shark )
INSTALL( TARGETS .././examples/EALib/ackleyES
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/EALib/paraboloidCMA ${paraboloidCMA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/paraboloidCMA shark )
INSTALL( TARGETS .././examples/EALib/paraboloidCMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/EALib/paraboloidElitistCMA ${paraboloidElitistCMA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/paraboloidElitistCMA shark )
INSTALL( TARGETS .././examples/EALib/paraboloidElitistCMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/EALib/countingOnes ${countingOnes_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/countingOnes shark )
INSTALL( TARGETS .././examples/EALib/countingOnes
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/EALib/integerES ${integerES_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/integerES shark )
INSTALL( TARGETS .././examples/EALib/integerES
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/EALib/sphereGA ${sphereGA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/sphereGA shark )
INSTALL( TARGETS .././examples/EALib/sphereGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/EALib/steadyState ${steadyState_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/steadyState shark )
INSTALL( TARGETS .././examples/EALib/steadyState
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/EALib/pvmSphereGA ${pvmSphereGA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/pvmSphereGA shark )
INSTALL( TARGETS .././examples/EALib/pvmSphereGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/EALib/TSP_GA ${TSPGA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/TSP_GA shark )
INSTALL( TARGETS .././examples/EALib/TSP_GA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/EALib/SchwefelEllipsoidCMSA ${SchwefelEllipsoidCMSA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/EALib/SchwefelEllipsoidCMSA shark )
INSTALL( TARGETS .././examples/EALib/SchwefelEllipsoidCMSA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( .././examples/FileUtil/FileUtilSimple ${FileUtilSimple_SRC} )
TARGET_LINK_LIBRARIES( .././examples/FileUtil/FileUtilSimple shark )
INSTALL( TARGETS .././examples/FileUtil/FileUtilSimple
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/FileUtil )

ADD_EXECUTABLE( .././examples/FileUtil/FileUtilClass ${FileUtilClass_SRC} )
TARGET_LINK_LIBRARIES( .././examples/FileUtil/FileUtilClass shark )
INSTALL( TARGETS .././examples/FileUtil/FileUtilClass
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/FileUtil )

ADD_EXECUTABLE( .././examples/Fuzzy/sugeno ${sugeno_SRC} )
TARGET_LINK_LIBRARIES( .././examples/Fuzzy/sugeno shark )
INSTALL( TARGETS .././examples/Fuzzy/sugeno
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Fuzzy )

ADD_EXECUTABLE( .././examples/LinAlg/rank_test ${ranktest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/rank_test shark )
INSTALL( TARGETS .././examples/LinAlg/rank_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/rank_decomp_test ${rankdecomptest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/rank_decomp_test shark )
INSTALL( TARGETS .././examples/LinAlg/rank_decomp_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/svd_test ${svdtest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/svd_test shark )
INSTALL( TARGETS .././examples/LinAlg/svd_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/svdrank_test ${svdranktest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/svdrank_test shark )
INSTALL( TARGETS .././examples/LinAlg/svdrank_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/svdsort_test ${svdsorttest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/svdsort_test shark )
INSTALL( TARGETS .././examples/LinAlg/svdsort_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/eigensymm_test ${eigensymmtest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/eigensymm_test shark )
INSTALL( TARGETS .././examples/LinAlg/eigensymm_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/eigensymmJacobi_test ${eigensymmJacobitest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/eigensymmJacobi_test shark )
INSTALL( TARGETS .././examples/LinAlg/eigensymmJacobi_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/eigensymmJacobi2_test ${eigensymmJacobi2test_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/eigensymmJacobi2_test shark )
INSTALL( TARGETS .././examples/LinAlg/eigensymmJacobi2_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/eigensort_test ${eigensorttest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/eigensort_test shark )
INSTALL( TARGETS .././examples/LinAlg/eigensort_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/eigenerr_test ${eigenerrtest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/eigenerr_test shark )
INSTALL( TARGETS .././examples/LinAlg/eigenerr_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/g_inverse_matrix ${ginversematrix_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/g_inverse_matrix shark )
INSTALL( TARGETS .././examples/LinAlg/g_inverse_matrix
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/detsymm_test ${detsymmtest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/detsymm_test shark )
INSTALL( TARGETS .././examples/LinAlg/detsymm_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/lnsrch_test ${lnsrchtest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/lnsrch_test shark )
INSTALL( TARGETS .././examples/LinAlg/lnsrch_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/cblnsrch_test ${cblnsrchtest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/cblnsrch_test shark )
INSTALL( TARGETS .././examples/LinAlg/cblnsrch_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/linmin_test ${linmintest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/linmin_test shark )
INSTALL( TARGETS .././examples/LinAlg/linmin_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/dlinmin_test ${dlinmintest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/dlinmin_test shark )
INSTALL( TARGETS .././examples/LinAlg/dlinmin_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/linalg_simple_test ${linalgsimpletest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/linalg_simple_test shark )
INSTALL( TARGETS .././examples/LinAlg/linalg_simple_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/covar_corrcoef_test ${covarcorrcoeftest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/covar_corrcoef_test shark )
INSTALL( TARGETS .././examples/LinAlg/covar_corrcoef_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/LinAlg/fft_test ${ffttest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/LinAlg/fft_test shark )
INSTALL( TARGETS .././examples/LinAlg/fft_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( .././examples/MOO-EALib/MOGA ${MOGA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/MOGA shark )
INSTALL( TARGETS .././examples/MOO-EALib/MOGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/MOO-EALib/NSGA2 ${NSGA2_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/NSGA2 shark )
INSTALL( TARGETS .././examples/MOO-EALib/NSGA2
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/MOO-EALib/PAES ${PAES_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/PAES shark )
INSTALL( TARGETS .././examples/MOO-EALib/PAES
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/MOO-EALib/VEGA ${VEGA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/VEGA shark )
INSTALL( TARGETS .././examples/MOO-EALib/VEGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/MOO-EALib/NSGA2example ${NSGA2example_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/NSGA2example shark )
INSTALL( TARGETS .././examples/MOO-EALib/NSGA2example
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/MOO-EALib/RMMEDAtest ${RMMEDAtest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/RMMEDAtest shark )
INSTALL( TARGETS .././examples/MOO-EALib/RMMEDAtest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/MOO-EALib/realcoded-NSGA2 ${realcodedNSGA2_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/realcoded-NSGA2 shark )
INSTALL( TARGETS .././examples/MOO-EALib/realcoded-NSGA2
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/MOO-EALib/fonMO-CMA ${fonMOCMA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/fonMO-CMA shark )
INSTALL( TARGETS .././examples/MOO-EALib/fonMO-CMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/MOO-EALib/fonSteadyStateMO-CMA ${fonSteadyStateMOCMA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/fonSteadyStateMO-CMA shark )
INSTALL( TARGETS .././examples/MOO-EALib/fonSteadyStateMO-CMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/MOO-EALib/HypervolumeBenchmark ${HypervolumeBenchmark_SRC} )
TARGET_LINK_LIBRARIES( .././examples/MOO-EALib/HypervolumeBenchmark shark )
INSTALL( TARGETS .././examples/MOO-EALib/HypervolumeBenchmark
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( .././examples/Mixture/mg-example ${mgexample_SRC} )
TARGET_LINK_LIBRARIES( .././examples/Mixture/mg-example shark )
INSTALL( TARGETS .././examples/Mixture/mg-example
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Mixture )

ADD_EXECUTABLE( .././examples/Mixture/rbfn-example ${rbfnexample_SRC} )
TARGET_LINK_LIBRARIES( .././examples/Mixture/rbfn-example shark )
INSTALL( TARGETS .././examples/Mixture/rbfn-example
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Mixture )

ADD_EXECUTABLE( .././examples/ReClaM/simpleRBFNet ${simpleRBFNet_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/simpleRBFNet shark )
INSTALL( TARGETS .././examples/ReClaM/simpleRBFNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/simpleFFNet ${simpleFFNet_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/simpleFFNet shark )
INSTALL( TARGETS .././examples/ReClaM/simpleFFNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/simpleRNNet ${simpleRNNet_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/simpleRNNet shark )
INSTALL( TARGETS .././examples/ReClaM/simpleRNNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/simpleFFNetSource ${simpleFFNetSource_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/simpleFFNetSource shark )
INSTALL( TARGETS .././examples/ReClaM/simpleFFNetSource
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/simpleMSERNNet ${simpleMSERNNet_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/simpleMSERNNet shark )
INSTALL( TARGETS .././examples/ReClaM/simpleMSERNNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/SVMclassification ${SVMclassification_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/SVMclassification shark )
INSTALL( TARGETS .././examples/ReClaM/SVMclassification
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

IF( LINUX )
ADD_EXECUTABLE( .././examples/ReClaM/SVMclassification-gnuplot ${SVMclassificationgnuplot_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/SVMclassification-gnuplot shark )
INSTALL( TARGETS .././examples/ReClaM/SVMclassification-gnuplot
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )
ENDIF( LINUX )

ADD_EXECUTABLE( .././examples/ReClaM/SVMregression ${SVMregression_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/SVMregression shark )
INSTALL( TARGETS .././examples/ReClaM/SVMregression
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

IF( LINUX )
ADD_EXECUTABLE( .././examples/ReClaM/SVMregression-gnuplot ${SVMregressiongnuplot_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/SVMregression-gnuplot shark )
INSTALL( TARGETS .././examples/ReClaM/SVMregression-gnuplot
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )
ENDIF( LINUX )

ADD_EXECUTABLE( .././examples/ReClaM/KNN ${KNN_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/KNN shark )
INSTALL( TARGETS .././examples/ReClaM/KNN
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/KM ${KM_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/KM shark )
INSTALL( TARGETS .././examples/ReClaM/KM
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/CrossValidation ${CrossValidation_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/CrossValidation shark )
INSTALL( TARGETS .././examples/ReClaM/CrossValidation
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/KernelOptimization ${KernelOptimization_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/KernelOptimization shark )
INSTALL( TARGETS .././examples/ReClaM/KernelOptimization
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/SvmApproximationExample ${SvmApproximationExample_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/SvmApproximationExample shark )
INSTALL( TARGETS .././examples/ReClaM/SvmApproximationExample
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/LinearClassifierTest ${LinearClassifierTest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/LinearClassifierTest shark )
INSTALL( TARGETS .././examples/ReClaM/LinearClassifierTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/PCAtest ${PCAtest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/PCAtest shark )
INSTALL( TARGETS .././examples/ReClaM/PCAtest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/FisherLDA ${FisherLDA_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/FisherLDA shark )
INSTALL( TARGETS .././examples/ReClaM/FisherLDA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/LinearRegressionTest ${LinearRegressionTest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/LinearRegressionTest shark )
INSTALL( TARGETS .././examples/ReClaM/LinearRegressionTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/KalmanFilterTest ${KalmanFilterTest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/KalmanFilterTest shark )
INSTALL( TARGETS .././examples/ReClaM/KalmanFilterTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/ReClaM/McSvm ${McSvm_SRC} )
TARGET_LINK_LIBRARIES( .././examples/ReClaM/McSvm shark )
INSTALL( TARGETS .././examples/ReClaM/McSvm
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( .././examples/Rng/RngTest ${RngTest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/Rng/RngTest shark )
INSTALL( TARGETS .././examples/Rng/RngTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Rng )

ADD_EXECUTABLE( .././examples/Rng/RngTest2 ${RngTest2_SRC} )
TARGET_LINK_LIBRARIES( .././examples/Rng/RngTest2 shark )
INSTALL( TARGETS .././examples/Rng/RngTest2
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Rng )

ADD_EXECUTABLE( .././examples/Rng/RngTestSimple ${RngTestSimple_SRC} )
TARGET_LINK_LIBRARIES( .././examples/Rng/RngTestSimple shark )
INSTALL( TARGETS .././examples/Rng/RngTestSimple
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Rng )

ADD_EXECUTABLE( .././examples/TimeSeries/embeddingTest ${embeddingTest_SRC} )
TARGET_LINK_LIBRARIES( .././examples/TimeSeries/embeddingTest shark )
INSTALL( TARGETS .././examples/TimeSeries/embeddingTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/TimeSeries )

ADD_EXECUTABLE( .././examples/TimeSeries/lorenz84 ${lorenz84_SRC} )
TARGET_LINK_LIBRARIES( .././examples/TimeSeries/lorenz84 shark )
INSTALL( TARGETS .././examples/TimeSeries/lorenz84
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

IF( ENABLE_QT_EXAMPLES )

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

	qt4_wrap_cpp( Regression_MOCS ../examples/ReClaM/QT/regression.h )
	SET( Regression_SRCS ../examples/ReClaM/QT/regression.cpp ${Regression_MOCS} )
	ADD_EXECUTABLE( ../examples/ReClaM/QT/Regression ${Regression_SRCS} )
	TARGET_LINK_LIBRARIES( ../examples/ReClaM/QT/Regression shark ${shark_QT_LIBS})
	INSTALL( TARGETS ../examples/ReClaM/QT/Regression
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM/QT )

	qt4_wrap_cpp( Classification_MOCS ../examples/ReClaM/QT/classification.h )
	SET( Classification_SRCS ../examples/ReClaM/QT/classification.cpp ${Classification_MOCS} )
	ADD_EXECUTABLE( ../examples/ReClaM/QT/Classification ${Classification_SRCS} )
	TARGET_LINK_LIBRARIES( ../examples/ReClaM/QT/Classification shark ${shark_QT_LIBS})
	INSTALL( TARGETS ../examples/ReClaM/QT/Classification
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM/QT )

	IF( Qwt_LIBRARY )
	qt4_wrap_cpp( GaussianProcess_MOCS ../examples/ReClaM/QT/gaussian_process.h )
	SET( GaussianProcess_SRCS ../examples/ReClaM/QT/gaussian_process.cpp ${GaussianProcess_MOCS} )
	ADD_EXECUTABLE( ../examples/ReClaM/QT/GaussianProcess ${GaussianProcess_SRCS} )
	TARGET_LINK_LIBRARIES( ../examples/ReClaM/QT/GaussianProcess shark ${shark_QT_LIBS} ${Qwt_LIBRARY} )
	INSTALL( TARGETS ../examples/ReClaM/QT/GaussianProcess
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM/QT )
	ENDIF( Qwt_LIBRARY )

	qt4_wrap_cpp( EA_MOCS
		../examples/EALib/QT/ComparisonView.h
		../examples/EALib/QT/ConfigCtrl.h
		../examples/EALib/QT/ControlWidget.h
		../examples/EALib/QT/Experiment.h
		../examples/EALib/QT/ExperimentPropertiesWidget.h
		../examples/EALib/QT/LandscapeView2D.h
		../examples/EALib/QT/LandscapeView3D.h
		../examples/EALib/QT/MainWidget.h
		../examples/EALib/QT/MoFitnessView.h
		../examples/EALib/QT/StatusWidget.h
		../examples/EALib/QT/TestWidget.h
		../examples/EALib/QT/ValueView.h
		../examples/EALib/QT/View.h
		../examples/EALib/QT/Wizard.h
	)
	SET( EA_SRCS 
		${EA_MOCS} 
		../examples/EALib/QT/Canyon.cpp
		../examples/EALib/QT/ComparisonView.cpp
		../examples/EALib/QT/config.cpp
		../examples/EALib/QT/ConfigCtrl.cpp
		../examples/EALib/QT/ControlWidget.cpp
		../examples/EALib/QT/demo.cpp
		../examples/EALib/QT/Experiment.cpp
		../examples/EALib/QT/ExperimentPropertiesWidget.cpp
		../examples/EALib/QT/LandscapeView2D.cpp
		../examples/EALib/QT/LandscapeView3D.cpp
		../examples/EALib/QT/MainWidget.cpp
		../examples/EALib/QT/MoFitnessView.cpp
		../examples/EALib/QT/Optimization.cpp
		../examples/EALib/QT/StatusWidget.cpp
		../examples/EALib/QT/TestWidget.cpp
		../examples/EALib/QT/UTest.cpp
		../examples/EALib/QT/ValueView.cpp
		../examples/EALib/QT/View.cpp
		../examples/EALib/QT/Wizard.cpp
	)
	ADD_EXECUTABLE( ../examples/EALib/QT/EA ${EA_SRCS} )
	TARGET_LINK_LIBRARIES( ../examples/EALib/QT/EA shark ${shark_QT_LIBS} )
	INSTALL( TARGETS ../examples/EALib/QT/EA
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib/QT )

ENDIF( ENABLE_QT_EXAMPLES )
