PROJECT( shark )



CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

OPTION( SHARK_OPT_ENABLE_QT_EXAMPLES "" OFF )

#CMAKE_POLICY( SET CMP0003 OLD )

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../include )

# LINK_DIRECTORIES( ${CMAKE_BINARY_DIR}/lib )

MESSAGE( ${CMAKE_BINARY_DIR} )
MESSAGE( ${CMAKE_CURRENT_BINARY_DIR} )

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

ADD_EXECUTABLE( ackleyES ${ackleyES_SRC} )
TARGET_LINK_LIBRARIES( ackleyES shark )
INSTALL( TARGETS ackleyES
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( paraboloidCMA ${paraboloidCMA_SRC} )
TARGET_LINK_LIBRARIES( paraboloidCMA shark )
INSTALL( TARGETS paraboloidCMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( paraboloidElitistCMA ${paraboloidElitistCMA_SRC} )
TARGET_LINK_LIBRARIES( paraboloidElitistCMA shark )
INSTALL( TARGETS paraboloidElitistCMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( countingOnes ${countingOnes_SRC} )
TARGET_LINK_LIBRARIES( countingOnes shark )
INSTALL( TARGETS countingOnes
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( integerES ${integerES_SRC} )
TARGET_LINK_LIBRARIES( integerES shark )
INSTALL( TARGETS integerES
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( sphereGA ${sphereGA_SRC} )
TARGET_LINK_LIBRARIES( sphereGA shark )
INSTALL( TARGETS sphereGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( steadyState ${steadyState_SRC} )
TARGET_LINK_LIBRARIES( steadyState shark )
INSTALL( TARGETS steadyState
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( pvmSphereGA ${pvmSphereGA_SRC} )
TARGET_LINK_LIBRARIES( pvmSphereGA shark )
INSTALL( TARGETS pvmSphereGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( TSP_GA ${TSPGA_SRC} )
TARGET_LINK_LIBRARIES( TSP_GA shark )
INSTALL( TARGETS TSP_GA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( SchwefelEllipsoidCMSA ${SchwefelEllipsoidCMSA_SRC} )
TARGET_LINK_LIBRARIES( SchwefelEllipsoidCMSA shark )
INSTALL( TARGETS SchwefelEllipsoidCMSA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib )

ADD_EXECUTABLE( FileUtilSimple ${FileUtilSimple_SRC} )
TARGET_LINK_LIBRARIES( FileUtilSimple shark )
INSTALL( TARGETS FileUtilSimple
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/FileUtil )

ADD_EXECUTABLE( FileUtilClass ${FileUtilClass_SRC} )
TARGET_LINK_LIBRARIES( FileUtilClass shark )
INSTALL( TARGETS FileUtilClass
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/FileUtil )

ADD_EXECUTABLE( sugeno ${sugeno_SRC} )
TARGET_LINK_LIBRARIES( sugeno shark )
INSTALL( TARGETS sugeno
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Fuzzy )

ADD_EXECUTABLE( rank_test ${ranktest_SRC} )
TARGET_LINK_LIBRARIES( rank_test shark )
INSTALL( TARGETS rank_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( rank_decomp_test ${rankdecomptest_SRC} )
TARGET_LINK_LIBRARIES( rank_decomp_test shark )
INSTALL( TARGETS rank_decomp_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( svd_test ${svdtest_SRC} )
TARGET_LINK_LIBRARIES( svd_test shark )
INSTALL( TARGETS svd_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( svdrank_test ${svdranktest_SRC} )
TARGET_LINK_LIBRARIES( svdrank_test shark )
INSTALL( TARGETS svdrank_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( svdsort_test ${svdsorttest_SRC} )
TARGET_LINK_LIBRARIES( svdsort_test shark )
INSTALL( TARGETS svdsort_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( eigensymm_test ${eigensymmtest_SRC} )
TARGET_LINK_LIBRARIES( eigensymm_test shark )
INSTALL( TARGETS eigensymm_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( eigensymmJacobi_test ${eigensymmJacobitest_SRC} )
TARGET_LINK_LIBRARIES( eigensymmJacobi_test shark )
INSTALL( TARGETS eigensymmJacobi_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( eigensymmJacobi2_test ${eigensymmJacobi2test_SRC} )
TARGET_LINK_LIBRARIES( eigensymmJacobi2_test shark )
INSTALL( TARGETS eigensymmJacobi2_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( eigensort_test ${eigensorttest_SRC} )
TARGET_LINK_LIBRARIES( eigensort_test shark )
INSTALL( TARGETS eigensort_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( eigenerr_test ${eigenerrtest_SRC} )
TARGET_LINK_LIBRARIES( eigenerr_test shark )
INSTALL( TARGETS eigenerr_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( g_inverse_matrix ${ginversematrix_SRC} )
TARGET_LINK_LIBRARIES( g_inverse_matrix shark )
INSTALL( TARGETS g_inverse_matrix
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( detsymm_test ${detsymmtest_SRC} )
TARGET_LINK_LIBRARIES( detsymm_test shark )
INSTALL( TARGETS detsymm_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( lnsrch_test ${lnsrchtest_SRC} )
TARGET_LINK_LIBRARIES( lnsrch_test shark )
INSTALL( TARGETS lnsrch_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( cblnsrch_test ${cblnsrchtest_SRC} )
TARGET_LINK_LIBRARIES( cblnsrch_test shark )
INSTALL( TARGETS cblnsrch_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( linmin_test ${linmintest_SRC} )
TARGET_LINK_LIBRARIES( linmin_test shark )
INSTALL( TARGETS linmin_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( dlinmin_test ${dlinmintest_SRC} )
TARGET_LINK_LIBRARIES( dlinmin_test shark )
INSTALL( TARGETS dlinmin_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( linalg_simple_test ${linalgsimpletest_SRC} )
TARGET_LINK_LIBRARIES( linalg_simple_test shark )
INSTALL( TARGETS linalg_simple_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( covar_corrcoef_test ${covarcorrcoeftest_SRC} )
TARGET_LINK_LIBRARIES( covar_corrcoef_test shark )
INSTALL( TARGETS covar_corrcoef_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( fft_test ${ffttest_SRC} )
TARGET_LINK_LIBRARIES( fft_test shark )
INSTALL( TARGETS fft_test
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/LinAlg )

ADD_EXECUTABLE( MOGA ${MOGA_SRC} )
TARGET_LINK_LIBRARIES( MOGA shark )
INSTALL( TARGETS MOGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( NSGA2 ${NSGA2_SRC} )
TARGET_LINK_LIBRARIES( NSGA2 shark )
INSTALL( TARGETS NSGA2
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( PAES ${PAES_SRC} )
TARGET_LINK_LIBRARIES( PAES shark )
INSTALL( TARGETS PAES
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( VEGA ${VEGA_SRC} )
TARGET_LINK_LIBRARIES( VEGA shark )
INSTALL( TARGETS VEGA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( NSGA2example ${NSGA2example_SRC} )
TARGET_LINK_LIBRARIES( NSGA2example shark )
INSTALL( TARGETS NSGA2example
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( RMMEDAtest ${RMMEDAtest_SRC} )
TARGET_LINK_LIBRARIES( RMMEDAtest shark )
INSTALL( TARGETS RMMEDAtest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( realcoded-NSGA2 ${realcodedNSGA2_SRC} )
TARGET_LINK_LIBRARIES( realcoded-NSGA2 shark )
INSTALL( TARGETS realcoded-NSGA2
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( fonMO-CMA ${fonMOCMA_SRC} )
TARGET_LINK_LIBRARIES( fonMO-CMA shark )
INSTALL( TARGETS fonMO-CMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( fonSteadyStateMO-CMA ${fonSteadyStateMOCMA_SRC} )
TARGET_LINK_LIBRARIES( fonSteadyStateMO-CMA shark )
INSTALL( TARGETS fonSteadyStateMO-CMA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( HypervolumeBenchmark ${HypervolumeBenchmark_SRC} )
TARGET_LINK_LIBRARIES( HypervolumeBenchmark shark )
INSTALL( TARGETS HypervolumeBenchmark
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/MOO-EALib )

ADD_EXECUTABLE( mg-example ${mgexample_SRC} )
TARGET_LINK_LIBRARIES( mg-example shark )
INSTALL( TARGETS mg-example
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Mixture )

ADD_EXECUTABLE( rbfn-example ${rbfnexample_SRC} )
TARGET_LINK_LIBRARIES( rbfn-example shark )
INSTALL( TARGETS rbfn-example
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Mixture )

ADD_EXECUTABLE( simpleRBFNet ${simpleRBFNet_SRC} )
TARGET_LINK_LIBRARIES( simpleRBFNet shark )
INSTALL( TARGETS simpleRBFNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( simpleFFNet ${simpleFFNet_SRC} )
TARGET_LINK_LIBRARIES( simpleFFNet shark )
INSTALL( TARGETS simpleFFNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( simpleRNNet ${simpleRNNet_SRC} )
TARGET_LINK_LIBRARIES( simpleRNNet shark )
INSTALL( TARGETS simpleRNNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( simpleFFNetSource ${simpleFFNetSource_SRC} )
TARGET_LINK_LIBRARIES( simpleFFNetSource shark )
INSTALL( TARGETS simpleFFNetSource
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( simpleMSERNNet ${simpleMSERNNet_SRC} )
TARGET_LINK_LIBRARIES( simpleMSERNNet shark )
INSTALL( TARGETS simpleMSERNNet
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( SVMclassification ${SVMclassification_SRC} )
TARGET_LINK_LIBRARIES( SVMclassification shark )
INSTALL( TARGETS SVMclassification
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

IF( LINUX )
ADD_EXECUTABLE( SVMclassification-gnuplot ${SVMclassificationgnuplot_SRC} )
TARGET_LINK_LIBRARIES( SVMclassification-gnuplot shark )
INSTALL( TARGETS SVMclassification-gnuplot
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )
ENDIF( LINUX )

ADD_EXECUTABLE( SVMregression ${SVMregression_SRC} )
TARGET_LINK_LIBRARIES( SVMregression shark )
INSTALL( TARGETS SVMregression
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

IF( LINUX )
ADD_EXECUTABLE( SVMregression-gnuplot ${SVMregressiongnuplot_SRC} )
TARGET_LINK_LIBRARIES( SVMregression-gnuplot shark )
INSTALL( TARGETS SVMregression-gnuplot
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )
ENDIF( LINUX )

ADD_EXECUTABLE( KNN ${KNN_SRC} )
TARGET_LINK_LIBRARIES( KNN shark )
INSTALL( TARGETS KNN
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( KM ${KM_SRC} )
TARGET_LINK_LIBRARIES( KM shark )
INSTALL( TARGETS KM
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( CrossValidation ${CrossValidation_SRC} )
TARGET_LINK_LIBRARIES( CrossValidation shark )
INSTALL( TARGETS CrossValidation
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( KernelOptimization ${KernelOptimization_SRC} )
TARGET_LINK_LIBRARIES( KernelOptimization shark )
INSTALL( TARGETS KernelOptimization
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( SvmApproximationExample ${SvmApproximationExample_SRC} )
TARGET_LINK_LIBRARIES( SvmApproximationExample shark )
INSTALL( TARGETS SvmApproximationExample
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( LinearClassifierTest ${LinearClassifierTest_SRC} )
TARGET_LINK_LIBRARIES( LinearClassifierTest shark )
INSTALL( TARGETS LinearClassifierTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( PCAtest ${PCAtest_SRC} )
TARGET_LINK_LIBRARIES( PCAtest shark )
INSTALL( TARGETS PCAtest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( FisherLDA ${FisherLDA_SRC} )
TARGET_LINK_LIBRARIES( FisherLDA shark )
INSTALL( TARGETS FisherLDA
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( LinearRegressionTest ${LinearRegressionTest_SRC} )
TARGET_LINK_LIBRARIES( LinearRegressionTest shark )
INSTALL( TARGETS LinearRegressionTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( KalmanFilterTest ${KalmanFilterTest_SRC} )
TARGET_LINK_LIBRARIES( KalmanFilterTest shark )
INSTALL( TARGETS KalmanFilterTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( McSvm ${McSvm_SRC} )
TARGET_LINK_LIBRARIES( McSvm shark )
INSTALL( TARGETS McSvm
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM )

ADD_EXECUTABLE( RngTest ${RngTest_SRC} )
TARGET_LINK_LIBRARIES( RngTest shark )
INSTALL( TARGETS RngTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Rng )

ADD_EXECUTABLE( RngTest2 ${RngTest2_SRC} )
TARGET_LINK_LIBRARIES( RngTest2 shark )
INSTALL( TARGETS RngTest2
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Rng )

ADD_EXECUTABLE( RngTestSimple ${RngTestSimple_SRC} )
TARGET_LINK_LIBRARIES( RngTestSimple shark )
INSTALL( TARGETS RngTestSimple
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Rng )

ADD_EXECUTABLE( embeddingTest ${embeddingTest_SRC} )
TARGET_LINK_LIBRARIES( embeddingTest shark )
INSTALL( TARGETS embeddingTest
RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/TimeSeries )

ADD_EXECUTABLE( lorenz84 ${lorenz84_SRC} )
TARGET_LINK_LIBRARIES( lorenz84 shark )
INSTALL( TARGETS lorenz84
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

 ADD_TEST( SchwefelEllipsoidCMSA SchwefelEllipsoidCMSA )
  ADD_TEST( TSP_GA TSP_GA )
  ADD_TEST( ackleyES ackleyES )
  ADD_TEST( countingOnes countingOnes )
  ADD_TEST( integerES integerES )
  ADD_TEST( paraboloidCMA paraboloidCMA )
  ADD_TEST( paraboloidElitistCMA paraboloidElitistCMA )
  ADD_TEST( sphereGA sphereGA )
  ADD_TEST( steadyState steadyState )

  #	ADD_TEST( sugeno examples/Fuzzy/sugeno )

  ADD_TEST( cblnsrch_test cblnsrch_test )
  ADD_TEST( covar_corrcoef_test covar_corrcoef_test )
  ADD_TEST( detsymm_test detsymm_test )
  ADD_TEST( dlinmin_test dlinmin_test )
  ADD_TEST( eigenerr_test eigenerr_test )
  ADD_TEST( eigensort_test eigensort_test )
  ADD_TEST( eigensymmJacobi2_test eigensymmJacobi2_test )
  ADD_TEST( eigensymmJacobi_test eigensymmJacobi_test )
  ADD_TEST( eigensymm_test eigensymm_test )
  ADD_TEST( fft_test fft_test )
  ADD_TEST( g_inverse_matrix g_inverse_matrix )
  ADD_TEST( linalg_simple_test linalg_simple_test )
  ADD_TEST( linmin_test linmin_test )
  ADD_TEST( lnsrch_test lnsrch_test )
  ADD_TEST( rank_decomp_test rank_decomp_test )
  ADD_TEST( rank_test rank_test )
  ADD_TEST( svd_test svd_test )
  ADD_TEST( svdrank_test svdrank_test )
  ADD_TEST( svdsort_test svdsort_test )

  ADD_TEST( mg-example mg-example )
  ADD_TEST( rbfn-example rbfn-example )

  ADD_TEST( HypervolumeBenchmark HypervolumeBenchmark ${CMAKE_CURRENT_SOURCE_DIR}/MOO-EALib/ran.40000pts.3d.1 )
  # ADD_TEST( MOGA MOGA )
  #ADD_TEST( NSGA2 NSGA2 )
  #ADD_TEST( NSGA2example NSGA2example )
  # ADD_TEST( PAES PAES )
  ADD_TEST( RMMEDAtest RMMEDAtest )
  #ADD_TEST( VEGA VEGA )
  ADD_TEST( fonMO-CMA fonMO-CMA )
  ADD_TEST( fonSteadyStateMO-CMA fonSteadyStateMO-CMA )
  ADD_TEST( realcoded-NSGA2 realcoded-NSGA2 )

  ADD_TEST( CrossValidation CrossValidation )
  ADD_TEST( FisherLDA FisherLDA )
  ADD_TEST( KM KM )
  ADD_TEST( KNN KNN )
  ADD_TEST( KalmanFilterTest KalmanFilterTest )
  ADD_TEST( KernelOptimization KernelOptimization )
  ADD_TEST( LinearClassifierTest LinearClassifierTest )
  ADD_TEST( LinearRegressionTest LinearRegressionTest )
  ADD_TEST( McSvm McSvm )
  ADD_TEST( PCAtest PCAtest )
  ADD_TEST( SVMclassification SVMclassification )
  ADD_TEST( SVMregression SVMregression )
  ADD_TEST( SvmApproximationExample SvmApproximationExample )
  ADD_TEST( simpleFFNet simpleFFNet )
  ADD_TEST( simpleFFNetSource simpleFFNetSource )
  ADD_TEST( simpleMSERNNet simpleMSERNNet ${CMAKE_CURRENT_SOURCE_DIR}/ReClaM/timeseries )
  ADD_TEST( simpleRBFNet simpleRBFNet )
  ADD_TEST( simpleRNNet simpleRNNet )

  ADD_TEST( RngTest RngTest )
  ADD_TEST( RngTest2 RngTest2 )
  ADD_TEST( RngTestSimple RngTestSimple )

  ADD_TEST( embeddingTest embeddingTest )
  ADD_TEST( lorenz84 lorenz84 )

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
	ADD_EXECUTABLE( Regression ${Regression_SRCS} )
	TARGET_LINK_LIBRARIES( Regression shark ${shark_QT_LIBS})
	INSTALL( TARGETS Regression
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM/QT )

	qt4_wrap_cpp( Classification_MOCS ReClaM/QT/classification.h )
	SET( Classification_SRCS ReClaM/QT/classification.cpp ${Classification_MOCS} )
	ADD_EXECUTABLE( Classification ${Classification_SRCS} )
	TARGET_LINK_LIBRARIES( Classification shark ${shark_QT_LIBS})
	INSTALL( TARGETS Classification
	RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ReClaM/QT )

	IF( Qwt_LIBRARY )
	qt4_wrap_cpp( GaussianProcess_MOCS ReClaM/QT/gaussian_process.h )
	SET( GaussianProcess_SRCS ReClaM/QT/gaussian_process.cpp ${GaussianProcess_MOCS} )
	ADD_EXECUTABLE( GaussianProcess ${GaussianProcess_SRCS} )
	TARGET_LINK_LIBRARIES( GaussianProcess shark ${shark_QT_LIBS} ${Qwt_LIBRARY} )
	INSTALL( TARGETS GaussianProcess
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
	ADD_EXECUTABLE( EA ${EA_SRCS} )
	TARGET_LINK_LIBRARIES( EA shark ${shark_QT_LIBS} )
	INSTALL( TARGETS EA RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/EALib/QT )

ENDIF( SHARK_OPT_ENABLE_QT_EXAMPLES )
