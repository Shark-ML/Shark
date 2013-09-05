#####################################################################
#		Shark Machine Learning Library
#		Setup for example compilation
#####################################################################

CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
CMAKE_POLICY(SET CMP0003 NEW)

#####################################################################
#	Adds an example for the shark library
#	Param: SRC Source files for compilation
#	Param: NAME Target name for the resulting executable
#	Param: COMPONENT Installation dir, e.g., EA/MOO
#	Output: Executable in
#          * ${SHARK_INSTALL_EXAMPLE_DIR}
#            * ${COMPONENT}
#       If OPT_LOG_TEST_OUTPUT is enabled, test log is written
#	to ${NAME_Log.xml} in ${SHARK}/Test/bin.
#####################################################################
MACRO( SHARK_ADD_EXAMPLE TUT_SRC NAME COMPONENT )

	#generate cpp file if needed
	if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.cpp" 
		OR ("${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.tpp" 
		IS_NEWER_THAN  "${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.cpp")
	)
		message("generate ${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.cpp")
		FILE( READ ${TUT_SRC}.tpp  contents )
		STRING( REGEX REPLACE "\t*//###[^\n]*\n" "" contents_cleared "${contents}")
		FILE( WRITE ${TUT_SRC}.cpp "${contents_cleared}")
	ENDIF()
	
	ADD_EXECUTABLE( ${NAME}
		${TUT_SRC}.cpp
	)

	SET( EXAMPLE_LIBRARIES
		${LINK_LIBRARIES}
		shark
	)

	TARGET_LINK_LIBRARIES( ${NAME} ${EXAMPLE_LIBRARIES} )

	INSTALL( TARGETS ${NAME}
		RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/${COMPONENT}/
		COMPONENT EXAMPLES
	)

	INSTALL( FILES
		${TUT_SRC}.cpp
	 	DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/${COMPONENT}
	)

	SOURCE_GROUP( ${COMPONENT} FILES ${SRC} )
ENDMACRO()

SHARK_ADD_EXAMPLE( Core/Oracle Oracle "Core" )

SHARK_ADD_EXAMPLE( EA/MOO/AdditiveEpsilonIndicatorMain AdditiveEpsilonIndicatorMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/ApproximatedHypervolumeMain ApproximatedHypervolumeMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/ExactHypervolumeMain ExactHypervolumeMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/BoundingBoxCalculatorMain BoundingBoxCalculatorMain "EA/MOO" )

SHARK_ADD_EXAMPLE( EA/MOO/MOCMASimple MOCMASimple "EA/MOO" )

## SHARK_ADD_EXAMPLE( EA/SOO/XNES XNES "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/CMASimple CMASimple "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/CMAPlot CMAPlot "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/ElitistCMASimple ElitistCMASimpleMain "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/TSP TSP "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/AckleyES AckleyES "EA/SOO" )

SHARK_ADD_EXAMPLE( Supervised/MultiTaskSvm MultiTaskSvm "Supervised" )

SHARK_ADD_EXAMPLE( Supervised/CVFolds CVFolds "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CSvmWithThresholdConverter CSvmWithThresholdConverter "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/McSvm McSvm "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/McSvmLinear McSvmLinear "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/KernelRegression KernelRegression "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/KernelSelection KernelSelection "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/OneVersusOne OneVersusOne "Supervised" ) #partially working. Dataset issue

SHARK_ADD_EXAMPLE( Supervised/CSvmMaxLikelihoodMS CSvmMaxLikelihoodMS "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CSvmGridSearchTutorial CSvmGridSearchTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CSvmTutorial CSvmTutorial "Supervised" )

##SHARK_ADD_EXAMPLE( Supervised/OnlineSparseMcSvmExp OnlineSparseMcSvmExp "Supervised" )
##SHARK_ADD_EXAMPLE( Supervised/OnlineCSMcSvmExp OnlineCSMcSvmExp "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/MklKernelTutorial MklKernelTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/SubrangeKernelTutorial SubrangeKernelTutorial "Supervised" )

SHARK_ADD_EXAMPLE( Supervised/KNNTutorial KNNTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/LDATutorial LDATutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/elmTutorial elmTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/kNN kNN "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/linearRegressionTutorial linearRegressionTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/LassoRegression LassoRegression "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNBasicTutorial FFNNBasicTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNSetStructureTutorial FFNNSetStructureTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNMultiClassCrossEntropy FFNNMultiClassCrossEntropy "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNMultiClassNormalizedTraining FFNNMultiClassNormalizedTraining "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/RFTutorial RFTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CARTTutorial CARTTutorial "Supervised" )

#basic tutorials
SHARK_ADD_EXAMPLE( Supervised/quickstartTutorial quickstartTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/regressionTutorial regressionTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/StoppingCriteria StoppingCriteriaTutorial "Supervised" )

#Statistics
SHARK_ADD_EXAMPLE( Statistics/StatisticsMain StatisticsMain "Statistiscs" )
SHARK_ADD_EXAMPLE( Statistics/Statistics Statistics "Statistiscs" )

#Unsupervisd
SHARK_ADD_EXAMPLE( Unsupervised/PCA PCA "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/PCATutorial PCATutorial "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/KMeansTutorial KMeansTutorial "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/BinaryRBM BinaryRBM "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/SparseAETutorial SparseAETutorial "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/HierarchicalClustering HierarchicalClustering "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/OneClassSvm OneClassSvm "Unsupervised"  )

INSTALL( DIRECTORY
	 "${CMAKE_CURRENT_SOURCE_DIR}/Supervised/data"
	 DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Supervised
	 PATTERN ".*" EXCLUDE
)
INSTALL( DIRECTORY
	 "${CMAKE_CURRENT_SOURCE_DIR}/Unsupervised/data"
	 DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Unsupervised
	 PATTERN ".*" EXCLUDE
)

#####Automatic Generation of example CMakeLists.txt
CONFIGURE_FILE (
  "${CMAKE_CURRENT_SOURCE_DIR}/ExampleProject/CMakeLists.txt.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/ExampleProject/CMakeLists.txt"
  @ONLY
  )

INSTALL(
	FILES ${CMAKE_CURRENT_SOURCE_DIR}/ExampleProject/CMakeLists.txt
	DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/ExampleProject/
)

