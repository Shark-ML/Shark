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
MACRO( SHARK_ADD_EXAMPLE SRC NAME COMPONENT )

	ADD_EXECUTABLE( ${NAME}
		${SRC}
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
		${SRC}
	 	DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/${COMPONENT}
		)

	SOURCE_GROUP( ${COMPONENT} FILES ${SRC} )
ENDMACRO()

SHARK_ADD_EXAMPLE( Core/Oracle.cpp Oracle "Core" )
SHARK_ADD_EXAMPLE( Core/Probe.cpp Probe "Core" )
SHARK_ADD_EXAMPLE( Core/Logger.cpp Logger "Logger" )

IF( OPT_ENABLE_NETWORKING )
    SHARK_ADD_EXAMPLE( Network/HttpServer.cpp HttpServer "Network")
    INSTALL( DIRECTORY
	     "${CMAKE_CURRENT_SOURCE_DIR}/Network/ext-4.0.2a"
	     DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Network
	     PATTERN ".*" EXCLUDE
    )
    INSTALL( DIRECTORY
	     "${CMAKE_CURRENT_SOURCE_DIR}/Network/js"
	     DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Network
	     PATTERN ".*" EXCLUDE
    )
   INSTALL( FILES
   	    "${CMAKE_CURRENT_SOURCE_DIR}/Network/groupgrid.html"
	    DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/Network
	)

ENDIF( OPT_ENABLE_NETWORKING )

SHARK_ADD_EXAMPLE( EA/MOO/AdditiveEpsilonIndicatorMain.cpp AdditiveEpsilonIndicatorMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/ApproximatedHypervolumeMain.cpp ApproximatedHypervolumeMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/ExactHypervolumeMain.cpp ExactHypervolumeMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/BoundingBoxCalculatorMain.cpp BoundingBoxCalculatorMain "EA/MOO" )

SHARK_ADD_EXAMPLE( EA/MOO/AGE1Main.cpp AGE1Main "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/SMSEMOAMain.cpp SMSEMOAMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/SteadyStateMOCMAMain.cpp SteadyStateMOCMAMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/EpsilonSteadyStateMOCMAMain.cpp EpsilonSteadyStateMOCMAMain "EA/MOO" )
## SHARK_ADD_EXAMPLE( EA/MOO/LocalitySensitiveEpsilonSteadyStateMOCMAMain.cpp LocalitySensitiveEpsilonSteadyStateMOCMAMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/MOCMAMain.cpp MOCMAMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/MOCMASimple.cpp MOCMASimple "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/EpsilonMOCMAMain.cpp EpsilonMOCMAMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/Experiment.cpp MooExperiment "EA/MOO" )

## SHARK_ADD_EXAMPLE( EA/SOO/XNES.cpp XNES "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/CMA.cpp CMAMain "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/CMASimple.cpp CMASimple "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/CMAProbes.cpp CMAProbes "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/CMSA.cpp CMSAMain "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/ElitistCMA.cpp ElitistCMAMain "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/ElitistCMASimple.cpp ElitistCMASimpleMain "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/OnePlusOneES.cpp OnePlusOneES "EA/SOO" )
##SHARK_ADD_EXAMPLE( EA/SOO/TSP.cpp TSP "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/AckleyES.cpp AckleyES "EA/SOO" )

SHARK_ADD_EXAMPLE( Supervised/CVFolds.cpp CVFolds "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CSvmWithThresholdConverter.cpp CSvmWithThresholdConverter "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/McSvm.cpp McSvm "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/McSvmLinear.cpp McSvmLinear "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/KernelRegression.cpp KernelRegression "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/KernelSelection.cpp KernelSelection "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/OneVersusOne.cpp OneVersusOne "Supervised" ) #partially working. Dataset issue

SHARK_ADD_EXAMPLE( Supervised/CSvmMaxLikelihoodMS.cpp CSvmMaxLikelihoodMS "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CSvmGridSearchTutorial.cpp CSvmGridSearchTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CSvmTutorial.cpp CSvmTutorial "Supervised" )

##SHARK_ADD_EXAMPLE( Supervised/OnlineSparseMcSvmExp.cpp OnlineSparseMcSvmExp "Supervised" )
##SHARK_ADD_EXAMPLE( Supervised/OnlineCSMcSvmExp.cpp OnlineCSMcSvmExp "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/SubrangeKernelTutorial.cpp SubrangeKernelTutorial "Supervised" )

SHARK_ADD_EXAMPLE( Supervised/KNNTutorial.cpp KNNTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/LDATutorial.cpp LDATutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/elmTutorial.cpp elmTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/kNN.cpp kNN "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/linearRegressionTutorial.cpp linearRegressionTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNBasicTutorial.cpp FFNNBasicTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNSetStructureTutorial.cpp FFNNSetStructureTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNMultiClassCrossEntropy.cpp FFNNMultiClassCrossEntropy "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNMultiClassNormalizedTraining.cpp FFNNMultiClassNormalizedTraining "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/RFTutorial.cpp RFTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CARTTutorial.cpp CARTTutorial "Supervised" )

#basic tutorials
SHARK_ADD_EXAMPLE( Supervised/quickstartTutorial.cpp quickstartTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/regressionTutorial.cpp regressionTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/StoppingCriteria.cpp StoppingCriteriaTutorial "Supervised" )

#Statistics
SHARK_ADD_EXAMPLE( Statistics/StatisticsMain.cpp StatisticsMain "Statistiscs" )
SHARK_ADD_EXAMPLE( Statistics/Statistics.cpp Statistics "Statistiscs" )

#Unsupervisd
SHARK_ADD_EXAMPLE( Unsupervised/PCA.cpp PCA "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/PCATutorial.cpp PCATutorial "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/KMeansTutorial.cpp KMeansTutorial "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/BinaryRBM.cpp BinaryRBM "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/HierarchicalClustering.cpp HierarchicalClustering "Unsupervised"  )
SHARK_ADD_EXAMPLE( Unsupervised/OneClassSvm.cpp OneClassSvm "Unsupervised"  )

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

