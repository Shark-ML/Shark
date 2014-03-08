#####################################################################
#		Shark Machine Learning Library
#		Setup for example compilation
#####################################################################

CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
CMAKE_POLICY(SET CMP0003 NEW)

IF( OPT_COMPILE_EXAMPLES )
	add_custom_target(examples)
ENDIF( OPT_COMPILE_EXAMPLES )

#add_custom_target(examples)

#set(EXAMPLES_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(EXAMPLES_CPP_DIR ${CMAKE_CURRENT_BINARY_DIR})

SET(TPP_FILES "")

MACRO( SHARK_ADD_EXAMPLE TUT_SRC NAME COMPONENT )

	#generate cpp file from tpp file
	ADD_CUSTOM_COMMAND(
		OUTPUT ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp
		COMMAND ${CMAKE_COMMAND}
		ARGS
			-DTUT_TPP:Filepath=${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.tpp
			-DTUT_CPP:Filepath=${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp
			-P ${CMAKE_CURRENT_SOURCE_DIR}/createExampleSource.cmake
		MAIN_DEPENDENCY
			${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.tpp
		COMMENT "creating ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp from ${CMAKE_CURRENT_SOURCE_DIR}/${TUT_SRC}.tpp"
	)
	LIST(APPEND TPP_FILES ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp)

	IF( OPT_COMPILE_EXAMPLES )
		ADD_EXECUTABLE( ${NAME}
			${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp
		)

		SET( EXAMPLE_LIBRARIES
			${LINK_LIBRARIES}
			shark
		)

		add_dependencies(examples ${NAME})

		TARGET_LINK_LIBRARIES( ${NAME} ${EXAMPLE_LIBRARIES} )

		INSTALL( TARGETS ${NAME}
			RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/${COMPONENT}/
			COMPONENT EXAMPLES
		)

		INSTALL( FILES
			${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp
		 	DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/${COMPONENT}
		)

		SOURCE_GROUP( ${COMPONENT} FILES ${SRC} )
	ENDIF( OPT_COMPILE_EXAMPLES )
ENDMACRO()


MACRO( SHARK_ADD_WRAPPER WRAPPER_SRC NAME COMPONENT )
    LIST(APPEND WRAPPER_CPP_FILES ${EXAMPLES_CPP_DIR}/${WRAPPER_SRC}.cpp)

    IF( OPT_COMPILE_EXAMPLES )
        ADD_EXECUTABLE( ${NAME}
            ${EXAMPLES_CPP_DIR}/${WRAPPER_SRC}.cpp
        )

        SET( EXAMPLE_LIBRARIES
            ${LINK_LIBRARIES}
            shark
        )

        add_dependencies(examples ${NAME})

        TARGET_LINK_LIBRARIES( ${NAME} ${EXAMPLE_LIBRARIES} )

        INSTALL( TARGETS ${NAME}
            RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/${COMPONENT}/
            COMPONENT EXAMPLES
        )

        INSTALL( FILES
            ${EXAMPLES_CPP_DIR}/${WRAPPER_SRC}.cpp
            DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/${COMPONENT}
        )

        SOURCE_GROUP( ${COMPONENT} FILES ${SRC} )
    ENDIF( OPT_COMPILE_EXAMPLES )
ENDMACRO()



SHARK_ADD_EXAMPLE( Core/Oracle Oracle "Core" )

SHARK_ADD_EXAMPLE( EA/MOO/AdditiveEpsilonIndicatorMain AdditiveEpsilonIndicatorMain "EA/MOO" )
#SHARK_ADD_EXAMPLE( EA/MOO/ApproximatedHypervolumeMain ApproximatedHypervolumeMain "EA/MOO" )
#SHARK_ADD_EXAMPLE( EA/MOO/ExactHypervolumeMain ExactHypervolumeMain "EA/MOO" )
SHARK_ADD_EXAMPLE( EA/MOO/BoundingBoxCalculatorMain BoundingBoxCalculatorMain "EA/MOO" )

SHARK_ADD_EXAMPLE( EA/MOO/MOCMASimple MOCMASimple "EA/MOO" )

## SHARK_ADD_EXAMPLE( EA/SOO/XNES XNES "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/CMASimple CMASimple "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/CMAPlot CMAPlot "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/ElitistCMASimple ElitistCMASimpleMain "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/TSP TSP "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/AckleyES AckleyES "EA/SOO" )
SHARK_ADD_EXAMPLE( EA/SOO/Archive Archive "EA/SOO" )

SHARK_ADD_EXAMPLE( Supervised/MultiTaskSvm MultiTaskSvm "Supervised" )

SHARK_ADD_EXAMPLE( Supervised/CVFolds CVFolds "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CSvmLinear CSvmLinear "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/McSvm McSvm "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/McSvmLinear McSvmLinear "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/KernelRegression KernelRegression "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/KernelSelection KernelSelection "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/OneVersusOne OneVersusOne "Supervised" ) #partially working. Dataset issue
SHARK_ADD_EXAMPLE( Supervised/KernelLogisticRegression KernelLogisticRegression "Supervised" )

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
SHARK_ADD_EXAMPLE( Supervised/KNNCrossValidationTutorial KNNCrossValidationTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/linearRegressionTutorial linearRegressionTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/LassoRegression LassoRegression "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNBasicTutorial FFNNBasicTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNSetStructureTutorial FFNNSetStructureTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNMultiClassCrossEntropy FFNNMultiClassCrossEntropy "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/FFNNMultiClassNormalizedTraining FFNNMultiClassNormalizedTraining "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/RFTutorial RFTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/CARTTutorial CARTTutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/KTA-tutorial KTA-tutorial "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/VersatileClassificationTutorial-LDA VersatileClassificationTutorial-LDA "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/VersatileClassificationTutorial-NN VersatileClassificationTutorial-NN "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/VersatileClassificationTutorial-SVM VersatileClassificationTutorial-SVM "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/VersatileClassificationTutorial-RF VersatileClassificationTutorial-RF "Supervised" )
SHARK_ADD_EXAMPLE( Supervised/VersatileClassificationTutorial-Network VersatileClassificationTutorial-Network "Supervised" )

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



# Wrappers
SHARK_ADD_WRAPPER( wrapper/sharksvm-train sharksvm-train "sharksvm-train" )
SHARK_ADD_WRAPPER( wrapper/sharksvm-predict sharksvm-predict "sharksvm-predict" )



add_custom_target(tpp2cpp DEPENDS ${TPP_FILES})

IF( OPT_COMPILE_EXAMPLES )

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

ENDIF( OPT_COMPILE_EXAMPLES )
