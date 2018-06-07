#####################################################################
#		Shark Machine Learning Library
#		Setup for example compilation
#####################################################################

add_custom_target(sharkexamples)
set(EXAMPLES_CPP_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TPP_FILES "")

macro( SHARK_ADD_EXAMPLE TUT_SRC NAME COMPONENT )
	# Generate cpp file from tpp file
	add_custom_command(
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

	list(APPEND TPP_FILES ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp)

	if( BUILD_EXAMPLES )
		add_executable(${NAME} ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp)
		target_link_libraries(${NAME} shark)
		set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 11)
		set_property(TARGET ${NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

		# Add the dependencies to the target examples
		add_dependencies(sharkexamples ${NAME})

		set_target_properties( ${NAME} PROPERTIES FOLDER "examples")

		# Installation
		install(TARGETS ${NAME}
			   RUNTIME DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/${COMPONENT}/
			   COMPONENT EXAMPLES)

		install(FILES ${EXAMPLES_CPP_DIR}/${TUT_SRC}.cpp
			   DESTINATION ${SHARK_INSTALL_EXAMPLE_DIR}/${COMPONENT})

		source_group(${COMPONENT} FILES ${TUT_SRC})
	endif()
endmacro()

shark_add_example( EA/MOO/MOCMASimple MOCMASimple "EA/MOO" )
shark_add_example( EA/MOO/MOCMAApproximated MOCMAApproximated "EA/MOO" )
shark_add_example( EA/MOO/MOCMAExperiment MOCMAExperiment "EA/MOO" )

shark_add_example( EA/SOO/CMASimple CMASimple "EA/SOO" )
shark_add_example( EA/SOO/CMAExperiment CMAExperiment "EA/SOO" )
shark_add_example( EA/SOO/CMAPlot CMAPlot "EA/SOO" )
shark_add_example( EA/SOO/ElitistCMASimple ElitistCMASimpleMain "EA/SOO" )
shark_add_example( EA/SOO/TSP TSP "EA/SOO" )
shark_add_example( EA/SOO/AckleyES AckleyES "EA/SOO" )
shark_add_example( EA/SOO/Archive Archive "EA/SOO" )

shark_add_example( Supervised/MultiTaskSvm MultiTaskSvm "Supervised" )


shark_add_example( Supervised/CVFolds CVFolds "Supervised" )
shark_add_example( Supervised/CSvmLinear CSvmLinear "Supervised" )
shark_add_example( Supervised/McSvm McSvm "Supervised" )
shark_add_example( Supervised/McSvmLinear McSvmLinear "Supervised" )
shark_add_example( Supervised/KernelRegression KernelRegression "Supervised" )
shark_add_example( Supervised/KernelSelection KernelSelection "Supervised" )
shark_add_example( Supervised/OneVersusOne OneVersusOne "Supervised" )
shark_add_example( Supervised/KernelLogisticRegression KernelLogisticRegression "Supervised" )

shark_add_example( Supervised/CSvmMaxLikelihoodMS CSvmMaxLikelihoodMS "Supervised" )
shark_add_example( Supervised/CSvmGridSearchTutorial CSvmGridSearchTutorial "Supervised" )
shark_add_example( Supervised/CSvmTutorial CSvmTutorial "Supervised" )

shark_add_example( Supervised/MklKernelTutorial MklKernelTutorial "Supervised" )
shark_add_example( Supervised/SubrangeKernelTutorial SubrangeKernelTutorial "Supervised" )

shark_add_example( Supervised/KNNTutorial KNNTutorial "Supervised" )
shark_add_example( Supervised/LDATutorial LDATutorial "Supervised" )
shark_add_example( Supervised/KNNCrossValidationTutorial KNNCrossValidationTutorial "Supervised" )
shark_add_example( Supervised/linearRegressionTutorial linearRegressionTutorial "Supervised" )
shark_add_example( Supervised/LassoRegression LassoRegression "Supervised" )
shark_add_example( Supervised/FFNNBasicTutorial FFNNBasicTutorial "Supervised" )
shark_add_example( Supervised/MNISTForExperts MNISTForExperts "Supervised" )
shark_add_example( Supervised/DeepNetworkTrainingRBM DeepNetworkTrainingRBM "Supervised" )
shark_add_example( Supervised/RFTutorial RFTutorial "Supervised" )
shark_add_example( Supervised/KTA-tutorial KTA-tutorial "Supervised" )
shark_add_example( Supervised/KernelBudgetedSGDTutorial KernelBudgetedSGDTutorial "Supervised" )

#basic tutorials
shark_add_example( Supervised/quickstartTutorial quickstartTutorial "Supervised" )
shark_add_example( Supervised/regressionTutorial regressionTutorial "Supervised" )
shark_add_example( Supervised/StoppingCriteria StoppingCriteriaTutorial "Supervised" )

#Statistics
shark_add_example( Statistics/Statistics Statistics "Statistiscs" )

#Data
shark_add_example( Data/Datasets Datasets "Data" )
shark_add_example( Data/Normalization Normalization "Data" )
shark_add_example( Data/Subsets Subsets "Data" )
shark_add_example( Data/Import Import "Data" )

#Unsupervisd
shark_add_example( Unsupervised/PCA PCA "Unsupervised" )
shark_add_example( Unsupervised/PCATutorial PCATutorial "Unsupervised" )
shark_add_example( Unsupervised/KMeansTutorial KMeansTutorial "Unsupervised" )
shark_add_example( Unsupervised/BinaryRBM BinaryRBM "Unsupervised" )
shark_add_example( Unsupervised/AutoEncoderTutorial AutoencoderTutorial "Unsupervised" )
shark_add_example( Unsupervised/VariationalAutoencoder VariationalAutoencoder "Unsupervised" )
shark_add_example( Unsupervised/HierarchicalClustering HierarchicalClustering "Unsupervised" )
shark_add_example( Unsupervised/OneClassSvm OneClassSvm "Unsupervised" )

add_custom_target(tpp2cpp DEPENDS ${TPP_FILES})

# Installation of the examples data
if( BUILD_EXAMPLES )
	install( DIRECTORY
		"${CMAKE_CURRENT_SOURCE_DIR}/Supervised/data"
		DESTINATION "${SHARK_INSTALL_EXAMPLE_DIR}/Supervised/data/"
		PATTERN ".*" EXCLUDE
	)

	install( DIRECTORY
		"${CMAKE_CURRENT_SOURCE_DIR}/Unsupervised/data"
		DESTINATION "${SHARK_INSTALL_EXAMPLE_DIR}/Unsupervised/data"
		PATTERN ".*" EXCLUDE
	)
endif()
		 
