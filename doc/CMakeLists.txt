CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

find_package( Doxygen REQUIRED )
find_package( PythonInterp 3 REQUIRED)

ADD_SUBDIRECTORY(tutToRst)


#configure doxygen

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doxygen_pages/tag_files/")
CONFIGURE_FILE (
	"${CMAKE_CURRENT_SOURCE_DIR}/Shark.dox.in"
	"${CMAKE_CURRENT_BINARY_DIR}/Shark.dox"
)
CONFIGURE_FILE(
	"${CMAKE_CURRENT_SOURCE_DIR}/doxygen_pages/css/besser.css"
	"${CMAKE_CURRENT_BINARY_DIR}/doxygen_pages/css/besser.css"
	COPYONLY
)

#configure sphinx
CONFIGURE_FILE (
	"${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/conf.py.in"
	"${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/conf.py"
)

#copy the adding_docs.rst because it fucks up tut2rst if it is a .tut file
CONFIGURE_FILE(
	"${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/rest_sources/tutorials/for_developers/adding_docs.rst"
	"${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/rest_sources/tutorials/for_developers/adding_docs.rst"
	COPYONLY
)

#create rst files from tut files and populate sphinx source
add_custom_target(doc_creation)
file(GLOB_RECURSE TutFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.tut)
foreach(tut ${TutFiles})
	GET_FILENAME_COMPONENT(tutPath ${tut} PATH)
	GET_FILENAME_COMPONENT(tutName ${tut} NAME_WE)
	#make sure, target directory exists
	file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${tutPath}/")
	add_custom_command(TARGET doc_creation POST_BUILD COMMAND
		tut2rst "${CMAKE_CURRENT_SOURCE_DIR}/${tutPath}/${tutName}.tut" "${CMAKE_CURRENT_BINARY_DIR}/${tutPath}/${tutName}.rst" "${PROJECT_SOURCE_DIR}/examples"
	)
endforeach()
add_dependencies(doc_creation tut2rst)
add_dependencies(tut2rst tpp2cpp)   # defined in examples folder


#run sphinx and doxygen
set( SPHINX_EXECUTABLE sphinx-build )
set( SPHINX_PARAMETERS -b html )

add_custom_target( doc ALL
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_CURRENT_SOURCE_DIR}/../examples/Supervised/data/
		${CMAKE_CURRENT_BINARY_DIR}/../examples/Supervised/data/
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_CURRENT_SOURCE_DIR}/../examples/Unsupervised/data/
		${CMAKE_CURRENT_BINARY_DIR}/../examples/Unsupervised/data/
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/templates/
		${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/templates/
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/static/
		${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/static/
		COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/ini_custom_themes/
		${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/ini_custom_themes/
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/rest_sources/about_shark/images/
		${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/rest_sources/about_shark/images/
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/rest_sources/tutorials/images/
		${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/rest_sources/tutorials/images
	COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Shark.dox"
	COMMAND ${SPHINX_EXECUTABLE} ${SPHINX_PARAMETERS} "${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages" "${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/build/html"
)
add_dependencies(doc doc_creation)

OPTION( OPT_INSTALL_DOCUMENTATION "Install documentation." OFF )

IF( OPT_INSTALL_DOCUMENTATION )

INSTALL( FILES index.html mlstyle.js frontpage.css
	lgplv3-147x51.png SharkLogo.png download.png sharkUnix.png  
	shark.png sharkUnix.png sharkWin.png shark16.ico sharklogo30x32.gif
	DESTINATION ${SHARK_INSTALL_DOC_DIR}
)

INSTALL( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen_pages
	DESTINATION ${SHARK_INSTALL_DOC_DIR}
	PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/build/html/
	DESTINATION ${SHARK_INSTALL_DOC_DIR}/sphinx_pages/build/html/
	PATTERN ".*" EXCLUDE
)

ENDIF( OPT_INSTALL_DOCUMENTATION )
