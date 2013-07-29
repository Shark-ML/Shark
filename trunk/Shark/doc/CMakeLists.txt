CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

find_package( Doxygen REQUIRED )
find_package( PythonInterp REQUIRED )

ADD_SUBDIRECTORY(tutToRst)

CONFIGURE_FILE (
  "${CMAKE_CURRENT_SOURCE_DIR}/Shark.dox.in"
  "${CMAKE_CURRENT_BINARY_DIR}/Shark.dox"
)

CONFIGURE_FILE (
  "${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/conf.py.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages/conf.py"
)

set( SPHINX_EXECUTABLE sphinx-build )
set( SPHINX_PARAMETERS -b html )

add_custom_target(doc_creation)

#find all .tut files
file(GLOB_RECURSE TutFiles sphinx_pages *.tut)
foreach(tut ${TutFiles})
	GET_FILENAME_COMPONENT(tutPath ${tut} PATH)
	GET_FILENAME_COMPONENT(tutName ${tut} NAME_WE)
	add_custom_command(TARGET doc_creation POST_BUILD COMMAND 
		tut2rst ${tutPath}/${tutName} ${PROJECT_SOURCE_DIR}/examples
	) 
endforeach()
add_dependencies(doc_creation tut2rst)

message("${CMAKE_CURRENT_SOURCE_DIR}/doxygen_pages/css/besser.css")
add_custom_target( doc ALL
	COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Shark.dox
	COMMAND ${SPHINX_EXECUTABLE} ${SPHINX_PARAMETERS} ${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages ${CMAKE_CURRENT_BINARY_DIR}/sphinx_pages/build/html
)
add_custom_command(TARGET doc PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy
		${CMAKE_CURRENT_SOURCE_DIR}/doxygen_pages/css/besser.css 
		${CMAKE_CURRENT_BINARY_DIR}/doxygen_pages/css/besser.css )


add_dependencies(doc doc_creation)

OPTION( OPT_INSTALL_DOCUMENTATION "Install documentation." OFF )

IF( OPT_INSTALL_DOCUMENTATION )

INSTALL( FILES index.html
	DESTINATION ${SHARK_INSTALL_DOC_DIR}
)

INSTALL( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen_pages
	DESTINATION ${SHARK_INSTALL_DOC_DIR}
	PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen_pages/css
	DESTINATION ${SHARK_INSTALL_DOC_DIR}/doxygen_pages
	PATTERN ".*" EXCLUDE
)

INSTALL( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sphinx_pages
	DESTINATION ${SHARK_INSTALL_DOC_DIR}
	PATTERN ".*" EXCLUDE
)

ENDIF( OPT_INSTALL_DOCUMENTATION )
