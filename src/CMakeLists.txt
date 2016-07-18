# Glob the files
file( GLOB_RECURSE SHARK_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB_RECURSE SHARK_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../include/*.h ../include/*.hpp ../include/*.inl)
list(REMOVE_ITEM SHARK_SRCS "Version.cpp")
# Group the source for MSVC
foreach(FILE ${SHARK_SRCS}) 
    # Get the directory of the source file
    get_filename_component(PARENT_DIR "${FILE}" PATH)
    # Make sure we are using windows slashes
    string(REPLACE "/" "\\" GROUP "${PARENT_DIR}")

    set(GROUP "src\\${GROUP}")

    source_group("${GROUP}" FILES "${FILE}")
endforeach()
#Also Group the Header Files
foreach(FILE ${SHARK_HEADERS}) 
    # Get the directory of the source file
    get_filename_component(PARENT_DIR "${FILE}" PATH)

	string(REPLACE "../" "" GROUP "${PARENT_DIR}")

    # Make sure we are using windows slashes
    string(REPLACE "/" "\\" GROUP "${GROUP}")

    source_group("${GROUP}" FILES "${FILE}")
endforeach()



# Create the shark library
add_library( shark ${SHARK_SRCS} ${SHARK_HEADERS} )
target_link_libraries( shark ${LINK_LIBRARIES})

set_target_properties( shark PROPERTIES DEBUG_POSTFIX _debug )
set_property(TARGET shark PROPERTY CXX_STANDARD 11)
set_property(TARGET shark PROPERTY CXX_STANDARD_REQUIRED ON)

if( WIN32 )
  set_target_properties( shark PROPERTIES DEBUG_PREFIX ../ )
endif()

# Add the version executable
add_executable(SharkVersion Version.cpp)
target_link_libraries(SharkVersion shark)
set_property(TARGET SharkVersion PROPERTY CXX_STANDARD 11)
set_property(TARGET SharkVersion PROPERTY CXX_STANDARD_REQUIRED ON)

# Install the library
set_target_properties( shark PROPERTIES
	VERSION ${SHARK_VERSION}
	SOVERSION ${SHARK_SOVERSION})

install(TARGETS shark
	    EXPORT SharkTargets
	    RUNTIME DESTINATION ${SHARK_INSTALL_BIN_DIR}
	    LIBRARY DESTINATION ${SHARK_INSTALL_LIB_DIR}
	    ARCHIVE DESTINATION ${SHARK_INSTALL_LIB_DIR})
	
install(TARGETS SharkVersion RUNTIME DESTINATION ${SHARK_INSTALL_BIN_DIR})
