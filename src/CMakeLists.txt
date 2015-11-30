# Glob the files
file( GLOB_RECURSE SHARK_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB_RECURSE SHARK_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../include/*.h ../include/*.hpp ../include/*.inl)

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
set_target_properties( shark PROPERTIES VERSION ${SHARK_VERSION_MAJOR}.${SHARK_VERSION_MINOR}.${SHARK_VERSION_PATCH} SOVERSION 0 )

if( WIN32 )
  set_target_properties( shark PROPERTIES DEBUG_PREFIX ../ )
endif()

# Add the version executable
add_executable(SharkVersion Core/Version.cpp)
target_link_libraries(SharkVersion shark)

# Install the library
set_target_properties( shark PROPERTIES
	VERSION ${SHARK_VERSION_MAJOR}.${SHARK_VERSION_MINOR}.${SHARK_VERSION_PATCH}
	SOVERSION ${SHARK_VERSION_MAJOR}.${SHARK_VERSION_MINOR}.${SHARK_VERSION_PATCH})

install(TARGETS shark
	    EXPORT SharkTargets
	    RUNTIME DESTINATION bin
	    LIBRARY DESTINATION lib
	    ARCHIVE DESTINATION lib)
	
install(TARGETS SharkVersion
	    EXPORT SharkTargets
	    RUNTIME DESTINATION bin)


