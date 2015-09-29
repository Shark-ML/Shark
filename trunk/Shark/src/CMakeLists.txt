# Glob the files
file( GLOB SHARK_SRCS
	Models/*.cpp
	ObjectiveFunctions/*.cpp
	ObjectiveFunctions/Loss/*.cpp
	Data/*.cpp
	Algorithms/*.cpp
	Algorithms/DirectSearch/*.cpp
	Algorithms/GradientDescent/*.cpp
	Fuzzy/*.cpp
)

# Group the source for MSVC
source_group( src/Models FILES Models/*.cpp )
source_group( src/ObjectiveFunctions FILES ObjectiveFunctions/*.cpp )
source_group( src/Algorithms FILES Algorithms/*.cpp )
source_group( src/Fuzzy FILES Fuzzy/*.cpp )

# Create the shark library
add_library( shark ${SHARK_SRCS} )
target_link_libraries( shark ${LINK_LIBRARIES})

set_target_properties( shark PROPERTIES DEBUG_POSTFIX _debug )

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


