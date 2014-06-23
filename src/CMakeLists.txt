
FILE( GLOB SRCS
	Models/*.cpp
	ObjectiveFunctions/*.cpp
	ObjectiveFunctions/Loss/*.cpp
	Data/*.cpp
	Algorithms/*.cpp
	Algorithms/DirectSearch/*.cpp
	Algorithms/GradientDescent/*.cpp
	Fuzzy/*.cpp
)

SOURCE_GROUP( src/Models FILES Models/*.cpp )
SOURCE_GROUP( src/ObjectiveFunctions FILES ObjectiveFunctions/*.cpp )
SOURCE_GROUP( src/Algorithms FILES Algorithms/*.cpp )
SOURCE_GROUP( src/Fuzzy FILES Fuzzy/*.cpp )


#####################################################################
#                       Library setup                               #
#####################################################################

IF( WIN32 )
    ADD_LIBRARY( shark ${SRCS} )
    SET_TARGET_PROPERTIES( shark PROPERTIES DEBUG_PREFIX ../ )
    SET_TARGET_PROPERTIES( shark PROPERTIES DEBUG_POSTFIX _debug )
ELSE( WIN32 )
      ADD_LIBRARY( shark ${LIB_TYPE} ${SRCS} )
      SET_TARGET_PROPERTIES( shark PROPERTIES DEBUG_POSTFIX _debug )
ENDIF( WIN32 )

TARGET_LINK_LIBRARIES( shark ${LINK_LIBRARIES} )

ADD_EXECUTABLE( Version Core/Version.cpp )
TARGET_LINK_LIBRARIES( Version ${LINK_LIBRARIES} )

#####################################################################
#                       Install the library                         #
#####################################################################
SET_TARGET_PROPERTIES( shark PROPERTIES
	VERSION ${SHARK_VERSION_MAJOR}.${SHARK_VERSION_MINOR}.{SHARK_VERSION_PATCH}
	SOVERSION ${SHARK_VERSION_MAJOR}.${SHARK_VERSION_MINOR}.{SHARK_VERSION_PATCH}
)

INSTALL(
	TARGETS shark
	EXPORT SharkTargets
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
INSTALL(
	TARGETS Version
	EXPORT SharkTargets
	RUNTIME DESTINATION bin
)


