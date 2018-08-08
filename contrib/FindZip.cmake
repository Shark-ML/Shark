# Searches for an installation of the zip library. On success, it sets the following variables:
#
#   Zip_FOUND              Set to true to indicate the zip library was found
#   Zip_INCLUDE_DIRS       The directory containing the header file zip/zip.h
#   Zip_LIBRARIES          The libraries needed to use the zip library
#
# To specify an additional directory to search, set Zip_ROOT.
#
# Author: Siddhartha Chaudhuri, 2009, adopted and fixed by Oswin Krause, 2018
#

# Look for the header, first in the user-specified location and then in the system locations
SET(Zip_INCLUDE_DOC "The directory containing the header file zip.h")
FIND_PATH(Zip_INCLUDE_DIRS NAMES zip.h PATHS ${Zip_ROOT} ${Zip_ROOT}/include ${Zip_ROOT}/include/zip/ DOC ${Zip_INCLUDE_DOC} NO_DEFAULT_PATH)
IF(NOT Zip_INCLUDE_DIRS)  # now look in system locations
  FIND_PATH(Zip_INCLUDE_DIRS NAMES zip.h DOC ${Zip_INCLUDE_DOC})
ENDIF()

SET(Zip_FOUND FALSE)

IF(Zip_INCLUDE_DIRS)
  SET(Zip_LIBRARY_DIRS ${Zip_INCLUDE_DIRS})

IF("${Zip_LIBRARY_DIRS}" MATCHES "/zip$")
    # Strip off the trailing "/zip" in the path.
    GET_FILENAME_COMPONENT(Zip_LIBRARY_DIRS ${Zip_LIBRARY_DIRS} PATH)
  ENDIF()

  IF("${Zip_LIBRARY_DIRS}" MATCHES "/include$")
    # Strip off the trailing "/include" in the path.
    GET_FILENAME_COMPONENT(Zip_LIBRARY_DIRS ${Zip_LIBRARY_DIRS} PATH)
  ENDIF()
  
  message(STATUS ${Zip_LIBRARY_DIRS})

  # Find Zip libraries
  FIND_LIBRARY(Zip_DEBUG_LIBRARY NAMES zipd zip_d libzipd libzip_d
               PATH_SUFFIXES Debug ${CMAKE_LIBRARY_ARCHITECTURE} ${CMAKE_LIBRARY_ARCHITECTURE}/Debug
               PATHS ${Zip_LIBRARY_DIRS} NO_DEFAULT_PATH)
  FIND_LIBRARY(Zip_RELEASE_LIBRARY NAMES zip libzip
               PATH_SUFFIXES Release ${CMAKE_LIBRARY_ARCHITECTURE} ${CMAKE_LIBRARY_ARCHITECTURE}/Release
               PATHS ${Zip_LIBRARY_DIRS}/lib/ ${Zip_LIBRARY_DIRS}/lib64/  NO_DEFAULT_PATH)

  SET(Zip_LIBRARIES )
  IF(Zip_DEBUG_LIBRARY AND Zip_RELEASE_LIBRARY)
    SET(Zip_LIBRARIES debug ${Zip_DEBUG_LIBRARY} optimized ${Zip_RELEASE_LIBRARY})
    GET_FILENAME_COMPONENT(Zip_RELEASE_LIB_DIR ${Zip_RELEASE_LIBRARY} PATH)
    GET_FILENAME_COMPONENT(Zip_DEBUG_LIB_DIR ${Zip_RELEASE_LIBRARY} PATH)
    SET(Zip_LIBRARY_DIRS ${Zip_RELEASE_LIB_DIR} ${Zip_DEBUG_LIB_DIR})
  ELSEIF(Zip_DEBUG_LIBRARY)
    SET(Zip_LIBRARIES ${Zip_DEBUG_LIBRARY})
    GET_FILENAME_COMPONENT(Zip_LIBRARY_DIRS ${Zip_DEBUG_LIBRARY} PATH)
  ELSEIF(Zip_RELEASE_LIBRARY)
    SET(Zip_LIBRARIES ${Zip_RELEASE_LIBRARY})
    GET_FILENAME_COMPONENT(Zip_LIBRARY_DIRS ${Zip_RELEASE_LIBRARY} PATH)
  ENDIF()

  IF(Zip_LIBRARIES)
    SET(Zip_FOUND TRUE)
  ENDIF()
ENDIF()

IF(Zip_FOUND)
  IF(NOT Zip_FIND_QUIETLY)
    MESSAGE(STATUS "Found Zip: headers at ${Zip_INCLUDE_DIRS}, libraries at ${Zip_LIBRARY_DIRS}")
  ENDIF(NOT Zip_FIND_QUIETLY)
ELSE(Zip_FOUND)
  IF(Zip_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Zip library not found")
  ENDIF(Zip_FIND_REQUIRED)
ENDIF(Zip_FOUND)