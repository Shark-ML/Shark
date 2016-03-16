#-------------------------------------------------------------------------------
#
# openblasConfig.cmake :  CMake configuration file for external projects.
#-------------------------------------------------------------------------------

#Make sure that openblas_CONFIG_PATH is set here
GET_FILENAME_COMPONENT(openblas_CONFIG_PATH "${CMAKE_CURRENT_LIST_FILE}" PATH)

# The LIBopenblas  include file directories.
SET(openblas_INCLUDE_DIRS "${openblas_CONFIG_PATH}/../include")

# The LIBopenblas  library directories.
SET(openblas_LIBRARY_DIRS "${openblas_CONFIG_PATH}/../lib")

# A list of all libraries for openblas.  
SET(openblas_LIBRARIES libopenblas)

#To trigger using OpenBlas instead of ATLAS
ADD_DEFINITIONS(-DSHARK_USE_OPENBLAS)






