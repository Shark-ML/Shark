SET(NonModularLibrary_hdf5 ON CACHE BOOL "Use hdf5 Library" FORCE)
SET(NonModularLibrary_openblas ON CACHE BOOL "Use OpenBlas Library" FORCE)
ADD_DEFINITIONS(-DSHARK_USE_OPENBLAS)
SET(MODULAR_MANDATORY_LIB_LIST boost-1_59)
SET(BOOST_MANDATORY_COMPONENTS
    thread
    system
    date_time
    program_options 
    serialization
    filesystem
	  unit_test_framework
)
FIND_PACKAGE(ThirdPartyLibraries_VS2015 REQUIRED)
list(APPEND LINK_LIBRARIES ${EXT_LIBS}) 