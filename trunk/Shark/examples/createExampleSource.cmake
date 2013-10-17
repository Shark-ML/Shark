FILE( READ ${TUT_TPP}  contents )
STRING( REGEX REPLACE "\t*//###[^\n]*\n" "" contents_cleared "${contents}")
FILE( WRITE ${TUT_CPP} "${contents_cleared}")