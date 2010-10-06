
# 
# This is a QT qmake project file.
#
# Before using this file with qmake to generate a
# platform specific project file please make sure
# that the pathes to QWT fit your configuration.
# 

SHARKHOME = ../../../

CONFIG  = qt console warn_on

win32{
   QWTHOME = C:\qwt-5.0.2
   INCLUDEPATH += $${QWTHOME}/src/
   LIBS = -L$${SHARKHOME}/lib/winnt/release -L$${QWTHOME}/lib/ -lShark -lqwt
}

!win32{
   QWTHOME = /usr/local/qwt-6.0.0-rc1
   INCLUDEPATH += $${QWTHOME}/include/
   LIBS = -L$${SHARKHOME}/lib -L$${QWTHOME}/lib/ -lShark -lqwt
}

INCLUDEPATH += . $${SHARKHOME}/include

SOURCES = gaussian_process.cpp
HEADERS = gaussian_process.h
TARGET = gaussian_process
