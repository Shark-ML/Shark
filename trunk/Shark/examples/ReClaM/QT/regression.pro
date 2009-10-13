SHARKHOME = ../../../

CONFIG  = qt console warn_on

INCLUDEPATH = . $${SHARKHOME}/include

win32{
   LIBS = -L$${SHARKHOME}/lib/winnt/release -lShark
}

!win32{
   LIBS = -L$${SHARKHOME}/lib/ -lShark
}

SOURCES = regression.cpp
HEADERS = regression.h
TARGET = regression
