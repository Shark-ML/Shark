SHARKHOME = ../../../

CONFIG  = qt console warn_on

INCLUDEPATH = . $${SHARKHOME}/include

win32{
   LIBS = -L$${SHARKHOME}/lib/winnt/release -lshark
}

!win32{
   LIBS = -L$${SHARKHOME}/lib/ -lshark
}

SOURCES = classification.cpp
HEADERS = classification.h
TARGET = classification
