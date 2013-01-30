
GPP = g++
# GPP = gfilt
# GPP = gfilt /cand:L

COMPILERPARAMS = -DNDEBUG -I../../include/ -O3 -Wall -c
LINKERPARAMS = -lshark -L../../lib/ -lboost_serialization

# COMPILERPARAMS = -DDEBUG -I../../include/ -g -Wall -c
# LINKERPARAMS = -lshark_debug -L../../lib/ -lboost_serialization


all: CSvm CSvmGridSearch McSvm KernelRegression KernelSelection EvidenceOptimization


CSvm: CSvm.o
	${GPP} ${LINKERPARAMS} -o CSvm CSvm.o

CSvmGridSearch: CSvmGridSearch.o
	${GPP} ${LINKERPARAMS} -o CSvmGridSearch CSvmGridSearch.o

McSvm: McSvm.o
	${GPP} ${LINKERPARAMS} -o McSvm McSvm.o

KernelRegression: KernelRegression.o
	${GPP} ${LINKERPARAMS} -o KernelRegression KernelRegression.o

KernelSelection: KernelSelection.o
	${GPP} ${LINKERPARAMS} -o KernelSelection KernelSelection.o

EvidenceOptimization: EvidenceOptimization.o
	${GPP} ${LINKERPARAMS} -o EvidenceOptimization EvidenceOptimization.o

mytest: mytest.o
	${GPP} ${LINKERPARAMS} -o mytest mytest.o -lboost_serialization

%.o: %.cpp
	${GPP} ${COMPILERPARAMS} $<

clean:
	rm -f *.o *~ CSvm CSvmGridSearch McSvm KernelRegression KernelSelection mytest
