CPPFLAGS=-O2 -Wall 

LIBS= -L /usr/include/
CXX=g++
export CPLUS_INCLUDE_PATH

L=../common
OBJS = vectors.o gzstream.o timer.o
INCS = $L/vectors.h $L/gzstream.h $L/timer.h $L/wrapper.h $L/assert.h

.PHONY : clean all

all: $(subst .cpp,.o,$(SOURCES))  lr flrl sgd flrl_predict lbfgs_lr pair_rank

vectors.o: $L/vectors.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ $L/vectors.cpp

gzstream.o: $L/gzstream.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ $L/gzstream.cpp

timer.o: $L/timer.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ $L/timer.cpp


%.O: %.cpp
		$(CXX) $(CPPFLAGS) ${LIBS} $^ $@
lr: lr.cpp data.o
		$(CXX) $(CPPFLAGS)  $^ ${LIBS} -o $@
lbfgs_lr:lbfgs_lr.cpp data.o
		$(CXX) $(CPPFLAGS)  $^ ${LIBS} -L./lib -llbfgs -o $@

sgd:sgd_lr.cpp data.o
		$(CXX) $(CPPFLAGS)  $^ ${LIBS} -L./lib -llbfgs -o $@

pair_rank:pair_rank.cpp data.o
		$(CXX) $(CPPFLAGS)  $^ ${LIBS} -L./lib -llbfgs -o $@

olr: online_svrg.cpp 
		$(CXX) $(CPPFLAGS) $^  ${LIBS} -o $@

flrl: flrl.cpp data.o
		$(CXX) $(CPPFLAGS) $^  ${LIBS} -o $@

flrl_predict: flrl_predict.cpp 
		$(CXX) $(CPPFLAGS) $^  ${LIBS} -o $@
clean:
		rm -rf  *.o  lr plr flrl flrl_predict lbfgs_lr pair_rank sgd
