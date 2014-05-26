CPPFLAGS=-O2 -Wall 

LIBS= -L /usr/include/
CXX=g++
export CPLUS_INCLUDE_PATH

.PHONY : clean all

all: $(subst .cpp,.o,$(SOURCES))  lr flrl flrl_predict lbfgs_lr

%.O: %.cpp
		$(CXX) $(CPPFLAGS) ${LIBS} $^ $@
lr: lr.cpp data.o
		$(CXX) $(CPPFLAGS)  $^ ${LIBS} -o $@
lbfgs_lr:lbfgs_lr.cpp data.o
		$(CXX) $(CPPFLAGS)  $^ ${LIBS} -L./lib -llbfgs -o $@

olr: online_lr.cpp 
		$(CXX) $(CPPFLAGS) $^  ${LIBS} -o $@

flrl: flrl.cpp data.o
		$(CXX) $(CPPFLAGS) $^  ${LIBS} -o $@

flrl_predict: flrl_predict.cpp 
		$(CXX) $(CPPFLAGS) $^  ${LIBS} -o $@
clean:
		rm -rf  *.o  lr plr flrl flrl_predict
