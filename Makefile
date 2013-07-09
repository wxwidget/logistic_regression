CPPFLAGS=-O2 -Wall 

LIBS= -L /usr/include/

export CPLUS_INCLUDE_PATH

.PHONY : clean all

all: $(subst .cpp,.o,$(SOURCES))  lr olr


%.O: %.cpp
		$(CXX) $(CPPFLAGS) ${LIBS} $^ $@
lr: lr.cpp 
		$(CXX) $(CPPFLAGS) $^  ${LIBS} -o $@

olr: online_lr.cpp 
		$(CXX) $(CPPFLAGS) $^  ${LIBS} -o $@
clean:
		rm -rf  *.o  lr plr
