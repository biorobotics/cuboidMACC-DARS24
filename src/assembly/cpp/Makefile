MACC_HOME=/home/ggutow/eclipse-workspace/macc_general_mamba
SRC=${MACC_HOME}/src/assembly/cpp/
INCLUDE=${MACC_HOME}/src/assembly/cpp/
BUILD=${SRC}
PYBIND_INCLUDE=/home/ggutow/eclipse-workspace/pybind11/include
PYTHON_INCLUDE=/home/ggutow/miniconda3/envs/macc/include/python3.12

OTHER_LIBS=-lm

PROFILE_FLAGS=-fPIC -O3 -pg
BUILD_FLAGS=-fPIC -O3 -g
INCLUDE_FLAGS=-I${INCLUDE} -I${PYBIND_INCLUDE} -I${PYTHON_INCLUDE}
LINKER_FLAGS=-L${OTHER_LIBS}# -pg
PROFILE_LINKER_FLAGS=-L${OTHER_LIBS} -pg
BINDING_BUILD_FLAGS=-shared

PYTHON_EXT_SUFFIX=.cpython-312-x86_64-linux-gnu.so

EXT_OBJS=${BUILD}/reachability${PYTHON_EXT_SUFFIX} 
OBJS=${EXT_OBJS}

reachability: reachability.h reachability.cpp
	g++ ${BUILD_FLAGS} ${BINDING_BUILD_FLAGS} -o reachability${PYTHON_EXT_SUFFIX} reachability.h reachability.cpp ${LINKER_FLAGS} ${INCLUDE_FLAGS}	