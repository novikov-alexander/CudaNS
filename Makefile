PROGRAM = PROG
BINDIR = bin
SRC = src
OBJ = obj
INC = include

CC = nvcc
COMMON_FLAGS = -w -m64 -Xptxas -dlcm=ca -D_FORCE_INLINES -Xcompiler -Wall -O3
ARCH = sm_60
CFLAGS = ${COMMON_FLAGS} -I${INC} -x cu -arch=${ARCH} -std=c++20
CLINKFLAGS = ${COMMON_FLAGS} -arch=${ARCH}

ifeq ($(OS),Windows_NT)
    C_LIB =
else
    C_LIB = -lm
endif

SRCS = $(wildcard $(SRC)/*.cu)

OBJS = $(patsubst $(SRC)/%.cu, $(OBJ)/%.obj, $(SRCS))

${BINDIR}/${PROGRAM}: ${OBJS} | ${BINDIR}
	${CC} ${CLINKFLAGS} -o $@ $^ ${C_LIB}

${OBJ}/%.obj: ${SRC}/%.cu | ${OBJ}
	${CC} ${CFLAGS} -c $< -o $@

${OBJ}:
	mkdir -p ${OBJ}

${BINDIR}:
	mkdir -p ${BINDIR}

-include $(OBJS:.obj=.d)

clean:
	rm -rf ${OBJ}/*.obj ${OBJ}/*.d ${BINDIR}/${PROGRAM}

cleanall: clean

.PHONY: all clean cleanall
