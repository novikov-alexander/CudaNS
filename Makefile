PROGRAM = PROG
BINDIR = bin
SRC = src
OBJ = obj
INC = include

CC = nvcc
C_LIB = -lm
COMMON_FLAGS = -w -m64 -Xptxas -dlcm=ca -D_FORCE_INLINES -Xcompiler -Wall -O3
CFLAGS = ${COMMON_FLAGS} -I${INC} -x cu -arch=sm_30
CLINKFLAGS = ${COMMON_FLAGS} -arch=sm_30

SRCS = $(wildcard $(SRC)/*.cu)
OBJS = $(patsubst $(SRC)/%.cu, $(OBJ)/%.o, $(SRCS))

# Use automatic variables to simplify rules
${BINDIR}/${PROGRAM}: ${OBJS}
	${CC} ${CLINKFLAGS} -o $@ $^ ${C_LIB}

${OBJ}/%.o: ${SRC}/%.cu | ${OBJ}
	${CC} ${CFLAGS} -MMD -MP -c $< -o $@

${OBJ}:
	mkdir -p ${OBJ}

# Include dependencies
-include $(OBJS:.o=.d)

clean:
	rm -rf ${OBJ}/*.o ${OBJ}/*.d ${BINDIR}/${PROGRAM}

cleanall: clean

# Declare these targets as phony
.PHONY: all clean cleanall