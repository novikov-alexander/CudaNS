PROGRAM=PROG
BINDIR = bin
SRC = src
OBJ = obj
INC = include

CC     = nvcc
CLINK  = $(CC)
C_LIB  = -lm
CFLAGS = -w -m64 -Xptxas -dlcm=ca -D_FORCE_INLINES -Xcompiler -Wall -O3 -I${INC} -x cu -arch=sm_30
CLINKFLAGS= -O3 -arch=sm_30

OBJS = ${OBJ}/_main_program.o \
	${OBJ}/initialize.o \
	${OBJ}/exact_solution.o \
	${OBJ}/exact_rhs.o \
	${OBJ}/set_constants.o \
	${OBJ}/adi.o \
	${OBJ}/compute_rhs.o \
	${OBJ}/x_solve.o \
	${OBJ}/y_solve.o \
	${OBJ}/z_solve.o \
	${OBJ}/error.o \
	${OBJ}/verify.o \
	${OBJ}/print_results.o \
	${OBJ}/timers.o

${BINDIR}/${PROGRAM}: ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${BINDIR}/${PROGRAM} ${OBJS} ${C_LIB}

${OBJ}/_main_program.o: ${SRC}/_main_program.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/_main_program.cu -o ${OBJ}/_main_program.o
${OBJ}/initialize.o: ${SRC}/initialize.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/initialize.cu -o ${OBJ}/initialize.o
${OBJ}/exact_solution.o: ${SRC}/exact_solution.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/exact_solution.cu -o ${OBJ}/exact_solution.o
${OBJ}/exact_rhs.o: ${SRC}/exact_rhs.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/exact_rhs.cu -o ${OBJ}/exact_rhs.o
${OBJ}/set_constants.o: ${SRC}/set_constants.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/set_constants.cu -o ${OBJ}/set_constants.o
${OBJ}/adi.o: ${SRC}/adi.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/adi.cu -o ${OBJ}/adi.o
${OBJ}/compute_rhs.o: ${SRC}/compute_rhs.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/compute_rhs.cu -o ${OBJ}/compute_rhs.o
${OBJ}/solve.o: ${SRC}/solve.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/solve.cu -o ${OBJ}/solve.o
${OBJ}/x_solve.o: ${SRC}/x_solve.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/x_solve.cu -o ${OBJ}/x_solve.o
${OBJ}/y_solve.o: ${SRC}/y_solve.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/y_solve.cu -o ${OBJ}/y_solve.o
${OBJ}/z_solve.o: ${SRC}/z_solve.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/z_solve.cu -o ${OBJ}/z_solve.o
${OBJ}/error.o: ${SRC}/error.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/error.cu -o ${OBJ}/error.o
${OBJ}/verify.o: ${SRC}/verify.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/verify.cu -o ${OBJ}/verify.o
${OBJ}/print_results.o: ${SRC}/print_results.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/print_results.cu -o ${OBJ}/print_results.o
${OBJ}/timers.o: ${SRC}/timers.cu ${INC}/header.hpp ${INC}/data_params.hpp
	$(CC) $(CFLAGS) -c ${SRC}/timers.cu -o ${OBJ}/timers.o

clean:
	rm -f ${OBJ}/*.o
cleanall:
	rm -f ${OBJ}/*.o ${BINDIR}/${PROGRAM}
