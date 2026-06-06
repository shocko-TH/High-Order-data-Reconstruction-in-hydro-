#usage: sh compile.sh filename.cpp

# # openmp flag of your compiler
OPENFLAG=-fopenmp
# #OPENFLAG=-openmp

# ==========================
IN=${1}

# CC=clang++
CC=g++
version=-std=c++17

OMP_PATH="/opt/homebrew/opt/libomp"

# FLAGS="-Xpreprocessor -fopenmp -I${OMP_PATH}/include -L${OMP_PATH}/lib -lomp"

# FLAGS="-Xpreprocessor ${OPENFLAG} -I${OMP_PATH}/include -L${OMP_PATH}/lib -lomp"


OUT=$(basename ${IN} .cpp).out

${CC} ${version} ${OPENFLAG} -O3 ${IN} -o ${OUT}
# ${CC} ${FLAGS} ${IN} -o ${OUT}

export OMP_NUM_THREADS=16
time ./${OUT}

# echo ${IN}
echo Threads = ${OMP_NUM_THREADS}