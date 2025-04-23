CXXFLAG = -O3 -std=c++11 -g -I/usr/local/cuda-12.6/include
NVCCFLAG = $(CXXFLAG) -arch=sm_80  -Xcompiler="-fopenmp"  

obj/kde.o: src/kde.cu src/kde.h
	@mkdir -p obj
	nvcc $(NVCCFLAG)  src/kde.cu -dc -o obj/kde.o

obj/colvar.o: src/colvar.cpp src/colvar.h
	@mkdir -p obj
	g++ $(CXXFLAG) -c src/colvar.cpp -o obj/colvar.o -fopenmp

obj/kernel.o: src/kernel.cu src/kernel.h
	@mkdir -p obj
	nvcc $(NVCCFLAG) src/kernel.cu -dc -o obj/kernel.o

.PHONY: clean dm
dm: obj/kde.o obj/colvar.o obj/kernel.o main.cu 
	nvcc $(NVCCFLAG) src/main.cu obj/*.o -o dm

clean:
	@rm -rf obj
