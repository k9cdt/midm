CXXFLAG = -O3 -std=c++11 -g -I/usr/local/cuda-12.6/include
NVCCFLAG = $(CXXFLAG) -arch=sm_80  -Xcompiler="-fopenmp"  

obj/kde.o: kde.cu kde.h
	@mkdir -p obj
	nvcc $(NVCCFLAG)  kde.cu -dc -o obj/kde.o

obj/colvar.o: colvar.cpp colvar.h
	@mkdir -p obj
	g++ $(CXXFLAG) -c colvar.cpp -o obj/colvar.o -fopenmp

obj/kernel.o: kernel.cu kernel.h
	@mkdir -p obj
	nvcc $(NVCCFLAG) kernel.cu -dc -o obj/kernel.o

.PHONY: clean dm
dm: obj/kde.o obj/colvar.o obj/kernel.o main.cu 
	nvcc $(NVCCFLAG) main.cu obj/*.o -o dm

clean:
	@rm -rf obj
