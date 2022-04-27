build: lin_sys.out lin_sys_seq.out

lin_sys.out: lin_sys.cu
	nvcc -arch=sm_35 -rdc=true -lineinfo -lcudadevrt -o lin_sys.out lin_sys.cu

lin_sys_seq.out: lin_sys_seq.cu
	nvcc -o lin_sys_seq.out lin_sys_seq.cu

clean:
	rm *.out