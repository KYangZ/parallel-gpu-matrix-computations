tacc_path := koryyang@maverick2.tacc.utexas.edu:/home1/08667/koryyang/

build: det.out det_seq.out det_test.out

det.out: determinant.cu
	nvcc -arch=sm_35 -rdc=true -lineinfo -lcudadevrt -o det.out determinant.cu

det_seq.out: determinant_seq.cu
	nvcc -o det_seq.out determinant_seq.cu

det_test.out: determinant_seq.c
	gcc -o det_test.out determinant_seq.c

upload:
	scp determinant.cu determinant_seq.cu determinant_seq.c Makefile tacc-batch $(tacc_path)determinant/

upload_full:
	scp -r ../determinant $(tacc_path)

download:
	scp $(tacc_path)determinant/*_out.txt .

download_full:
	scp -r $(tacc_path)determinant/ ..

clean:
	rm *.out det_out.txt det_seq_out.txt matrix_det*
