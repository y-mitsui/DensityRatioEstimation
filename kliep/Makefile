install_dir=/usr/local/lib/python3.5/dist-packages/


cuda_kliep2:
	nvcc -I/usr/local/magma/include -O2 -g -arch=sm_20 -c -Xcompiler -fPIC cuda_kliep.cu  -o cuda_kliep.o
	cython --cplus cuda_kliep_wrap.pyx
	x86_64-linux-gnu-gcc -pthread -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fno-strict-aliasing -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security -fPIC -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -I/usr/include/python2.7 -c cuda_kliep_wrap.cpp -o cuda_kliep_wrap.o
	c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wl,-Bsymbolic-functions -Wl,-z,relro -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security cuda_kliep_wrap.o ./cuda_kliep.o -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -L/opt/OpenBLAS/lib -L/usr/local/cuda-7.5/lib64 -L/usr/local/magma/lib/  -lm -lgsl -lopenblas -lcudart -lcusolver -lcublas -lmagma -lgomp -o cuda_kliep_wrap.so

install:
	install cuda_lsif.so $(install_dir)
	
clean:
	rm cuda_lsif.cpp
	rm lsif.o
	rm cuda_lsif.o
	rm cuda_lsif.so

