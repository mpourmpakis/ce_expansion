
.PHONY : clean all

all : _lib.so

lib.o : src/lib.c
	gcc -fPIC -O3 -c src/lib.c -o obj/lib.o

_lib.so : lib.o
	gcc -shared obj/lib.o -o bin/_lib.so

clean : 
	rm obj/lib.o
