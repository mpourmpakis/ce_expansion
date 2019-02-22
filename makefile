
.PHONY : clean all

all : release debug
release : _lib.so
debug : _lib_debug.so

lib_unix.o : src/lib.c
	gcc -fPIC -O3 -c src/lib.c -o obj/lib_unix.o

lib_debug_unix.o : src/lib.c
	gcc -fPIC -O3 -c src/lib.c -o obj/lib_debug_unix.o

_lib.so : lib_unix.o
	gcc -shared obj/lib_unix.o -o bin/_lib.so

_lib_debug.so : lib_debug_unix.o
	gcc -shared obj/lib_debug_unix.o -DPRINT_DEBUG_INFO -o bin/_lib_debug.so

clean : 
	rm obj/lib_unix.o
	rm obj/lib_debug_unix.o
