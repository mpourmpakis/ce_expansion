ifeq ($(OS), Windows_NT)
    release : _lib.dll
    debug : _lib_debug.dll
    clean :
	    del obj/*.o
else
    release : _lib.so
    debug : _lib_debug.so
    clean :
	    rm obj/*.o
endif

.PHONY : clean

lib_unix.o : ce_expansion/lib.c
	gcc -fPIC -O3 -c atomgraph/lib.c -o obj/lib_unix.o
lib_debug_unix.o : atomgraph/lib.c
	gcc -fPIC -O3 -c atomgraph/lib.c -D PRINT_DEBUG_INFO  -o obj/lib_debug_unix.o
_lib.so : lib_unix.o
	gcc -shared obj/lib_unix.o -o bin/_lib.so
_lib_debug.so : lib_debug_unix.o
	gcc -shared obj/lib_debug_unix.o -DPRINT_DEBUG_INFO -o bin/_lib_debug.so

lib_win64.o : ce_expansion/lib.c
	gcc -fPIC -O3 -c atomgraph/lib.c -o obj/lib_win64.o
lib_debug_win64.o : atomgraph/lib.c
	gcc -fPIC -O3 -c atomgraph/lib.c -D PRINT_DEBUG_INFO  -o obj/lib_debug_win64.o
_lib.dll : lib_win64.o
	gcc -shared obj/lib_win64.o -o bin/_lib.dll
_lib_debug.dll : lib_debug_win64.o
	gcc -shared obj/lib_debug_win64.o -o bin/_lib_debug.dll

