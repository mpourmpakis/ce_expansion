.PHONY : clean_unix clean_win64

win64 : release_win64 debug_win64
release_win64 : _lib.dll
debug_win64 : _lib_debug.dll     

unix : release_unix debug_unix
release_unix : _lib.so
debug_unix : _lib_debug.so

lib_unix.o : src/lib.c
	gcc -fPIC -O3 -c src/lib.c -o obj/lib_unix.o
lib_debug_unix.o : src/lib.c
	gcc -fPIC -O3 -c src/lib.c -D PRINT_DEBUG_INFO  -o obj/lib_unix.o
_lib.so : lib_unix.o
	gcc -shared obj/lib_unix.o -o bin/_lib.so
_lib_debug.so : lib_debug_unix.o
	gcc -shared obj/lib_debug_unix.o -DPRINT_DEBUG_INFO -o bin/_lib_debug.so

lib_win64.o : src/lib.c
	gcc -fPIC -O3 -c src/lib.c -o obj/lib_win64.o
lib_debug_win64.o : src/lib.c
	gcc -fPIC -O3 -c src/lib.c -D PRINT_DEBUG_INFO  -o obj/lib_debug_win64.o
_lib.dll : lib_win64.o
	gcc -shared obj/lib_win64.o -o bin/_lib.dll
_lib_debug.dll : lib_debug_win64.o
	gcc -shared obj/lib_debug_win64.o -o bin/_lib_debug.dll

clean_unix : 
	rm obj/lib_unix.o

clean_win64:
	del obj/lib_win.o
