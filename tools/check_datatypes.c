// C datatype length printer
// James Dean, 2019

// Code that shows the size of various relevant datatypes on the current system
// Used because the windows port of the code was having compatibility issues between the datatypes ctypes was casting
//      arrays into (e.g. the ctypes definition of c_long was 32 bits long, whereas the compiler definition was 64 bits)
#include <stdio.h>

int main(){
    //Integers
    short int sample_short;
    int sample_int;
    long int sample_long;
    long long int sample_long_long;
    printf("The size of a short integer is %lu\n", (unsigned long)sizeof(sample_short));
    printf("The size of an integer is %lu\n", (unsigned long)sizeof(sample_int));
    printf("The size of a long integer is %lu\n", (unsigned long)sizeof(sample_long));
    printf("The size of a long long integer is %lu\n", (unsigned long)sizeof(sample_long_long));


    //Floating Points
    float sample_float;
    double sample_double;
    long double sample_long_double;

    printf("The size of a float is %lu\n", (unsigned long)sizeof(sample_float));
    printf("The size of a double is %lu\n", (unsigned long)sizeof(sample_double));
    printf("The size of a long double is %lu\n", (unsigned long)sizeof(sample_long_double));

    return 0;
}