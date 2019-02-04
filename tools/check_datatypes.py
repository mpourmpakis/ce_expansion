# Use in conjunction with check_datatypes.exe; prints out the datatypes that ctypes is using

import ctypes

print("The size of a short is: "  + str(ctypes.sizeof(ctypes.c_short)))
print("The size of an integer is: "  + str(ctypes.sizeof(ctypes.c_int)))
print("The size of a long integer is: "  + str(ctypes.sizeof(ctypes.c_long)))
print("The size of a long long integer is: "  + str(ctypes.sizeof(ctypes.c_longlong)))
print("The size of a float is: "  + str(ctypes.sizeof(ctypes.c_float)))
print("The size of a double is: "  + str(ctypes.sizeof(ctypes.c_double)))
print("The size of a long double is: "  + str(ctypes.sizeof(ctypes.c_longdouble)))
