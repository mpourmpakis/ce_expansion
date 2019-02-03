//BC Model Calculator
// James Dean, 2019

#define COMPILE_FOR_WINDOWS 1
#define PREPROCESSOR_PRINT_DEBUG_INFO 1

#include <stdio.h>
#include <stdint.h>
const long int num_elements = 2;
const long int max_coordination = 13;

int char_to_int(char character){
    long int result = character - '0';
    return result;
}

double calculate_ce(double bond_energies[num_elements][num_elements][max_coordination], //Table of bond energies
                    long int num_atoms, // Number of atoms in the nanoparticle
                    #if COMPILE_FOR_WINDOWS
                        int64_t cns[num_atoms],
                    #else
                        long int cns[num_atoms],
                    #endif
                    long int num_bonds, // Number of bonds in the system
                    long int adj_table [num_bonds][2], // Adjacency table
                    long int id_array[num_atoms]){ // Representing the identity of each element
    // Loop over the bond system
    double cohesion = 0;
    long int i=0;

	#if PREPROCESSOR_PRINT_DEBUG_INFO
    //Debugging information
	printf("Printing bond energies\n");
    for (i=0; i < 2*2*13; i++){
		printf("Element %d, " , i);
        printf("CN %d: %f\n", i%13, bond_energies[0][0][i]);
    }
	printf("Printing the table way\n");
	for (i=0; i < 2; i++){
		int j = 0;
		for (j=0; j < 2; j++){
			int k = 0;
			for (k = 0; k < 13; k++){
				printf("Index (%d, %d), CN %d: %f\n", i, j, k, bond_energies[i][j][k]);
			}
		}
        
    }

	for(i=0; i < num_atoms; i++){
		printf("Atom %d: Coordination %d\n", i, cns[i]);
	}
    #endif

    for (i=0; i < num_bonds; i++){
        #if PREPROCESSOR_PRINT_DEBUG_INFO
		printf("Bond %d, ", i);
        long int bond_source = id_array[adj_table[i][0]];
		printf("Source kind = %d, ", bond_source);
        long int bond_destination = id_array[adj_table[i][1]];
		printf("Source destination = %d\n", bond_destination);
        long int coordination = cns[adj_table[i][0]];
		printf("Coordination = %d, ", coordination);
        // Add the bond energy to the running total
		double contribution = bond_energies[bond_source][bond_destination][coordination];
        cohesion += bond_energies[bond_source][bond_destination][coordination];
		// Print debug stuff
		printf("bond energy = %f\n", contribution);
		printf("Total cohesion is now: %f\n", cohesion);

		#else
		printf("Bond %d, ", i);
        long int bond_source = id_array[adj_table[i][0]];
        long int bond_destination = id_array[adj_table[i][1]];
        long int coordination = cns[adj_table[i][0]];

        // Add the bond energy to the running total
		double contribution = bond_energies[bond_source][bond_destination][coordination];
        cohesion += bond_energies[bond_source][bond_destination][coordination];
        #endif
    }
    cohesion /= num_atoms;
    #if PREPROCESSOR_PRINT_DEBUG_INFO
	printf("Dividing by %d atoms, resulting in %f\n", num_atoms, cohesion);
	#endif
    return cohesion;
}
