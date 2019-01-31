#include <stdio.h>
const long int num_elements = 2;
const long int max_coordination = 12;

long int fib(long int a);

int char_to_int(char character){
    long int result = character - '0';
    return result;
}

double calculate_ce(double bond_energies[num_elements][num_elements][max_coordination], //Table of bond energies
                    long int num_atoms, // Number of atoms in the nanoparticle
                    long int cns[num_atoms], // Coordination numbers in system
                    long int num_bonds, // Number of bonds in the system
                    long int adj_table [num_bonds][2], // Adjacency table
                    long int id_array[num_atoms]){ // Representing the identity of each element
    // Loop over the bond system
    double cohesion = 0;
    long int i=0;
    for (i; i < num_bonds; i++){
        long int bond_source = id_array[adj_table[i][0]];
        long int bond_destination = id_array[adj_table[i][1]];
        long int coordination = cns[adj_table[i][0]];

        // Add the bond energy to the running total
        cohesion += bond_energies[bond_source][bond_destination][coordination];
    }
    cohesion /= num_atoms;
    return cohesion;
}
