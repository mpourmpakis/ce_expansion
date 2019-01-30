#include <stdio.h>
const long int num_elements = 2;
const long int max_coordination = 12;

long int fib(long int a);


double calculate_ce(double bond_energies[num_elements][num_elements][max_coordination], //Table of bond energies
                    long int num_atoms, // Number of atoms in the nanoparticle
                    long int cns[num_atoms], // Coordination numbers in system
                    long int num_bonds, // Number of bonds in the system
                    long int adj_table[num_bonds][2], // List of bonds in the system
                    long int id_string[num_atoms]); // String representing the identity of each element

/* Various test functions commented out below

long int print_array(long int* array, long int size){
        // Test function to print a 1D array
        // Works
        int i;
        for(i=0; i < size; i++){
                printf("%d\n",array[i]);
        }
        return 1;
}

long int print_2D(long int dim1, long int dim2, long int array[dim1][dim2]){
        // Test function to print a 2D array
        // Works
        int i, j;
        for(i=0; i < dim1; i++){
                for (j=0; j < dim2; j++){
                        printf("%d\n", array[i][j]);
                }
        }
        return 1;
}
                                        
long int print_3D(long int dim1, long int dim2, long int dim3, long int array[dim1][dim2][dim3]){
        //Test function to print a 3D array
        int i,j,k;
        for(i=0; i<dim1; i++){
                for(j=0;j<dim2;j++){
                        for(k=0;k<dim3;k++){
                                printf("%d\n",array[i][j][k]);
                        }
                }
        }
        return 1;
}
                                        
long int fib(long int a){
        // Test function to print the fibonacci sequence
                if (a <= 0)
                        return -1;
                else if (a == 1)
                        return 0;
                else if ((a == 2) || (a == 3))
                        return 1;
                else
                        return fib(a-2) + fib(a-1);
}



double debug_ce(double bond_energies[num_elements][num_elements][max_coordination], //Table of bond energies
                    long int num_atoms, // Number of atoms in the nanoparticle
                    long int cns[num_atoms], // Coordination numbers in system
                    long int num_bonds, // Number of bonds in the system
                    long int adj_table [num_bonds][2], // Adjacency table
                    char id_string[num_atoms] // String representing the identity of each element
                 ){
        printf("Number of atoms: %d\n", num_atoms);
        printf("Number of bonds: %d\n", num_bonds);
        printf("Printing CNs\n");
        long int i=0;
        for (i; i < num_atoms; i++){
            printf("Atom %d: %d\n",i, cns[i]);
        }
        printf("Printing ID string\n");
        i=0;
        for (i; i < num_atoms; i++){
            printf("Atom %d: %d\n", i, char_to_int(id_string[i]));
        }
        printf("Printing adjacency table\n");
        i=0;
        for (i; i < num_bonds; i++){
            printf("Source: %d, Destination: %d\n", adj_table[i][0], adj_table[i][1]);
        }
        printf("Printing bond energies table\n");
        i=0;
        int j = 0;
        int k = 0;
        for (i = 0; i < 2; i++){
            for (j = 0; j < 2; j++){
                for (k = 0; k < 12; k++){
                    double energy;
                    energy = bond_energies[i][j][k];
                    printf("Element %d with Element %d at CN %d: %f\n", i, j, k+1, energy);
                }
            }
        }
    printf("\n\n\nEntering main loop\n");
    // Loop over bond system:
    double cohesion = 0;
    i = 0;
    for (i; i < num_bonds; i++){
        long int bond_source = char_to_int(id_string[adj_table[i][0]]);
        long int bond_destination = char_to_int(id_string[adj_table[i][1]]);
        long int coordination = cns[adj_table[i][0]];
        printf("Source: %d Destination: %d Coordination: %d\n", bond_source, bond_destination, coordination);
        double energy = bond_energies[bond_source][bond_destination][coordination];
        printf("Energy: %f\n",energy);
        cohesion += energy;
    }
    cohesion /= num_atoms;
    printf("Cohesion: %f\n", cohesion);
    return cohesion;
}
*/

int char_to_int(char character){
    long int result = character - '0';
    return result;
}

double calculate_ce(double bond_energies[num_elements][num_elements][max_coordination], //Table of bond energies
                    long int num_atoms, // Number of atoms in the nanoparticle
                    long int cns[num_atoms], // Coordination numbers in system
                    long int num_bonds, // Number of bonds in the system
                    long int adj_table [num_bonds][2], // Adjacency table
                    long int id_string[num_atoms]){ // Representing the identity of each element
    // Loop over the bond system
    double cohesion = 0;
    long int i=0;
    for (i; i < num_bonds; i++){
        long int bond_source = id_string[adj_table[i][0]];
        long int bond_destination = id_string[adj_table[i][1]];
        long int coordination = cns[adj_table[i][0]];

        // Add the bond energy to the running total
        cohesion += bond_energies[bond_source][bond_destination][coordination];
    }
    cohesion /= num_atoms;
    return cohesion;
}
