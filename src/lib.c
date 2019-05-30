//BC Model Calculator
// James Dean, 2019

// Controls the printing of various statements describing the flow of the DLL, because this is less convenient
// to run through a more traditional debugger
#include <stdio.h>
const long int num_elements = 2;
const long int max_coordination = 13;

long int calculate_mixing(long int num_atoms, // Number of atoms in the system
                          long int num_bonds, // Number of bonds in the system
                          long int adj_table [num_bonds][2], // Adjacency table aka bondlist
                          long int id_array[num_atoms], // Representing the identity of each element
                          long int return_array[3]){ // Array holding the hetero/homoatomic bond counts respectively
    // Calculates the number of homo/heteroatomic bonds in the system
    // Changes array in place

    // Zero out return-array
    long int i=0;
    for (i=0; i < 3; i++){
      return_array[i] = 0;
    }

    // Count the bonds
    for (i=0; i < num_bonds; i++){
      long int bond_source = id_array[adj_table[i][0]];
      long int bond_destination = id_array[adj_table[i][1]];

      if (bond_source == bond_destination){
        if (bond_source == 1){
          // A-A Bond
          return_array[0] = return_array[0] + 1;
        } else {
          // B-B Bond
          return_array[1] = return_array[1] + 1;
        }
      } else {
        // A-B Bond
        return_array[2] = return_array[1] + 1;
      }
    }

    #ifdef PRINT_DEBUG_INFO
        printf("A-A Bonds: %d", return_array[0]);
        printf("B-B Bonds: %d", return_array[1]);
        printf("A-B Bonds: %d", return_array[2]);
    #endif

    // This double-counts the bonds due to the way we define the bond lists.
    // Hence, we always should have an even number for return_array entries.
    int incorrect_counting = 0;
    for (i=0; i < 3; i++){
      // Something has gone wrong if we get an odd number; check now:
      if (return_array[i] % 2 != 0){
        incorrect_counting = 1;
      }
      // Divide return_array entry by 2
      return_array[i] = return_array[i] / 2;
    }
    return incorrect_counting;
}

double calculate_ce(double bond_energies[num_elements][num_elements][max_coordination], //Table of bond energies
                    long int num_atoms, // Number of atoms in the nanoparticle
                    long int cns[num_atoms],
                    long int num_bonds, // Number of bonds in the system
                    long int adj_table [num_bonds][2], // Adjacency table
                    long int id_array[num_atoms]){ // Representing the identity of each element
  // Calculates the cohesive energy of the system
  // Loop over the bond system
  double cohesion = 0;
  long int i = 0;

	#ifdef PRINT_DEBUG_INFO
    long int j = 0;
    long int k = 0;
    //Debugging information
  	printf("Printing bond energies\n");
    for (i=0; i < 2*2*13; i++){
    printf("Element %d, " , i);
    printf("CN %d: %f\n", i%13, bond_energies[0][0][i]);
    }
  	printf("Printing the table way\n");
  	for (i=0; i < 2; i++){
      for (j=0; j < 2; j++){
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
    #ifdef PRINT_DEBUG_INFO
      // Print bond source/destination/coordination
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
      // Debug info is not printed
      long int bond_source = id_array[adj_table[i][0]];
      long int bond_destination = id_array[adj_table[i][1]];
      long int coordination = cns[adj_table[i][0]];

      // Add the bond energy to the running total
      // double contribution = bond_energies[bond_source][bond_destination][coordination];
      cohesion += bond_energies[bond_source][bond_destination][coordination];
    #endif
  }

  cohesion /= num_atoms;

  #ifdef PRINT_DEBUG_INFO
    printf("Dividing by %d atoms, resulting in %f\n", num_atoms, cohesion);
	#endif

  return cohesion;
}
