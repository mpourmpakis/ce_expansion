import matplotlib.pyplot as plt
import pandas

import npdb.db_inter

# The following creates a plot that overlays the mixing parameter with excess energy
csv = pandas.read_csv("mixing_parameter_data.csv")
fig, ax0 = plt.subplots()
ax0.plot(csv['Ag'], csv['Mixing_Parameter'], color="#1C2957")
ax0.set_ylabel("Mixing Parameter (unitless)")

ax1 = ax0.twinx()
ax1.plot(csv['Ag'], csv['Excess_Energy_(eV)'], color="#CDB87D")
ax1.set_ylabel("Excess Energy (eV)")

lines = ax0.get_lines() + ax1.get_lines()
plt.legend(lines, [line.get_label() for line in lines], fontsize=20, loc="upper center")
plt.savefig("Mixing_Parameter_Excess_Energy_Overlay.png", dpi=1200)
plt.close()

# The following compares the excess energies calculated by the BC model
#   for the structures reported by Larson et al and calculated by us

# Build up a list of 309-atom icosahedral AgAu NPs
results = npdb.db_inter.get_bimet_result(metals="AgAu", shape="icosahedron", num_atoms=309)
ag = list(map(lambda i: i.n_metal1, results))
excess_energy = list(map(lambda i: i.EE, results))

# Make the plot:
plt.plot(ag, excess_energy, color="#1C2957", label="BC Model")
plt.plot(csv['Ag'], csv['Excess_Energy_(eV)'], color="#CDB87D", label="Larson et al")
plt.xlabel("Ag")
plt.ylabel("Excess Energy (eV)")
plt.legend(loc="upper center", fontsize=20)
plt.savefig("Excess_Energy_Comparison.png", dpi=1200)

