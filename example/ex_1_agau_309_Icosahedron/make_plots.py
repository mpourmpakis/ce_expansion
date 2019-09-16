import matplotlib.pyplot as plt
import pandas

DEFAULT_DPI = 1200

# The following creates a plot that overlays the mixing parameter with BC-calculated excess energy
csv = pandas.read_csv("mixing_parameter_data.csv")
fig, ax0 = plt.subplots()
ax0.plot(csv['Ag'], csv['Mixing_Parameter'], color="#1C2957")
ax0.set_ylabel("Mixing Parameter (unitless)")
plt.xlabel("Ag")
plt.title("Structures Reported by Larson et al")
ax1 = ax0.twinx()
ax1.plot(csv['Ag'], csv['Excess_Energy_(eV)'], color="#CDB87D")
ax1.set_ylabel("BC Excess Energy (eV)")
lines = ax0.get_lines() + ax1.get_lines()
plt.legend(lines, [line.get_label() for line in lines], fontsize=20, loc="upper center")
plt.tight_layout()
plt.savefig("Mixing_Parameter_Excess_Energy_Overlay.png", dpi=DEFAULT_DPI)
plt.close()

# The following compares our lowest-energy structures with Larson by way of Excess Energy

# Make the plot:
plt.plot(csv['Ag'], csv["Our_Best_EE(eV)"], color="#1C2957", label="GA Results")
plt.plot(csv['Ag'], csv['Excess_Energy_(eV)'], color="#CDB87D", label="Larson et al")
plt.title("Comparison of Lowest-Energy Structures")
plt.xlabel("Ag")
plt.ylabel("BC Excess Energy (eV)")
plt.legend(loc="upper center", fontsize=20)
plt.tight_layout()
plt.show()
#plt.savefig("Excess_Energy_Comparison.png", dpi=DEFAULT_DPI)
plt.close()
