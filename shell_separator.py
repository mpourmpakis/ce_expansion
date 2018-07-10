import ase.cluster
import math
import ase.io


#def shell_divider(shell_number,atom):


shell_num=4

nanop =ase.cluster.Icosahedron("Cu",shell_num)#ase.cluster.Icosahedron("Cu",shell_num) #ase.io.read("shell_divider.xyz")   #
nanop_list=nanop.get_chemical_symbols()
n=0
magic_number= (2*n+1)*(5*n**2+5*n+3)/3
i=0
z=[]#[None]*(shell_num)
shell_storage=[None]*(shell_num)

while magic_number < len(nanop_list):
    magic_number = (2*n+1)*(5*n**2+5*n+3)/3
    shell_storage[i]=magic_number
    print shell_storage[i]
    z.append(nanop_list[shell_storage[i-1]:shell_storage[i]])
    print z
    n+=1
    i+=1

#shell_divider(3,ase.cluster.Icosahedron("Cu",4))
#j=0

bnp=[None]*(shell_num+1)
k=0
shell_storage.insert(0,0)
for atomnum in shell_storage:
    if atomnum==0:
        bnp[k]=nanop[0]
        print nanop[k]
       
    else:
        bnp[k]=nanop[shell_storage[k-1]:shell_storage[k]]
    
    k+=1
    print bnp[k]    
