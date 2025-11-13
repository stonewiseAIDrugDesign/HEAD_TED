self_energies = []
d = {}
path='sae_linfit.dat'
with open(path) as f:
    for i in f:
        line = [x.strip() for x in i.split('=')]
        species = line[0].split(',')[0].strip()
        index = int(line[0].split(',')[1].strip())
        value = float(line[1])
        d[species] = value
        self_energies.append((index, value))
self_energies = [i for _, i in sorted(self_energies)]
print(self_energies)