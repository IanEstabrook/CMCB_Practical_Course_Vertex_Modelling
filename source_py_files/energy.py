#from .data_types import Vertex, Cell, Edge
import numpy as np

def energy_elasticity(tissue, ka):
    energy = 0.0
    for cell in tissue.cells:
        area = cell.area()
        energy += 0.5 * ka * (area - cell.A0)**2

    return energy

def energy_adhesion(tissue, Lambda):
    energy = 0.0
    for edge in tissue.edges:

        r1 = edge.v1.position[:2]
        r2 = edge.v2.position[:2]

        length = np.linalg.norm(r2 - r1)

        energy += Lambda * length


    return energy

def energy_contraction(tissue, gamma):
    energy = 0.0
    for cell in tissue.cells:
        perimeter = cell.perimeter()
        energy += 0.5 * gamma * perimeter**2
    return energy
