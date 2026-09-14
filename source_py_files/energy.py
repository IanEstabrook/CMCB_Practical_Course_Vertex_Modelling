#from .data_types import Vertex, Cell, Edge
import numpy as np


# Define the energies in the following on 3 scales: that of the tissue, that of a cell, and that of a vertex.

#############################################################

#                     Tissue Energies 

#############################################################
def energy_elasticity_tissue(tissue, ka):
    energy = 0.0
    for cell in tissue.cells:
        area = cell.area()
        energy += 0.5 * ka * (area - cell.A0)**2

    return energy

def energy_adhesion_tissue(tissue, Lambda):
    energy = 0.0
    for edge in tissue.edges:

        r1 = edge.v1.position[:2]
        r2 = edge.v2.position[:2]

        length = np.linalg.norm(r2 - r1)

        energy += Lambda * length


    return energy

def energy_contraction_tissue(tissue, kp):
    energy = 0.0
    for cell in tissue.cells:
        perimeter = cell.perimeter()
        energy += 0.5 * kp * (perimeter-cell.P0)**2
    return energy


def total_energy_tissue(tissue, ka=1.0, kp=1.0, Lambda=1.0):

    return (
        energy_elasticity_tissue(tissue, ka)
        + energy_contraction_tissue(tissue, kp)
        + energy_adhesion_tissue(tissue, Lambda)
    )
#############################################################

#                     Cell Energies 

#############################################################

def energy_elasticity_cell(cell, ka):
    area = cell.area()
    return 0.5 * ka * (area - cell.A0)**2

def energy_contraction_cell(cell, kp):
    perimeter = cell.perimeter()    
    return 0.5 * kp * (perimeter-cell.P0)**2

def cell_edges(cell):
    edges =[]

    for vertex in cell.vertices:
        #print("vertex edges:",vertex.edges)
        for edge in vertex.edges:
            
            if edge.v1 in cell.vertices and edge.v2 in cell.vertices:
                if edge not in edges:
                    edges.append(edge)
                
    #print("edges=",edges)
    return edges

def energy_adhesion_cell(cell, Lambda):
    energy = 0.0

    n=0;
    for edge in cell_edges(cell):
        r1 = edge.v1.position[:2]
        r2 = edge.v2.position[:2]

        length = np.linalg.norm(r2 - r1)
        energy += Lambda * length
        #n +=1
    #print("Adhesion energy:",energy,"length", length)
    #print("number of edges added for energy=",n)
    return energy

def local_cell_energy(tissue, vertex_id, ka=1.0, kp=1.0, Lambda=1.0):
    E = 0.0

    vertex = tissue.vertices[vertex_id]

    for cell in vertex.cells:
        #print(f"Cell {cell.id}: "f"A={cell.area():.3f}, "f"A0={cell.A0:.3f}, "f"P={cell.perimeter():.3f}, "f"P0={cell.P0:.3f} " )
    
        E += energy_elasticity_cell(cell, ka)
        E += energy_contraction_cell(cell, kp)
        E += energy_adhesion_cell(cell,Lambda)
    
    #print(f"Intermediate E,{E:.3f}");
    for p1 in vertex.periodic_partners:
        for cell in p1.cells:
     #       print(f"p1Cell {cell.id}: "f"A={cell.area():.3f}, "f"A0={cell.A0:.3f}, "f"P={cell.perimeter():.3f}, "f"P0={cell.P0:.3f} " )

            E += energy_elasticity_cell(cell, ka)
            E += energy_contraction_cell(cell, kp)
            E += energy_adhesion_cell(cell,Lambda)
    return E