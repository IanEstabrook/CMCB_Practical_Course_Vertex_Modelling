
#from .data_types import Vertex, Cell, Edge
import numpy as np

def cell_area(vertices):
    #Calculate the area of a 2D polygon using the shoelace formula.

    #The shoelace formula, and cell area only makes physical sense if there are >=3 vertices.
    if len(vertices) < 3:
        return 0.0

    # Extract x and y coordinates
    x = np.array([vertex.position[0] for vertex in vertices])
    y = np.array([vertex.position[1] for vertex in vertices])

    # Shoelace formula
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    return area

def cell_perimeter(vertices):
    perimeter=0;
    #Sum over all the edge lengths by using 
    
    x = np.array([vertex.position[0] for vertex in vertices])
    y = np.array([vertex.position[1] for vertex in vertices])

    # Shoelace formula
    perimeter =np.sum(np.sqrt((x - np.roll(x, -1))**2 + (y - np.roll(y, -1))**2) )

    return perimeter


def edge_length(v1,v2):
    pos1=np.asarray(v1.position[:2])
    pos2=np.asarray(v2.position[:2])
    return np.linalg.norm(pos1-pos2);

    