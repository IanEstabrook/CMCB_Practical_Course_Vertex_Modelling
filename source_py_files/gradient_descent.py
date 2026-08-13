from dataclasses import dataclass, field
import numpy as np
from .geometry import cell_area, cell_perimeter

#Introduce classes for : Vertices, Cells and Edges. Cells contain a list of vertices, which are shared between
#Different cells. Edges join two cells as lines between vertices.. 
@dataclass
class Vertex:
    id: int #Each vertex has a unique identifier, which is shared between cells. It contains a position and the cells which this vertex is part of.
    position: np.ndarray #A 2d position of the cell vertex.
    cells: list = field(default_factory=list) #The cells the vertex belongs to
    periodic_partners: list = field(default_factory=list) #We will need to track which, if any vertices are periodic over the boundaries. 
    edges: list = field(default_factory=list)

@dataclass
class Cell:
    id: int #Each cell is also labelled with an identifier. It contains no other information other than the centre of the cell
    i: int #i,j are used only for the initialisation step where each cell is a periodic lattice and we need to know the boundaries to define periodic conditions
    j: int
    centre: np.ndarray #Cell centre        
    A0: float = 0.0
    P0: float = 0.0
    vertices: list = field(default_factory=list) #Each cell has a list of vertices
    
    def area(self):
        return cell_area(self.vertices)

    def perimeter(self):
        return cell_perimeter(self.vertices)

@dataclass
class Edge:
    id: int
    v1: Vertex #Each edge joins between Two vertices. It is then a part of two cells, whichever cells share the boundaries. 
    v2: Vertex
    length: float
    cells: list = field(default_factory=list)
    
