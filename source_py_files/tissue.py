##########File created by Ian Estabrook for the 

from .data_types import Vertex, Cell, Edge
import numpy as np

class Tissue:
    def __init__(self):

        self.vertices = []
        self.cells = []
        self.edges = []
        # Used to identify already-created vertices
        self.vertex_lookup = {}
        self.edge_lookup = {}

    def get_vertex(self, x, y, tol=8):
        #Returns an existing vertex if one already exists, otherwise creates a new one.

        key = (round(x, tol), round(y, tol))

        if key in self.vertex_lookup:
            return self.vertex_lookup[key]

        v = Vertex(id=len(self.vertices),position=np.array([x, y, 0.0]))

        self.vertices.append(v)
        self.vertex_lookup[key] = v

        return v

    def get_edge(self, v1, v2):
            """Return an existing edge or create a new one."""

            # An edge is undirected:
            # (v1, v2) and (v2, v1) are the same edge.
            key = tuple(sorted((v1.id, v2.id)))

            if key in self.edge_lookup:
                return self.edge_lookup[key]

            edge = Edge(id=len(self.edges),v1=v1,v2=v2)            

            self.edges.append(edge)
            self.edge_lookup[key] = edge

            return edge
        
    def move_vertex(self, vertex_id, displacement):
        """ Move a vertex by the specified displacement.
        
        vertex_id : int, ID of the vertex to move.

        displacement : array-like  [dx, dy]
        """

        vertex = self.vertices[vertex_id]
        #print("vertex id:",vertex_id,"at position:",vertex.position)
        displacement = np.asarray(displacement, dtype=float)

        if displacement.shape == (3,):
            displacement = np.array([displacement[0],displacement[1],displacement[2]])
        else:
            raise ValueError("displacement must have shape (3,) ")                    

        vertex.position += displacement
        
        #Also be sure to update any periodic partner shared vertices too...
        for p1 in vertex.periodic_partners:
            #shared_vertex= self.vertices[p1]
            #shared_vertex.position += displacement
            p1.position += displacement

    def set_preferred_properties(self, A0, P0):
        #We will need to set the cell properties
        A0 = float(A0)
        P0 = float(P0)

        for cell in self.cells:
            cell.A0 = A0
            cell.P0 = P0
            
    def mean_area(self):
        if not self.cells:
            return 0.0

        return sum(cell.area() for cell in self.cells) / len(self.cells)
    
    def mean_perimeter(self):
        if not self.cells: 
            return 0.0
        
        return sum(cell.perimeter() for cell in self.cells)/len(self.cells)