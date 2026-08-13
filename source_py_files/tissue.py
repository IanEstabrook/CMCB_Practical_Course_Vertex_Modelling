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