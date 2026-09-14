
#This function contains 4 implementations: Checks for when T1 transitions should occur (An edge is below a length threshold)
#When a T2 transition should occur (When a cell is below a lower area threshold (cell deletion) or above an upper area threshold (Cell division), and the methods for these two to occur.
from . import plotting
from . import mesh
import copy
from .data_types import Vertex, Cell, Edge
import numpy as np

#To perform a T1 transition, we need to know which if any edges are shorter than a stated threshold
def find_short_edges(tissue, threshold):
    """
    Return edges whose current length is below `threshold`.

    Boundary edges are included in the returned list, but be careful in implementations that the boundaries are 
    """
    short_edges = []

    for edge in tissue.edges:
        
        if edge.edgeLength() < threshold:
            short_edges.append(edge)

    return short_edges

def same_vertex_periodic(v, target):
    #v is a vertex, target is a target vertex which is potentially periodic over a boundary.
    #We will need thisfunction to later check if boundaries over the boundary conditions are periodic.
    if v is target:
        return True

    if v.id == target.id:
        return True

    for p in target.periodic_partners:
        if p is v or p.id == v.id:
            return True

    for p in v.periodic_partners:
        if p is target or p.id == target.id:
            return True

    return False

def vertex_list_with_periodic_equivalent_vertices(vertex):
    #Return vertex plus all of its periodic copies.
    return [vertex] + list(vertex.periodic_partners)

def edge_direction_in_cell(cell, i1, i2):
    n = len(cell.vertices)

    if (i1 + 1) % n == i2:
        return +1       # cell contains v1 -> v2
    elif (i2 + 1) % n == i1:
        return -1       # cell contains v2 -> v1
    else:
        raise ValueError(
            f"Vertices {cell.vertices[i1].id} and "
            f"{cell.vertices[i2].id} are not adjacent in cell {cell.id}"
        )
    
def get_periodic_image_in_cell(cell, target, reference):
    """Given a cell containing vertex reference, identify 
    """

    # Find the representation of reference that is actually in the cell
    reference_index = find_periodic_vertex_index(cell, reference)
    reference_in_cell = cell.vertices[reference_index]

    # Periodic shift between canonical reference and its representation
    shift = reference_in_cell.position - reference.position

    # All possible representations of target
    candidates = [target] + list(target.periodic_partners)

    # Find candidate with the same periodic shift
    distances = [
        np.linalg.norm((candidate.position - target.position) - shift)
        for candidate in candidates
    ]

    best = np.argmin(distances)

    # Tolerance should be comfortably above floating point error
    if distances[best] > 1e-8:
        raise ValueError(
            f"Could not find periodic image of vertex {target.id} "
            f"matching image shift {shift} in cell {cell.id}. "
            f"Distances = {distances}"
        )

    return candidates[best]

def find_periodic_vertex_index(cell, target):
    for i, v in enumerate(cell.vertices):
        if v is target:
            return i

        if v.id == target.id:
            return i

        if any(p is target or p.id == target.id
               for p in v.periodic_partners):
            return i

        if any(p is v or p.id == v.id
               for p in target.periodic_partners):
            return i

    raise ValueError(
        f"Could not find vertex {target.id} "
        f"in cell {cell.id}, including periodic partners."
    )

def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]

#########################Functions to check which cells to keep in removal of a vertex from A , B in T1 transition: We choose which to keep by the option that has no intersecting edges. Copied from gradient_descent

def orientation(a, b, c, tol=1e-12):
    """    Returns the signed cross product (b-a) x (c-a).    """
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def on_segment(a, b, p, tol=1e-12):
    """    True if p lies on the segment a-b, within tolerance.    """
    return (
        min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
        and
        min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
    )


def segments_intersect(a, b, c, d, tol=1e-12):
    """    Check whether closed 2D line segments AB and CD intersect.    """
    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    # Proper intersection
    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)):
        if ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
            return True

    # Collinear / touching cases
    if abs(o1) <= tol and on_segment(a, b, c, tol):
        return True
    if abs(o2) <= tol and on_segment(a, b, d, tol):
        return True
    if abs(o3) <= tol and on_segment(c, d, a, tol):
        return True
    if abs(o4) <= tol and on_segment(c, d, b, tol):
        return True

    return False                



def vertex_removal_creates_intersection(CellA, CellB, v1, v2, tol=1e-12):

    #Once we have the updated positions of the vertices (i.e. AFTER the movement of v1 and v2 while they're still in CellA,CellB), I want to remove one of them from each cell only
    
    #Copy the cells so we can test if removing v1 from the copyCellA and v2 from the copyCellB
    copyCellA=copy.deepcopy(CellA)
    copyCellB=copy.deepcopy(CellB)
    
    i1A = find_periodic_vertex_index(CellA, v1)
    i2A = find_periodic_vertex_index(CellA, v2)

    v1A = CellA.vertices[i1A]
    v2A = CellA.vertices[i2A]
    
    i1B = find_periodic_vertex_index(CellB, v1)
    i2B = find_periodic_vertex_index(CellB, v2)

    #Since we are only testing for now to see if removing v1 from cell a causes an intersection with 
    remove_index_A = i1A
    remove_index_B = i2B


    v1B = CellB.vertices[i1B]
    v2B = CellB.vertices[i2B]


    copyCellA.vertices.pop(remove_index_A)
    copyCellB.vertices.pop(remove_index_B)

    for vertexA, next_vertexA in zip(copyCellA.vertices, copyCellA.vertices[1:] + copyCellA.vertices[:1]):
        edgeA = (vertexA, next_vertexA)
        
        for vertexB, next_vertexB in zip(copyCellB.vertices, copyCellB.vertices[1:] + copyCellB.vertices[:1]):
            edgeB = (vertexB, next_vertexB)
                        
            # Don't test an edge against itself, should only be v1 and v2 cases, however since we removed one of each vertex from copyCellA cand copyCellB, we should never trigger this, nonetheless...
            if((vertexA==vertexB and next_vertexA==next_vertexB)
               or (vertexA==next_vertexB and next_vertexA==vertexB)):
                
                continue            
            if segments_intersect(vertexA.position[:2], next_vertexA.position[:2],vertexB.position[:2], next_vertexB.position[:2],tol=tol):
                
                #Then the removal of v1 from CellA and v2 from CellB causes an intersection, so the correct removal is the other way around.
                CellA.vertices.pop(i2A)
                CellB.vertices.pop(i1B)
                print("\n\n (1)TestRemoval of v1 from A would cause intersection, removing v2")
                print("For debugging, positions were:\n",vertexA.position[:2], next_vertexA.position[:2], vertexB.position[:2], next_vertexB.position[:2])
                removeA = v2A
                removeB = v1B
                return removeA,removeB
    #If we reached the end of the loop here, no intersection was created and we can safely remove 
    CellA.vertices.pop(i1A)
    CellB.vertices.pop(i2B)
    removeA = v1A
    removeB=v2B
    print("\n\n (1)TestRemoval of v2 from A would cause intersection, removing v1")
    return removeA,removeB

             
    '''
    if(vertex.periodic_partners):
        for p in vertex.periodic_partners:
            #print(p.id," is a periodic partner of ",vertex.id)
            incident_p_edges = p.edges
            candidate_p_edges = get_candidate_edges_for_vertex(p)
            for p_edge in incident_p_edges:


                if p_edge.v1.id == p.id:
                    p1 = p_edge.v1.position[:2]+displacement[:2]
                    p2 = p_edge.v2.position[:2]
                else:
                    p1 = p_edge.v1.position[:2]
                    p2 = p_edge.v2.position[:2]+displacement[:2]

                for other_p_edge in candidate_p_edges:

                    # Don't test an edge against itself
                    if other_p_edge.id == p_edge.id:
                        continue

                    # Edges which already share a vertex are allowed
                    # to meet at that vertex.
                    if edges_share_vertex(p_edge, other_p_edge):
                        continue

                    q1 = other_p_edge.v1.position[:2]
                    q2 = other_p_edge.v2.position[:2]

                    if segments_intersect(p1, p2,q1, q2,tol=tol):
    #                                print("Edges",p_edge.id,"and ",other_p_edge.id,"intersect? check")

                        return True
            #if(vertex_move_creates_intersection(p, displacement, tol=1e-12, periodic_recurse=False)):
            #    return True
    '''
    #print("--- %s seconds to check if one move intersects another candidate edges ---" % (time.time() - start_time))    
    return False


def vertex_addition_creates_intersection(CellC, CellD, CellA, CellB, vertexAddToC, vertexAddToD, tol=1e-12):

    #This function determines whether adding vertex_to_add to cellC and CellD causes an intersection with any of CellA,CellB,CellC,CellD
    
    #Copy the cells so we can test if removing v1 from the copyCellA and v2 from the copyCellB
    copyCellC=copy.deepcopy(CellC)
    copyCellD=copy.deepcopy(CellD)
    
    #The vertex to add to D already exists on C
    iC = find_periodic_vertex_index(copyCellC, vertexAddToD)
    #i2C = find_periodic_vertex_index(CellC, vertex_to_add_D)

    vC = CellC.vertices[iC]
    
    
    #The vertex to add to C already exists on D
    iD = find_periodic_vertex_index(copyCellD, vertexAddToC)
    vD = CellD.vertices[iD]

    #We need to test whether the new vertex to each of cell C, D should be added before or after the existing vertex.
    # Extremely brute force approach, but we try adding the vertex before and after until we find a combination which doesn't cause an intersection between cell C and D and A and B...
    #We try this by adding first the vertex before to copyCellC 
    
    #################################
    #
    #
    # possibility 1: addition at iC,iD
    #
    #################################
    copyCellC.vertices.insert((iC) %len(copyCellC.vertices) , vertexAddToC)
    copyCellD.vertices.insert((iD) %len(copyCellD.vertices) , vertexAddToD)
            
    correctPosCFound=True
    correctPosDFound=True
    #Loop over C and all 3 other cells
    for vertexC, next_vertexC in zip(copyCellC.vertices, copyCellC.vertices[1:] + copyCellC.vertices[:1]):
        edgeC = (vertexC, next_vertexC)                
        
        for vertexA, next_vertexA in zip(CellA.vertices, CellA.vertices[1:] + CellA.vertices[:1]):
            edgeA = (vertexA, next_vertexA)
            
            if(vertexA.id==vertexC.id or vertexA.id == next_vertexC.id or next_vertexA.id==vertexC.id or next_vertexA.id == next_vertexC.id):
                print(
                        "Case 1a: SKIPPING:",
                        f"C edge {vertexC.id}->{next_vertexC.id}",
                        f"A edge {vertexA.id}->{next_vertexA.id}"
                    )
            #if((vertexA.id==vertexC.id and vertexA.id == next_vertexC.id) or (next_vertexA.id==vertexC.id and next_vertexA.id == next_vertexC.id)):
                continue
                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexA.position[:2], next_vertexA.position[:2],tol=tol):                
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellA, so this is not the correct position to add. 
                print("Case 1A intersect:", vertexC.position[:2], next_vertexC.position[:2],vertexA.position[:2], next_vertexA.position[:2])
                correctPosCFound=False
            
        for vertexB, next_vertexB in zip(CellB.vertices, CellB.vertices[1:] + CellB.vertices[:1]):
            edgeB = (vertexB, next_vertexB)
            
            if(vertexB.id==vertexC.id or vertexB.id == next_vertexC.id or next_vertexB.id==vertexC.id or next_vertexB.id == next_vertexC.id):
            #if((vertexB.id==vertexC.id and vertexB.id == next_vertexC.id) or (next_vertexB.id==vertexC.id and next_vertexB.id == next_vertexC.id)):
                print(
                        "Case 1b: SKIPPING:",
                        f"C edge {vertexC.id}->{next_vertexC.id}",
                        f"B edge {vertexB.id}->{next_vertexB.id}"
                    )
                continue
                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexB.position[:2], next_vertexB.position[:2],tol=tol):
                print("Case 3B intersect:", vertexC.position[:2], next_vertexC.position[:2],vertexB.position[:2], next_vertexB.position[:2])
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellB, so this is not the correct position to add.                              
                correctPosCFound=False
            
        for vertexD, next_vertexD in zip(copyCellD.vertices, copyCellD.vertices[1:] + copyCellD.vertices[:1]):
            edgeD = (vertexD, next_vertexD)
            if(vertexD.id==vertexC.id or vertexD.id == next_vertexC.id or next_vertexD.id==vertexC.id or next_vertexD.id == next_vertexC.id):
            #if((vertexD.id==vertexC.id and vertexD.id == next_vertexC.id) or (next_vertexD.id==vertexC.id and next_vertexD.id == next_vertexC.id)):
                #This edge is shared and so always intersects
                print(
                        "Case 1d: SKIPPING:",
                        f"C edge {vertexC.id}->{next_vertexC.id}",
                        f"D edge {vertexD.id}->{next_vertexD.id}"
                    )
                continue

                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexD.position[:2], next_vertexD.position[:2],tol=tol):
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellD, so this is not the correct position to add.to cell C         
                correctPosCFound=False
                correctPosDFound=False
            
    
    #Loop over D and all other cells
    for vertexD, next_vertexD in zip(copyCellD.vertices, copyCellD.vertices[1:] + copyCellD.vertices[:1]):
        edgeD = (vertexD, next_vertexD)
        for vertexA, next_vertexA in zip(CellA.vertices, CellA.vertices[1:] + CellA.vertices[:1]):
            edgeA = (vertexA, next_vertexA)
            
            if(vertexA.id==vertexD.id or vertexA.id == next_vertexD.id or next_vertexA.id==vertexD.id or next_vertexA.id == next_vertexD.id):
            #if((vertexA.id==vertexD.id and vertexA.id == next_vertexD.id) or ( next_vertexA.id==vertexD.id and next_vertexA.id == next_vertexD.id)):
                continue
            
            if segments_intersect(vertexD.position[:2], next_vertexD.position[:2],vertexA.position[:2], next_vertexA.position[:2],tol=tol):
                
                #Then the addition of vertexAddToC to D before ic causes an intersection with CellA, so this is not the correct position to add.
                
                correctPosDFound=False
            
        for vertexB, next_vertexB in zip(CellB.vertices, CellB.vertices[1:] + CellB.vertices[:1]):
            edgeB = (vertexB, next_vertexB)
            if(vertexB.id==vertexD.id or vertexB.id == next_vertexD.id or next_vertexB.id==vertexD.id or next_vertexB.id == next_vertexD.id):
            #if((vertexB.id==vertexD.id and vertexB.id == next_vertexD.id) or (next_vertexB.id==vertexD.id and next_vertexB.id == next_vertexD.id)):
                continue
            if segments_intersect(vertexD.position[:2], next_vertexD.position[:2],vertexB.position[:2], next_vertexB.position[:2],tol=tol):
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellB, so this is not the correct position to add.
                
                correctPosDFound=False
          
        
    if(correctPosCFound and correctPosDFound):   
        print("Step 1 success Adding vertices at", iC, iD)
        return iC, iD
    else:
        copyCellC.vertices.pop(iC)   
        copyCellD.vertices.pop(iD)                                

    #################################
    #
    #
    # possibility 2: addition at iC+1,iD
    #
    #################################
    copyCellC.vertices.insert((iC+1) %len(copyCellC.vertices) , vertexAddToC)
    copyCellD.vertices.insert((iD) %len(copyCellD.vertices) , vertexAddToD)
            
    correctPosCFound=True
    correctPosDFound=True
    #Loop over C and all 3 other cells
    for vertexC, next_vertexC in zip(copyCellC.vertices, copyCellC.vertices[1:] + copyCellC.vertices[:1]):
        edgeC = (vertexC, next_vertexC)                
        
        for vertexA, next_vertexA in zip(CellA.vertices, CellA.vertices[1:] + CellA.vertices[:1]):
            edgeA = (vertexA, next_vertexA)
            
            if(vertexA.id==vertexC.id or vertexA.id == next_vertexC.id or next_vertexA.id==vertexC.id or next_vertexA.id == next_vertexC.id):
            #if((vertexA.id==vertexC.id and vertexA.id == next_vertexC.id) or (next_vertexA.id==vertexC.id and next_vertexA.id == next_vertexC.id)):
                continue
                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexA.position[:2], next_vertexA.position[:2],tol=tol):                
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellA, so this is not the correct position to add. 
                print("Case 3A intersect:", vertexC.position[:2], next_vertexC.position[:2],vertexA.position[:2], next_vertexA.position[:2])
                correctPosCFound=False
            
        for vertexB, next_vertexB in zip(CellB.vertices, CellB.vertices[1:] + CellB.vertices[:1]):
            edgeB = (vertexB, next_vertexB)
            
            if(vertexB.id==vertexC.id or vertexB.id == next_vertexC.id or next_vertexB.id==vertexC.id or next_vertexB.id == next_vertexC.id):
            #if((vertexB.id==vertexC.id and vertexB.id == next_vertexC.id) or (next_vertexB.id==vertexC.id and next_vertexB.id == next_vertexC.id)):
                continue
                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexB.position[:2], next_vertexB.position[:2],tol=tol):
                print("Case 3B intersect:", vertexC.position[:2], next_vertexC.position[:2],vertexB.position[:2], next_vertexB.position[:2])
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellB, so this is not the correct position to add.                              
                correctPosCFound=False
            
        for vertexD, next_vertexD in zip(copyCellD.vertices, copyCellD.vertices[1:] + copyCellD.vertices[:1]):
            edgeD = (vertexD, next_vertexD)
            if(vertexD.id==vertexC.id or vertexD.id == next_vertexC.id or next_vertexD.id==vertexC.id or next_vertexD.id == next_vertexC.id):
            #if((vertexD.id==vertexC.id and vertexD.id == next_vertexC.id) or (next_vertexD.id==vertexC.id and next_vertexD.id == next_vertexC.id)):
                #This edge is shared and so always intersects
                continue

                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexD.position[:2], next_vertexD.position[:2],tol=tol):
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellD, so this is not the correct position to add.to cell C         
                correctPosCFound=False
                correctPosDFound=False
            
    
    #Loop over D and all other cells
    for vertexD, next_vertexD in zip(copyCellD.vertices, copyCellD.vertices[1:] + copyCellD.vertices[:1]):
        edgeD = (vertexD, next_vertexD)
        for vertexA, next_vertexA in zip(CellA.vertices, CellA.vertices[1:] + CellA.vertices[:1]):
            edgeA = (vertexA, next_vertexA)
            
            if(vertexA.id==vertexD.id or vertexA.id == next_vertexD.id or next_vertexA.id==vertexD.id or next_vertexA.id == next_vertexD.id):
            #if((vertexA.id==vertexD.id and vertexA.id == next_vertexD.id) or ( next_vertexA.id==vertexD.id and next_vertexA.id == next_vertexD.id)):
                continue
            
            if segments_intersect(vertexD.position[:2], next_vertexD.position[:2],vertexA.position[:2], next_vertexA.position[:2],tol=tol):
                
                #Then the addition of vertexAddToC to D before ic causes an intersection with CellA, so this is not the correct position to add.
                
                correctPosDFound=False
            
        for vertexB, next_vertexB in zip(CellB.vertices, CellB.vertices[1:] + CellB.vertices[:1]):
            edgeB = (vertexB, next_vertexB)
            if(vertexB.id==vertexD.id or vertexB.id == next_vertexD.id or next_vertexB.id==vertexD.id or next_vertexB.id == next_vertexD.id):
            #if((vertexB.id==vertexD.id and vertexB.id == next_vertexD.id) or (next_vertexB.id==vertexD.id and next_vertexB.id == next_vertexD.id)):
                continue
            if segments_intersect(vertexD.position[:2], next_vertexD.position[:2],vertexB.position[:2], next_vertexB.position[:2],tol=tol):
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellB, so this is not the correct position to add.
                
                correctPosDFound=False
          
            
    if(correctPosCFound and correctPosDFound):        
        print("Step 2 success Adding vertices at", iC+1, iD)
        return iC+1, iD
    else:
        copyCellC.vertices.pop(iC+1)   
        copyCellD.vertices.pop(iD)                                
    
    #################################
    #
    #
    # possibility 3: addition at iC,iD+1
    #
    #################################
    copyCellC.vertices.insert((iC) %len(copyCellC.vertices) , vertexAddToC)
    copyCellD.vertices.insert((iD+1) %len(copyCellD.vertices) , vertexAddToD)
            
    correctPosCFound=True
    correctPosDFound=True
    #Loop over C and all 3 other cells
    for vertexC, next_vertexC in zip(copyCellC.vertices, copyCellC.vertices[1:] + copyCellC.vertices[:1]):
        edgeC = (vertexC, next_vertexC)                
        
        for vertexA, next_vertexA in zip(CellA.vertices, CellA.vertices[1:] + CellA.vertices[:1]):
            edgeA = (vertexA, next_vertexA)
            
            if(vertexA.id==vertexC.id or vertexA.id == next_vertexC.id or next_vertexA.id==vertexC.id or next_vertexA.id == next_vertexC.id):
            #if((vertexA.id==vertexC.id and vertexA.id == next_vertexC.id) or (next_vertexA.id==vertexC.id and next_vertexA.id == next_vertexC.id)):
                continue
                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexA.position[:2], next_vertexA.position[:2],tol=tol):                
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellA, so this is not the correct position to add. 
                print("Case 3A intersect:", vertexC.position[:2], next_vertexC.position[:2],vertexA.position[:2], next_vertexA.position[:2])
                correctPosCFound=False
            
        for vertexB, next_vertexB in zip(CellB.vertices, CellB.vertices[1:] + CellB.vertices[:1]):
            edgeB = (vertexB, next_vertexB)
            
            if(vertexB.id==vertexC.id or vertexB.id == next_vertexC.id or next_vertexB.id==vertexC.id or next_vertexB.id == next_vertexC.id):
            #if((vertexB.id==vertexC.id and vertexB.id == next_vertexC.id) or (next_vertexB.id==vertexC.id and next_vertexB.id == next_vertexC.id)):
                continue
                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexB.position[:2], next_vertexB.position[:2],tol=tol):
                print("Case 3B intersect:", vertexC.position[:2], next_vertexC.position[:2],vertexB.position[:2], next_vertexB.position[:2])
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellB, so this is not the correct position to add.                              
                correctPosCFound=False
            
        for vertexD, next_vertexD in zip(copyCellD.vertices, copyCellD.vertices[1:] + copyCellD.vertices[:1]):
            edgeD = (vertexD, next_vertexD)
            if(vertexD.id==vertexC.id or vertexD.id == next_vertexC.id or next_vertexD.id==vertexC.id or next_vertexD.id == next_vertexC.id):
            #if((vertexD.id==vertexC.id and vertexD.id == next_vertexC.id) or (next_vertexD.id==vertexC.id and next_vertexD.id == next_vertexC.id)):
                #This edge is shared and so always intersects
                continue

                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexD.position[:2], next_vertexD.position[:2],tol=tol):
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellD, so this is not the correct position to add.to cell C         
                correctPosCFound=False
                correctPosDFound=False
            
    
    #Loop over D and all other cells
    for vertexD, next_vertexD in zip(copyCellD.vertices, copyCellD.vertices[1:] + copyCellD.vertices[:1]):
        edgeD = (vertexD, next_vertexD)
        for vertexA, next_vertexA in zip(CellA.vertices, CellA.vertices[1:] + CellA.vertices[:1]):
            edgeA = (vertexA, next_vertexA)
            
            if(vertexA.id==vertexD.id or vertexA.id == next_vertexD.id or next_vertexA.id==vertexD.id or next_vertexA.id == next_vertexD.id):
            #if((vertexA.id==vertexD.id and vertexA.id == next_vertexD.id) or ( next_vertexA.id==vertexD.id and next_vertexA.id == next_vertexD.id)):
                continue
            
            if segments_intersect(vertexD.position[:2], next_vertexD.position[:2],vertexA.position[:2], next_vertexA.position[:2],tol=tol):
                
                #Then the addition of vertexAddToC to D before ic causes an intersection with CellA, so this is not the correct position to add.
                
                correctPosDFound=False
            
        for vertexB, next_vertexB in zip(CellB.vertices, CellB.vertices[1:] + CellB.vertices[:1]):
            edgeB = (vertexB, next_vertexB)
            if(vertexB.id==vertexD.id or vertexB.id == next_vertexD.id or next_vertexB.id==vertexD.id or next_vertexB.id == next_vertexD.id):
            #if((vertexB.id==vertexD.id and vertexB.id == next_vertexD.id) or (next_vertexB.id==vertexD.id and next_vertexB.id == next_vertexD.id)):
                continue
            if segments_intersect(vertexD.position[:2], next_vertexD.position[:2],vertexB.position[:2], next_vertexB.position[:2],tol=tol):
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellB, so this is not the correct position to add.
                
                correctPosDFound=False
          
          
    if(correctPosCFound and correctPosDFound):        
        print("Step 3 success Adding vertices at", iC, iD+1)
        return iC, iD+1
    else:
        copyCellC.vertices.pop(iC)   
        copyCellD.vertices.pop(iD+1)  
        
    #################################
    #
    #
    # possibility 4: addition at iC+1,iD+1
    #
    #################################
    copyCellC.vertices.insert((iC+1) %len(copyCellC.vertices) , vertexAddToC)
    copyCellD.vertices.insert((iD+1) %len(copyCellD.vertices) , vertexAddToD)
            
    correctPosCFound=True
    correctPosDFound=True
    #Loop over C and all 3 other cells
    for vertexC, next_vertexC in zip(copyCellC.vertices, copyCellC.vertices[1:] + copyCellC.vertices[:1]):
        edgeC = (vertexC, next_vertexC)                
        
        for vertexA, next_vertexA in zip(CellA.vertices, CellA.vertices[1:] + CellA.vertices[:1]):
            edgeA = (vertexA, next_vertexA)
            
            if(vertexA.id==vertexC.id or vertexA.id == next_vertexC.id or next_vertexA.id==vertexC.id or next_vertexA.id == next_vertexC.id):
            #if((vertexA.id==vertexC.id and vertexA.id == next_vertexC.id) or (next_vertexA.id==vertexC.id and next_vertexA.id == next_vertexC.id)):
                continue
                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexA.position[:2], next_vertexA.position[:2],tol=tol):                
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellA, so this is not the correct position to add. 
                print("Case 3A intersect:", vertexC.position[:2], next_vertexC.position[:2],vertexA.position[:2], next_vertexA.position[:2])
                correctPosCFound=False
            
        for vertexB, next_vertexB in zip(CellB.vertices, CellB.vertices[1:] + CellB.vertices[:1]):
            edgeB = (vertexB, next_vertexB)
            
            if(vertexB.id==vertexC.id or vertexB.id == next_vertexC.id or next_vertexB.id==vertexC.id or next_vertexB.id == next_vertexC.id):
            #if((vertexB.id==vertexC.id and vertexB.id == next_vertexC.id) or (next_vertexB.id==vertexC.id and next_vertexB.id == next_vertexC.id)):
                continue
                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexB.position[:2], next_vertexB.position[:2],tol=tol):
                print("Case 3B intersect:", vertexC.position[:2], next_vertexC.position[:2],vertexB.position[:2], next_vertexB.position[:2])
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellB, so this is not the correct position to add.                              
                correctPosCFound=False
            
        for vertexD, next_vertexD in zip(copyCellD.vertices, copyCellD.vertices[1:] + copyCellD.vertices[:1]):
            edgeD = (vertexD, next_vertexD)
            if(vertexD.id==vertexC.id or vertexD.id == next_vertexC.id or next_vertexD.id==vertexC.id or next_vertexD.id == next_vertexC.id):
            #if((vertexD.id==vertexC.id and vertexD.id == next_vertexC.id) or (next_vertexD.id==vertexC.id and next_vertexD.id == next_vertexC.id)):
                #This edge is shared and so always intersects
                continue

                
            if segments_intersect(vertexC.position[:2], next_vertexC.position[:2],vertexD.position[:2], next_vertexD.position[:2],tol=tol):
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellD, so this is not the correct position to add.to cell C         
                correctPosCFound=False
                correctPosDFound=False
            
    
    #Loop over D and all other cells
    for vertexD, next_vertexD in zip(copyCellD.vertices, copyCellD.vertices[1:] + copyCellD.vertices[:1]):
        edgeD = (vertexD, next_vertexD)
        for vertexA, next_vertexA in zip(CellA.vertices, CellA.vertices[1:] + CellA.vertices[:1]):
            edgeA = (vertexA, next_vertexA)
            
            if(vertexA.id==vertexD.id or vertexA.id == next_vertexD.id or next_vertexA.id==vertexD.id or next_vertexA.id == next_vertexD.id):
            #if((vertexA.id==vertexD.id and vertexA.id == next_vertexD.id) or ( next_vertexA.id==vertexD.id and next_vertexA.id == next_vertexD.id)):
                continue
            
            if segments_intersect(vertexD.position[:2], next_vertexD.position[:2],vertexA.position[:2], next_vertexA.position[:2],tol=tol):
                
                #Then the addition of vertexAddToC to D before ic causes an intersection with CellA, so this is not the correct position to add.
                
                correctPosDFound=False
            
        for vertexB, next_vertexB in zip(CellB.vertices, CellB.vertices[1:] + CellB.vertices[:1]):
            edgeB = (vertexB, next_vertexB)
            if(vertexB.id==vertexD.id or vertexB.id == next_vertexD.id or next_vertexB.id==vertexD.id or next_vertexB.id == next_vertexD.id):
            #if((vertexB.id==vertexD.id and vertexB.id == next_vertexD.id) or (next_vertexB.id==vertexD.id and next_vertexB.id == next_vertexD.id)):
                continue
            if segments_intersect(vertexD.position[:2], next_vertexD.position[:2],vertexB.position[:2], next_vertexB.position[:2],tol=tol):
                #Then the addition of vertexAddToC to C before ic causes an intersection with CellB, so this is not the correct position to add.
                
                correctPosDFound=False
          
          
    if(correctPosCFound and correctPosDFound):        
        print("Step 4 success Adding vertices at", iC+1, iD+1)
        return iC+1, iD+1
    else:
        copyCellC.vertices.pop(iC+1)   
        copyCellD.vertices.pop(iD+1)          
    
    print("Warning, unable to identify where to add cell vertices to C and D...")
    return


def closest_periodic_vertex(vertex, cell):
    """
    Return the periodic copy of `vertex` that is closest to `cell`'s centroid.

    Traverses all recursively connected periodic partners, so corner vertices
    with periodicity in both x and y are handled correctly.
    """

    centrePos = cell.centroid()

    closest_vertex = vertex
    shortest_length = np.linalg.norm(vertex.position[:2] - centrePos[:2])
    
    # track the vertices already examined to prevent endless loops....
    processed_vertices=set()

    # Start traversal from the supplied vertex
    vertices_to_check = [vertex]
    visited = set()

    while vertices_to_check:
        current = vertices_to_check.pop()

        if current.id in visited:
            continue

        visited.add(current.id)

        # Check current vertex
        curr_length = np.linalg.norm(current.position[:2] - centrePos[:2])

        if curr_length < shortest_length:
            closest_vertex = current
            shortest_length = curr_length
            

        for partner in current.periodic_partners:
            if partner.id not in visited:
                vertices_to_check.append(partner)
                
    return closest_vertex

def remove_unconnected_vertices(vertices):
    return [v for v in vertices if v.edges]

def perform_t1_transitions(tissue,t1_threshold=0.01,new_edge_length=None,max_transitions=None):
    
    num_t1_transitions=0;
    testMessages=True;
    #Get the short edges.
    short_edges=find_short_edges(tissue, t1_threshold);
    if len(short_edges) == 0:
            return 0
    else :
        
        processed_edges = set()

        
        # ------------------------------------------------------------------
        # Find the vertices opposite the edge. i.e. in ASCII art.... with x = v1, y = v2
        #
        #        \         /
        #        \ cell a/
        #cell c   x-----y cell d
        #        /cell b\
        #       /        \
        # 
        # the T1 transition needs to know which cells c and d are 
        # The opposite vertex is the vertex following v2/v1.
        # ------------------------------------------------------------------

        #Steps to a T1 transition
        #vertices x and y merge in to one: if a and b were hexagons, cell a and cell b become pentagons (i.e. cells with 5 vertices)
        #cell c and d now  shared this merged vertex. 
        #          cell a
        #           \  /
        #            \/
        #     cell c X cell d
        #           /\
        #          /  \
        #           cell b

        #Then this merged vertex is divided in to a new edge with a horizontal (in this diagram) split.
        #
        #          cell a
        #           \  /
        #            \/
        #            x'
        #            |   
        #  cell c    |    cell d
        #           y' 
        #           /\
        #          /  \
        #           cell b

        #We can skip the middle step here, so what we need to do is : Remove vertex x & y from cell a, b, c, d 
        #Add vertex x' to cell 1, y' to cell b and both to cell c, d, in appropriate orders. We also have to define the positions of x' and y'

        #First we need to identify which cells cell c and d are, fortunately we have this information in vertex.cells
        for edge in short_edges:
            
            if testMessages:
                print(f"Short edge found, id",edge.id) 
                print(f"Edge joins vertices",edge.v1.id,edge.v2.id)
                
            if edge not in tissue.edges:
                # Re-check that the edge still exists. A previous T1 may
                # have modified the topology.
                continue
            #Get the cells involved in the transition, this includes over the boundaries if periodic.
            adjacent_cells = edge.cells

            # Boundary edges in a non periodic configuration cannot undergo a T1 transition.
            if len(adjacent_cells) != 2:
                print(f"Warning: Edge is adjacent to", len(adjacent_cells),"cells, skipping")
                continue
            
            #Track and update the edge id_s in such a way that for periodic boundaries, no update is performed twice in one step.
            v1 = edge.v1
            v2 = edge.v2
            edge_ids = {edge.id}
            edge_ids.update(p.id for p in edge.periodic_partners)

            if edge_ids & processed_edges:
                if testMessages:
                    print("However, it is already processed, skipping")
                continue
                
            #We will need the original positions later to determine the orientation, save them too            
            original_v1=v1.position.copy()
            original_v2=v2.position.copy()
                        
            #The first thing we need to do is identify the four cells involved in the T1 transition, we will label them A,B,C,D
            #A and B are the cells which are already connected by the short edge.                        
            cellA = adjacent_cells[0]
            cellB = adjacent_cells[1]
            
            #Cells C and D are the cells that are around the existing two vertices, but NOT cell A or B.
            #Identify these cells from the list of cells that a vertex is a part of.
            cellList1=[];
            if v1.cells:
                for p in v1.cells:
                    cellList1.append(p)
                for p in v1.periodic_partners:
                    for j in p.cells:
                        cellList1.append(j)
            cellList2=[];
            if v2.cells:
                for p in v2.cells:
                    cellList2.append(p)
                for p in v2.periodic_partners:
                    for j in p.cells:
                        cellList2.append(j)
            # Elements in cellList1/cellList2 that are NOT adjacent will give us the other two cells, C and D. 
            
            cellC_candidates = [x for x in cellList1 if x not in adjacent_cells]
            cellD_candidates = [x for x in cellList2 if x not in adjacent_cells]

            if len(cellC_candidates) != 1 or len(cellD_candidates) != 1:
                continue

            cellC = cellC_candidates[0]
            cellD = cellD_candidates[0]
            
            if(testMessages):
                print(f"Cell A: ",cellA.id,f"Cell B: ",cellB.id,f"Cell C: ",cellC.id,f"Cell D: ",cellD.id)
            
            #Rather than delete vertices/edges, we are going to change their connectivity - which cells they are part of, and the vertex positions. 
            
            #First we move the existing vertex v1 and v2 to the midpoint of x and y and then we move them perpendicularly to the initial v1-v2 direction a distance new_edge_length. 
            
            #First define the positions of the v1 and v2 in the xy plane:
            v1X=v1.position[0]
            v1Y=v1.position[1]
            v2X=v2.position[0]
            v2Y=v2.position[1]
            #Get the distance from these poisitions to the midpoint from v1:
            v1v2MidpointDeltaX=(v1X+v2X)/2.0 - v1X
            v1v2MidpointDeltaY=(v1Y+v2Y)/2.0 - v1Y
            
            #And from v2:
            v2v1MidpointDeltaX=(v1X+v2X)/2.0 - v2X
            v2v1MidpointDeltaY=(v1Y+v2Y)/2.0 - v2Y
            
            #We also  need to know the orientation of the v1-v2 line. We define a vector from v1 to v2:
            edge_vector = v2.position[:2] - v1.position[:2]
            
            #And we need to know a point in space nearby that isn't on the line to define the orientation. Entirely arbitrary, but we take the next point on cell A in increasing index order for simplicity.
            test_vertex = next(v for v in cellA.vertices if v.id != v1.id and v.id != v2.id)                        
            cell_vector = test_vertex.position[:2] - v1.position[:2]

            #The cross product of these two then gives us the orientation of the direction from v1 to v2, which changes the orientation angle of the surface.            
            side = cross2d(edge_vector, cell_vector)
            initialDirTheta=np.arctan2(v2Y-v1Y,v2X-v1X)
            if(testMessages):
                print(f"Side value =",side)
                print(f"initial theta dir=",initialDirTheta)
            
            #The perpendicular direction to the line is either + or - 90 degrees (pi in radians)
            if side > 0:
                # Cell A is on the left of v1 -> v2
                perpTheta = initialDirTheta + np.pi/2.0
                #This also defines which vertex to remove from cell A and cell B... If side>0 , remove v1 from A and v2 from B, otherwise vice versa.
            else:
                # Cell A is on the right of v1 -> v2
                perpTheta = initialDirTheta - np.pi/2.0
            
            if(testMessages):
                print(f"Perpendicular theta dir=",perpTheta)
            
            v1v2MidpointDeltaX=v1v2MidpointDeltaX+new_edge_length*np.cos(perpTheta)/2.0
            v1v2MidpointDeltaY=v1v2MidpointDeltaY+new_edge_length*np.sin(perpTheta)/2.0
            
            v2v1MidpointDeltaX=v2v1MidpointDeltaX+new_edge_length*np.cos(perpTheta+np.pi)/2.0
            v2v1MidpointDeltaY=v2v1MidpointDeltaY+new_edge_length*np.sin(perpTheta+np.pi)/2.0
            
            #Now that we know the distances to move the vertices,we use the existing function to move the points. 
            if testMessages:
                print(f"Moving vertices, id",v1.id,v2.id, f"from v1=(",v1.position[0],v1.position[1],f"),v2=(",v2.position[0],v2.position[1],")")
                
            tissue.move_vertex(v1.id,[v1v2MidpointDeltaX,v1v2MidpointDeltaY,0.0])
            tissue.move_vertex(v2.id,[v2v1MidpointDeltaX,v2v1MidpointDeltaY,0.0])
            #This move function also handles periodicity so all periodic points will be moved alongside
            if testMessages:
                print(f"to v1=(",v1.position[0],v1.position[1],f"),v2=(",v2.position[0],v2.position[1],")")
                print(f"new edge length=(",np.sqrt((v1.position[0]-v2.position[0])**2+(v1.position[1]-v2.position[1])**2))
            #We need to get the vertices before and after the current one on cell C and cell D, so that we know where to insert the new vertices afterwards.
            i = next(i for i, v in enumerate(cellC.vertices) if same_vertex_periodic(v, v1))
            beforeMoveC = cellC.vertices[(i - 1) % len(cellC.vertices)]
            currC = cellC.vertices[(i ) % len(cellC.vertices)]
            afterMoveC  = cellC.vertices[(i + 1) % len(cellC.vertices)]
                        
            i = next(i for i, v in enumerate(cellD.vertices) if same_vertex_periodic(v, v2))
            beforeMoveD = cellD.vertices[(i - 1) % len(cellD.vertices)]
            currD= cellD.vertices[(i ) % len(cellD.vertices)]
            afterMoveD  = cellD.vertices[(i + 1) % len(cellD.vertices)]
            
            #Define the orientations now, only using the positions of the vertices that were originally on cell C and D. This is important otherwise errors can be introduced through periodicity
            
            # = cross2d(original_v1[:2] - beforeMoveC.position[:2],afterMoveC.position[:2] - original_v1[:2])
            #orientationD = cross2d(original_v2[:2] - beforeMoveD.position[:2],afterMoveD.position[:2] - original_v2[:2])
            
            orientationC = cross2d(currC.position[:2] - beforeMoveC.position[:2],afterMoveC.position[:2] - currC.position[:2])
            orientationD = cross2d(currD.position[:2] - beforeMoveD.position[:2],afterMoveD.position[:2] - currD.position[:2])
                        
            new_edge = v2.position[:2] - v1.position[:2]
            #old_edge = np.array([v2X-v1X, v2Y-v1Y])
            old_edge = np.array([original_v2[0]-original_v1[0], original_v2[1]-original_v1[1]])
            
            if(testMessages):
                print("old edge:", old_edge)
                print("new edge:", new_edge)
                print("dot:", np.dot(old_edge, new_edge))
            
            
            #We now know the vertex positions after the transition, as well as the oreintations of each cell.
            #The final step is to update our tissue and associated objects. 
           
            #For Cell A and Cell B, we need to remove a single vertex index from each, depending on the orientation
            # ------------------------------------------------------------
            # Cell A
            # ------------------------------------------------------------

            i1A = find_periodic_vertex_index(cellA, v1)
            i2A = find_periodic_vertex_index(cellA, v2)

            # IMPORTANT: these must come from cellA
            v1A = cellA.vertices[i1A]
            v2A = cellA.vertices[i2A]
            

            # ------------------------------------------------------------
            # Cell B
            # ------------------------------------------------------------

            i1B = find_periodic_vertex_index(cellB, v1)
            i2B = find_periodic_vertex_index(cellB, v2)

            v1B = cellB.vertices[i1B]
            v2B = cellB.vertices[i2B]
                                
            removeA,removeB=vertex_removal_creates_intersection(cellA, cellB, v1, v2, tol=1e-12)
                
            print("Removed from Cell A vertex id=",removeA.id, " Removed from Cell B vertex id=",removeB.id)
            #if(testMessages):
            #    print("For Cell A, ", i1A, i2A,"Removing entry",remove_index_A,"vertex frrom Cell A id :",cellA.vertices[remove_index_A].id)
                
                
            #print("directionA=",directionA,"directionB=",directionB)
            
            # ------------------------------------------------------------
            # Cell C
            # C originally contains v1, or periodic equivalent so locate v1
            # However, on boundaries, we need to be very careful which version of v2 we add- we may  need to generate a new vertex, which we can check if adding the closest existing periodic version of v2 will cause a cell to be malformed across the tissue, crossing the entire surface. To do this, we check that the cell distance frrom v1 to v2 is not significantly larger than new_edge_length 
            # ------------------------------------------------------------
            
            #We already got the orientation of C and D before in orientationC, orientationD, which describes the orientation BEFORE the addition of the vertex. What we now need to do is ensure that the addition of the new vertex is consistent, so that the direction before -> v1 -> v2 -> after or before -> v2 -> v1 -> after is consistent in direction. In this case, we want to determine which of the directions is consistent. I.e. one of the sign of before -> v1 -> v2  or after -> v1 -> v2 should be consistent with the earlier measured oreintationC and orientationD values (same sign).
            
            #Make sure we use the correct cell over the boundary, in case periodicity causes shenanigans.
            #v1C = get_periodic_image_in_cell(cellC, v1, v1)
            #v2C = get_periodic_image_in_cell(cellC, v2, v1)
            
            vertexAddToC=closest_periodic_vertex(v2, cellC)
            i1C=find_periodic_vertex_index(cellC, v1)
            
            
            v1C=cellC.vertices[i1C].position
            #Get the equivalent of the new position periodically if needed.
            v2C=vertexAddToC.position
            
            if(np.linalg.norm(v1C - v2C)>new_edge_length+0.1):
                #need to create a new vertex for v2C periodically because the distance is too large.
                if(testMessages):
                    print("Warning: distance between v1 and v2 on cell C is too large")
                if(testMessages):
                    
                    
                    print("Delta x between vertexAddToC and v2 =(",v2.position[0]-vertexAddToC.position[0],",",v2.position[1]-vertexAddToC.position[1]);
                deltaX=v2.position[0]-vertexAddToC.position[0]
                deltaY=v2.position[1]-vertexAddToC.position[1]
                
                if(deltaX<0.001): #The shift is along a boundary from the left side of the surface to the right, we want to know the largest degree of the shift
                    XShift=np.max([ p.position[0] for p in tissue.vertices[v2.id].periodic_partners])
                    V2XShift=np.max([ p.position[0] for p in tissue.vertices[v1.id].periodic_partners])
                else: #Otherwise it is from the right to left and we need to know the minimum value
                    XShift=np.min([ p.position[0] for p in tissue.vertices[v2.id].periodic_partners])
                    V2XShift=np.min([ p.position[0] for p in tissue.vertices[v1.id].periodic_partners])
                
                #Same logic for Y
                if(deltaY<0.001): #the shift is along a boundary from the bottom to the top
                    YShift=np.max([ p.position[1] for p in tissue.vertices[v2.id].periodic_partners])
                    V2YShift=np.max([ p.position[1] for p in tissue.vertices[v1.id].periodic_partners])
                    
                else: #the shift is along a boundary from the top to the bottom, need minimum 
                    YShift=np.min([ p.position[1] for p in tissue.vertices[v2.id].periodic_partners])
                    V2YShift=np.min([ p.position[1] for p in tissue.vertices[v1.id].periodic_partners])

                if(np.abs(XShift)<np.abs(YShift)):
                   #the x positions are at the same value, but the boundary means we expected a shift along X as well, creating a new vertex at the existing Y but shifted X by the largest distance between the v1 and its own periodic partners.
                   #Use the existing functions from mesh.py to add a vertex, we know the x position is
                    print("Testing method A")
                    newVertexX=vertexAddToC.position[0]+V2XShift
                    newVertexY=vertexAddToC.position[1]
                    
                elif(np.abs(XShift)>np.abs(YShift)):
                    print("Testing method B")
                    #Same as above, but creating a vertex with the y shift of v2.
                    newVertexX=vertexAddToC.position[0]
                    newVertexY=vertexAddToC.position[1]+V2YShift
                    
                newVertexC = tissue.get_vertex(newVertexX, newVertexY)
                if(testMessages):
                    print("New vertex created at",newVertexC.position[0],newVertexC.position[1])
                vertexAddToC=newVertexC
                v1C=newVertexC.position
                if(testMessages):
                    print("Testing C: New length =",np.linalg.norm(v1C - v2C))
                   #the y positions are at the same value, but the boundary means we expected a shift along y as well, creating a new vertex at the existing x but shifted y...
                #It can only be a periodic shift along x or along y, vertexAddToD must share with v1 either the x or y coordinate
                                
            #cross_before = cross2d(v1C[:2] - beforeMoveC.position[:2],v2C[:2] -v1C[:2])            
            #cross_after = cross2d(v2C[:2] - v1C[:2],afterMoveC.position[:2] - v2C[:2])
            #cross_after=cross2d(v2C[:2]- beforeMoveC.position[:2],v1C[:2] -v2C[:2])
            #cross_before = cross2d(v2C[:2] - beforeMoveC.position[:2],v1C[:2] - beforeMoveC.position[:2])
            #cross_after = cross2d(afterMoveC.position[:2] - v2C[:2],v1C[:2] - v2C[:2])
            '''
            B = beforeMoveC.position[:2]
            V = v1C[:2]
            A = afterMoveC.position[:2]
            Q = v2C[:2]

            # Existing local orientation
            orientation = cross2d(V - B, A - V)

            # Which side of B -> V is Q on?
            cross_before = cross2d(Q - B, V - Q)

            # Which side of V -> A is Q on?
            cross_after = cross2d(Q - V, A - Q)

            before_valid = cross_before * orientation > 0
            after_valid  = cross_after * orientation > 0
            if(testMessages):
                print("C:")
                print("beforeMoveCPosition=", beforeMoveC.position[:2])
                print("v1C=",v1C[:2], "v2C=",v2C[:2]);
                print("orientation =", orientation)
                print("cross_before =", cross_before)
                print("cross_after  =", cross_after)
                print("i1C=",i1C, "vertex initially on cell C =",cellC.vertices[i1C].id)
                    
            if before_valid:#cross_before * orientationC > 0:
                if(testMessages):
                    print("(1)Adding to cell C=",cellC.id," vertex", vertexAddToC.id, "between vertex" ,beforeMoveC.id,"and",cellC.vertices[(i1C) % len(cellC.vertices)].id)
                cellC.vertices.insert((i1C) %len(cellC.vertices) , vertexAddToC)
                
                #cellC.vertices.insert(i1C +1 , vertexAddToC)
            elif after_valid:#cross_after * orientationC > 0:
                if(testMessages):
                    print("(2)Adding to cell C=",cellC.id," vertex", vertexAddToC.id, "between vertex" ,cellC.vertices[(i1C+1)].id,"and",cellC.vertices[(i1C+1) % len(cellC.vertices)].id)
                
                cellC.vertices.insert((i1C+1) % len(cellC.vertices) , vertexAddToC)

            else:
                print("WARNING: could not determine C insertion")
            '''
                
            # ------------------------------------------------------------
            # Cell D
            # D originally contains v2, so locate v2, periodically if needed
            # ------------------------------------------------------------

            vertexAddToD=closest_periodic_vertex(v1, cellD)
            i2D=find_periodic_vertex_index(cellD, v2)
            v1D=vertexAddToD.position
            v2D=cellD.vertices[i2D].position
            
            if(np.linalg.norm(v1D - v2D)>new_edge_length+0.1):
                #need to create a new vertex for v2C periodically because the distance is too large.
                if(testMessages):
                    print("Warning: distance between v1 and v2 on cell D is too large")
                    
                    print("Delta x between vertexAddToD and v1 =(",v1.position[0]-vertexAddToD.position[0],",",v1.position[1]-vertexAddToD.position[1]);
                deltaX=v1.position[0]-vertexAddToD.position[0]
                deltaY=v1.position[1]-vertexAddToD.position[1]
                
                if(deltaX<0.001): #The shift is along a boundary from the left side of the surface to the right, we want to know the largest degree of the shift
                    XShift=np.max([ p.position[0] for p in tissue.vertices[v1.id].periodic_partners])
                    V2XShift=np.max([ p.position[0] for p in tissue.vertices[v2.id].periodic_partners])
                else: #Otherwise it is from the right to left and we need to know the minimum value
                    XShift=np.min([ p.position[0] for p in tissue.vertices[v1.id].periodic_partners])
                    V2XShift=np.min([ p.position[0] for p in tissue.vertices[v2.id].periodic_partners])
                
                #Same logic for Y
                if(deltaY<0.001): #the shift is along a boundary from the bottom to the top
                    YShift=np.max([ p.position[1] for p in tissue.vertices[v1.id].periodic_partners])
                    V2YShift=np.max([ p.position[1] for p in tissue.vertices[v2.id].periodic_partners])
                    
                else: #the shift is along a boundary from the top to the bottom, need minimum 
                    YShift=np.min([ p.position[1] for p in tissue.vertices[v1.id].periodic_partners])
                    V2YShift=np.min([ p.position[1] for p in tissue.vertices[v2.id].periodic_partners])
    
                if(np.abs(XShift)<np.abs(YShift)):
                   #the x positions are at the same value, but the boundary means we expected a shift along X as well, creating a new vertex at the existing Y but shifted X by the largest distance between the v1 and its own periodic partners.
                   #Use the existing functions from mesh.py to add a vertex, we know the x position is
                    print("Testing method A")
                    newVertexX=vertexAddToD.position[0]+V2XShift
                    newVertexY=vertexAddToD.position[1]
                    
                elif(np.abs(XShift)>np.abs(YShift)):
                    print("Testing method B")
                    #Same as above, but creating a vertex with the y shift of v2.
                    newVertexX=vertexAddToD.position[0]
                    newVertexY=vertexAddToD.position[1]+V2YShift
                    
                newVertexD = tissue.get_vertex(newVertexX, newVertexY)
                if(testMessages):
                    print("New vertex created at",newVertexD.position[0],newVertexD.position[1])
                vertexAddToD=newVertexD
                v1D=newVertexD.position
                if(testMessages):
                    print("Testing D: New length =",np.linalg.norm(v1D - v2D))
                   #the y positions are at the same value, but the boundary means we expected a shift along y as well, creating a new vertex at the existing x but shifted y...
                #It can only be a periodic shift along x or along y, vertexAddToD must share with v1 either the x or y coordinate
                
            #cross_before = cross2d(v1D[:2] - beforeMoveD.position[:2],v2D[:2] -v1D[:2])
            #cross_after = cross2d(v1D[:2] - v1D[:2],afterMoveD.position[:2] - v2D[:2])
            #cross_after=cross2d(v2D[:2]- beforeMoveD.position[:2],v1D[:2] -v2D[:2])
            #cross_before = cross2d(v2D[:2] - beforeMoveD.position[:2],v1D[:2] - beforeMoveD.position[:2])
            #cross_after = cross2d(afterMoveD.position[:2] - v2D[:2],v1D[:2] - v2D[:2])
            '''
            B = beforeMoveD.position[:2]
            V = v2D[:2]
            A = afterMoveD.position[:2]
            Q = v1D[:2]

            # Existing local orientation
            orientation = cross2d(V - B, A - V)

            # Which side of B -> V is Q on?
            cross_before = cross2d(Q - B, V - Q)

            # Which side of V -> A is Q on?
            cross_after = cross2d(Q - V, A - Q)

            before_valid = cross_before * orientation > 0
            after_valid  = cross_after * orientation > 0
            if(testMessages):
                print("D:")
                print("beforeMoveDPosition=", beforeMoveD.position[:2])
                print("v1D=",v1D[:2], "v2D=",v2D[:2]);
                print("orientation =", orientation)
                print("cross_before =", cross_before)
                print("cross_after  =", cross_after)
                print("i2D=",i2D, "vertex initially on cell D=",cellD.vertices[i2D].id)
            
            if before_valid:#cross_before * orientationD > 0:
                if(testMessages):
                    print("(3)Adding to cell D=",cellD.id," vertex", vertexAddToD.id, "between vertex" ,beforeMoveD.id,"and",cellD.vertices[(i2D) % len(cellD.vertices)].id)
                
                cellD.vertices.insert((i2D) % len(cellD.vertices) , vertexAddToD)
            elif after_valid:#cross_after * orientationD > 0:                
                if(testMessages):
                    print("(4)Adding to cell D=",cellD.id," vertex", vertexAddToD.id, "between vertex" ,cellD.vertices[i2D].id,"and",cellD.vertices[(i2D+1) % len(cellD.vertices)].id)
                    
                cellD.vertices.insert((i2D+1)% len(cellD.vertices) , vertexAddToD)
            else:
                print("WARNING: could not determine D insertion")
            '''
            
            AdditionIndexC,AdditionIndexD= vertex_addition_creates_intersection(cellC, cellD, cellA, cellB, vertexAddToC, vertexAddToD, tol=1e-12)
            print("Adding to cell C=",cellC.id," vertex", vertexAddToC.id, "between vertex" ,cellC.vertices[(AdditionIndexC)% len(cellC.vertices)].id,"and",cellC.vertices[(AdditionIndexC+1)% len(cellC.vertices) ].id)
            print("Adding to cell D=",cellD.id," vertex", vertexAddToD.id, "between vertex" ,cellD.vertices[(AdditionIndexD)% len(cellD.vertices)].id,"and",cellD.vertices[(AdditionIndexD+1)% len(cellD.vertices) ].id)
            
            cellC.vertices.insert((AdditionIndexC)% len(cellC.vertices) , vertexAddToC)
            cellD.vertices.insert((AdditionIndexD)% len(cellD.vertices) , vertexAddToD)
            
            # ------------------------------------------------------------
            # We have now added the vertices to the new locations in the cells C,D that they will belong to, and removed them 
            # from cell A and B.
            # We now need to update the rest of the data structures appropriately.
            # ------------------------------------------------------------
            
            
            # ------------------------------------------------------------
            # Other cell updates? 
            # None needed: vertices were added and removed as needed already, over the periodic boundary.
            # ------------------------------------------------------------

            #------------------------------------------------------------
            # class Vertex: updates. Vertex maintains a list of positions of a vertex, 
            # cells a vertex is in , 
            # periodic partners to a vertex 
            # edges the vertex is in
            # Each of these needs to be updated for the changed vertices v1 and v2.
            # print(f"v1 edges: {[p.id for p in edge.v1.edges]}")
            # ------------------------------------------------------------
            
            # Most important is the list of edges that a vertex was in. If a boundary causes the 
            
            #------------------------------------------------------------
            # class Edge:   id: int, v1: Vertex , v2: Vertex,  cells: list = field(default_factory=list), periodic_partners: 
            # list = field(default_factory=list)             
            # ------------------------------------------------------------

            #id doesn't need to be updated. v1 v2 potentially need to be swapped to the periodic equivalents
            
            #
            # The edge between v1 and v2 remains - leave v1.edge and v2.edges unchanged, unlcess a boundary condition 
            # necessitates it. We check if v1 and v2 remain on the original cell, or the periodic partners are instead, in 
            # which case, the vertices between edge 1 and 2 are swapped to the periodic partner.
            #
            

            
            # Each of the vertices v1 and v2 is now no longer bounding one of cells A, B 
            # Update the vertex v.cells list to remove only the cell from A and B from which each vertex was actually removed            
            if removeA is v1:                
                for v in vertex_list_with_periodic_equivalent_vertices(v1):
                    v.cells = [c for c in v.cells if c.id != cellA.id]
            else:                
                for v in vertex_list_with_periodic_equivalent_vertices(v2):
                    v.cells = [c for c in v.cells if c.id != cellA.id]

            if removeB is v1:                
                for v in vertex_list_with_periodic_equivalent_vertices(v1):
                    v.cells = [c for c in v.cells if c.id != cellB.id]
            else:                
                for v in vertex_list_with_periodic_equivalent_vertices(v2):
                    v.cells = [c for c in v.cells if c.id != cellB.id]
                                                            
            #We've now updated the cell lists on the cell objects, cells do not track edges so nothing to change there.
            #However We must also update the vertex, edge lists. 
            #For the vertices...
            #vertices.cells: list = field(default_factory=list) #The cells the vertex belongs to                
            #vertices.edges: list = field(default_factory=list)                        
            
            
            #The edge has a list edge.cells which contains the cells the edge bounds, remove the old ones and add the new ones
            #As for the vertices, the actual vertex ids involved do not change.
            
            #i_removeA = find_periodic_vertex_index(cellA, removeA)
            #i_removeB = find_periodic_vertex_index(cellB, removeB)

            if cellA in edge.cells:
                edge.cells.remove(cellA)
            
            if cellB in edge.cells:
                edge.cells.remove(cellB)
                
            if cellC not in edge.cells:
                edge.cells.append(cellC)

            if cellD not in edge.cells:
                edge.cells.append(cellD)
                                    
            #The edge v1-v2 is unchanged, unless it is shifted across a boundary but the cell list is changed.
            #update vertices on the edge for periodicity
                        
            if(vertexAddToC.id != v2.id):
                if(testMessages):
                    print("Was over a periodic boundary, removing vertices from edge and replacing with periodic equivalents")
               
                
                        
            if(vertexAddToD.id != v1.id):
                if(testMessages):
                    print("Was over a periodic boundary, removing vertices from edge and replacing with periodic equivalents")
                
            
                
            #processed_t1_vertices.add(pair_key)
            processed_edges.update(edge_ids)
            
            #Also remove any vertices which are no longer connected to any edges.
            vertices = remove_unconnected_vertices(tissue.vertices)
            num_t1_transitions=num_t1_transitions+1;
            if(num_t1_transitions >2):
                plotting.plot_mesh(tissue,show_cell_ids=True,show_vertex_ids=True,show_periodic=False,show_edge_ids=True)
                raise RuntimeError("Stopping here for debugging")
    return num_t1_transitions




#To perform a T2 transition, we need to know when cells are below a area threshold.
def find_small_cells(tissue, threshold):
    """
    Return cell ids whose current area is below `threshold`.
    """
    small_cells = []

    for cell in tissue.cells:
        if(cell.area() < threshold):
            small_cells.append(cell.id)
    

    return small_cells

def find_large_cells(tissue, threshold):
    """
    Return cell ids whose current area is below `threshold`.
    """
    large_cells = []

    for cell in tissue.cells:
        if(cell.area() > threshold):
            large_cells.append(cell.id)
    

    return large_cells
