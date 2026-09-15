##########File created by Ian Estabrook for the CMCB course 2026.


from . import forces
from . import topology_v3
from . import plotting

import time

#A gradient descent algorithm follows the steepest descent of the energy function. This direction is already calculated in force.py.
#The purpose of this module is to run the same simulations over many timesteps and see what the final shape looks like
#You can add 

def gradient_descent_update_positions(tissue,nTimesteps=10,ka=1,kp=1,Lambda=1,perturb_distance_x=0.001,perturb_distance_y=0.001, friction=0.01,print_force=False):
        
    for i in range(nTimesteps):
        if i % 10 == 3:
            print(f"Update: iteration {i}")
            
        
        all_forces=forces.forces_all_vertices(tissue,ka=1,kp=1,Lambda=1,perturb_distance_x=0.001, perturb_distance_y=0.001,friction=0.01 ,print_force=False)
        updated_vertex_list = set()
        
        for vertex in tissue.vertices:
            vertex_id=vertex.id;
            #Check we don't update any periodic vertex multiple times. 
            if vertex_id in updated_vertex_list:
                continue
                                            
            
            force_x, force_y, force_z = all_forces[vertex_id]

            tissue.move_vertex(vertex_id,[force_x*friction, force_y*friction, force_z*friction])
            # Mark current vertex and all periodic partners as updated
            updated_vertex_list.add(vertex_id)
            
            if(vertex.periodic_partners):
                 updated_vertex_list.update(partner.id for partner in vertex.periodic_partners)

            
def gradient_descent_update_positions_with_noise(tissue,nTimesteps=10,ka=1,kp=1,Lambda=1,perturb_distance_x=0.001, perturb_distance_y=0.001, friction=0.01,noise_sigma=0.0,print_force=False):
        
    for i in range(nTimesteps):
        updated_vertex_list = set()
        
        if i % 10 == 3:
            print(f"Update: iteration {i}")
            
        all_forces=forces.forces_all_vertices_with_noise(tissue,ka=1,kp=1,Lambda=1,perturb_distance_x=perturb_distance_x, perturb_distance_y=perturb_distance_y,friction=friction,noise_sigma=noise_sigma ,print_force=False)
        
        start_time = time.time()
        for vertex in tissue.vertices:
            vertex_id=vertex.id;
            
            #Check we don't update any periodic vertex multiple times. 
            if vertex_id in updated_vertex_list:
                continue
            
            force_x, force_y, force_z = all_forces[vertex_id]            
            displacement = np.array([force_x * friction,force_y * friction,force_z * friction])
            
            # Test before actually moving the vertex that it won't intersect a neighbour
            #print("Testing moving vertex =",vertex_id)
            if not vertex_move_creates_intersection(vertex,displacement):
                #Moving_vertex_start_time = time.time()
                tissue.move_vertex(vertex_id, displacement)
                #print("--- %s seconds to update move one vertex positions ---" % (time.time() - Moving_vertex_start_time))
            #else:
                # Reject move
                #print(f"Rejected move of vertex {vertex_id}")                   
                #plotting.plot_mesh(tissue,show_cell_ids=True,show_vertex_ids=False,show_periodic=False,show_edge_ids=True)
                
            
            updated_vertex_list.add(vertex_id)
            if(vertex.periodic_partners):
                 updated_vertex_list.update(partner.id for partner in vertex.periodic_partners)
    
        #print("--- %s seconds to update all positions ---" % (time.time() - start_time))
    return 
                
                
#We also have to have a robust check to make sure a vertex move doesn't intersect any other edges, however because this easily becomes very numerically expensive, we can avoid checking every other edge and only the ones on cells that are actually neighbouring the vertex (or periodic parterners)              
import numpy as np

def orientation(a, b, c, tol=1e-12):
    """    Returns the signed cross product (b-a) x (c-a).  This provides information on   """
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def on_segment(a, b, p, tol=1e-12):
    """    True if p lies on the segment a-b, within tolerance.   If you find your simulations resulting in overlapping segments, you probably need to increase the tolerance """
    
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

def edges_share_vertex(edge1, edge2):
    return (
        edge1.v1.id == edge2.v1.id
        or edge1.v1.id == edge2.v2.id
        or edge1.v2.id == edge2.v1.id
        or edge1.v2.id == edge2.v2.id
    )

    
def get_candidate_edges_for_vertex(vertex, n_rings=2):
    """
    Find candidate edges that could intersect an edge incident
    to `vertex`.

    Uses:        vertex -> incident edges -> neighbouring cells -> cell vertices-> their edges
    """

    candidate_edges = {}    
    visited_cells = set()
    
    current_cells = {}

    # Cells directly attached to the vertex
    for edge in vertex.edges:
        for cell in edge.cells:
            current_cells[cell.id] = cell
    #Then start exploring the surrounding cells. 
    for ring in range(n_rings):

        next_cells = {}

        for cell_id, cell in current_cells.items():

            if cell_id in visited_cells:
                continue

            visited_cells.add(cell_id)

            # Add all edges belonging to this cell
            for cell_vertex in cell.vertices:
                for edge in cell_vertex.edges:
                    candidate_edges[edge.id] = edge

            # Find neighbouring cells
            for cell_vertex in cell.vertices:

                for edge in cell_vertex.edges:

                    for neighbouring_cell in edge.cells:

                        if neighbouring_cell.id not in visited_cells:
                            next_cells[neighbouring_cell.id] = neighbouring_cell

        current_cells = next_cells

        if not current_cells:
            break

    return list(candidate_edges.values())

def vertex_move_creates_intersection(vertex, displacement, tol=1e-10, periodic_recurse=True):

    incident_edges = vertex.edges
    candidate_edges = get_candidate_edges_for_vertex(vertex)
    start_time=time.time()
    
    for edge in incident_edges:

        # Construct the proposed position of this edge.
        if edge.v1.id == vertex.id:
            p1 = edge.v1.position[:2]+displacement[:2]
            p2 = edge.v2.position[:2]
        else:
            p1 = edge.v1.position[:2]
            p2 = edge.v2.position[:2]+displacement[:2]

        for other_edge in candidate_edges:

            # Don't test an edge against itself
            if other_edge.id == edge.id:
                continue

            # Edges which already share a vertex are allowed
            # to meet at that vertex.
            if edges_share_vertex(edge, other_edge):
                continue

            q1 = other_edge.v1.position[:2]
            q2 = other_edge.v2.position[:2]

            if segments_intersect(p1, p2,q1, q2,tol=tol):
                #print("Edges",edge.id,"and ",other_edge.id,"intersect? check")
                return True
             #We also need to check for any periodic repeats of the vertex.  Fortunately, this can be done by a recursive call over the periodic vertices.
            #if(periodic_recurse==True  and vertex.periodic_partners):
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
    #print("--- %s seconds to check if one move intersects another candidate edges ---" % (time.time() - start_time))    
    return False