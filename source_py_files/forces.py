##########File created by Ian Estabrook for the CMCB course 2026.

from . import energy
import numpy as np
import time 

def force_single_vertex(tissue,vertex_id,ka=1,kp=1,Lambda=1,perturb_distance_x=0.001,perturb_distance_y=0.001, friction=0.01,noise_strength=0.01,print_force=True):
    
    #The force on a vertex is defined by doing some perturbations to the position of the vertex, measure the change in the energy
    # and then using central difference methods to generate a 2D force vector 
    
    #energy_before=energy.local_cell_energy(tissue, vertex_id, ka=ka, kp=kp, Lambda=Lambda)
        
    tissue.move_vertex(vertex_id,[perturb_distance_x,0.0,0.0])
    energy_positive_x_perturb=energy.local_cell_energy(tissue, vertex_id, ka=ka, kp=kp, Lambda=Lambda)
    
    tissue.move_vertex(vertex_id,[-perturb_distance_x,perturb_distance_y,0.0])        
    energy_positive_y_perturb=energy.local_cell_energy(tissue, vertex_id, ka=ka, kp=kp, Lambda=Lambda)
    
    tissue.move_vertex(vertex_id,[-perturb_distance_x,-perturb_distance_y,0.0])    
    energy_negative_x_perturb=energy.local_cell_energy(tissue, vertex_id, ka=ka, kp=kp, Lambda=Lambda)    
    
    tissue.move_vertex(vertex_id,[+perturb_distance_x,-perturb_distance_y,0.0])    
    energy_negative_y_perturb=energy.local_cell_energy(tissue, vertex_id, ka=ka, kp=kp, Lambda=Lambda)
    
    tissue.move_vertex(vertex_id,[0.0,+perturb_distance_y,0.0])    
    
    #if(print_force):
    #    print(f"The energies of perturbing the vertex {vertex_id} are {energy_before:.16e},{energy_positive_x_perturb:.16e},{energy_positive_y_perturb:.16e},{energy_negative_x_perturb:16e},{energy_negative_y_perturb:.16e}")
    
    dEdx=(energy_positive_x_perturb-energy_negative_x_perturb)/(2.0*perturb_distance_x)
    dEdy=(energy_positive_y_perturb-energy_negative_y_perturb)/(2.0*perturb_distance_y)
    
    force_x=-dEdx #* friction is accounted for after
    force_y=-dEdy #* friction
    if(print_force):
        print(f"The resulting force vector on vertex {vertex_id} is ({force_x:.16e},{force_y:.16e}");
    
    return(force_x,force_y,0.0)


def forces_all_vertices(tissue,ka=1,kp=1,Lambda=1,perturb_distance_x=0.001,perturb_distance_y=0.001, friction=0.01,print_force=False):

    forces = {}
    #start_time = time.time()
    for vertex in tissue.vertices:
        vertex_id = vertex.id

        forces[vertex_id] = force_single_vertex(tissue,vertex_id,ka=ka,kp=kp,Lambda=Lambda,perturb_distance_x=perturb_distance_x,perturb_distance_y=perturb_distance_y,friction=friction,print_force=print_force)
    
    #print("--- %s seconds to calculate all forces ---" % (time.time() - start_time))
    return forces


def forces_all_vertices_with_noise(tissue,ka=1,kp=1,Lambda=1,perturb_distance_x=0.001,perturb_distance_y=0.001, friction=0.01,noise_sigma=0.0,print_force=False):

    forces = {}
    #start_time = time.time()
    for vertex in tissue.vertices:
        vertex_id=vertex.id
        force_x, force_y, force_z = force_single_vertex(tissue,vertex_id,ka=ka,kp=kp,Lambda=Lambda,perturb_distance_x=perturb_distance_x,perturb_distance_y=perturb_distance_y,friction=friction,print_force=print_force)

        # Add Gaussian noise independently to x and y
        force_x += np.random.normal(0.0, noise_sigma)
        force_y += np.random.normal(0.0, noise_sigma)

        forces[vertex_id] = (force_x, force_y, force_z)
    #print("--- %s seconds to calculate all forces ---" % (time.time() - start_time))
    return forces



def single_vertex_second_derivative(tissue,vertex_id,ka=1.0,kp=1.0,Lambda=0.0,perturb_distance=1e-4,print_curvature=True):
    """
    Calculate the second derivatives of the local energy with respect
    to x and y displacement of a single vertex.

    Returns:
        d2Edx2, d2Edy2
    """

    # Energy at the original position
    E0 = energy.local_cell_energy(tissue, vertex_id, ka=ka, kp=kp, Lambda=Lambda)

    # ---------------------------------------------------------
    # X direction
    # ---------------------------------------------------------

    # E(x + delta)
    tissue.move_vertex(vertex_id, [perturb_distance, 0.0, 0.0])

    E_plus_x = energy.local_cell_energy( tissue, vertex_id,ka=ka, kp=kp,
        Lambda=Lambda
    )

    # Return to original position
    tissue.move_vertex(
        vertex_id,
        [-perturb_distance, 0.0, 0.0]
    )

    # E(x - delta)
    tissue.move_vertex(
        vertex_id,
        [-perturb_distance, 0.0, 0.0]
    )

    E_minus_x = energy.local_cell_energy(
        tissue,
        vertex_id,
        ka=ka,
        kp=kp,
        Lambda=Lambda
    )

    # Return to original position
    tissue.move_vertex(
        vertex_id,
        [perturb_distance, 0.0, 0.0]
    )

    # ---------------------------------------------------------
    # Y direction
    # ---------------------------------------------------------

    # E(y + delta)
    tissue.move_vertex(
        vertex_id,
        [0.0, perturb_distance, 0.0]
    )

    E_plus_y = energy.local_cell_energy(tissue, vertex_id, ka=ka, kp=kp, Lambda=Lambda)

    # Return to original position
    tissue.move_vertex( vertex_id,[0.0, -perturb_distance, 0.0])

    # E(y - delta)
    tissue.move_vertex( vertex_id,[0.0, -perturb_distance, 0.0])

    E_minus_y = energy.local_cell_energy(tissue, vertex_id, ka=ka, kp=kp, Lambda=Lambda)

    # Return to original position
    tissue.move_vertex(vertex_id,[0.0, perturb_distance, 0.0])

    # ---------------------------------------------------------
    # Central second derivatives
    # ---------------------------------------------------------

    d2Edx2 = (E_plus_x - 2.0 * E0 + E_minus_x) / perturb_distance**2

    d2Edy2 = ( E_plus_y- 2.0 * E0 + E_minus_y) / perturb_distance**2

    if print_curvature:
        print( f"Vertex {vertex_id}: " f"d²E/dx² = {d2Edx2:.6e}, " f"d²E/dy² = {d2Edy2:.6e}" )

    return d2Edx2, d2Edy2