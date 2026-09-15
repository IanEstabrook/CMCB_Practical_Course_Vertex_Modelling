##########File created by Ian Estabrook for the CMCB course 2026.

import matplotlib.pyplot as plt
from .data_types import Vertex, Cell, Edge
import numpy as np

def plot_mesh(tissue, show_vertex_ids=False, show_periodic=False,show_cell_ids=False,show_edge_ids=False):

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot cells
    for cell in tissue.cells:

        coords = np.array([v.position[:2] for v in cell.vertices])

        # close polygon
        coords = np.vstack([coords, coords[0]])

        ax.plot(coords[:, 0], coords[:, 1],'k-', linewidth=1)

        # label cell centre
        if(show_cell_ids):
            ax.text(cell.centroid()[0],cell.centroid()[1],f"C{cell.id}", fontsize=8, color="blue", ha="center", va="center")
        #ax.text(cell.centre[0],cell.centre[1],f"C{cell.i, cell.j}", fontsize=8, color="blue", ha="center", va="center")
    # Plot vertices
    for vertex in tissue.vertices:
        x, y = vertex.position[:2]
        ax.plot(x, y, 'ro', markersize=4)

        if show_vertex_ids:
            ax.text(x, y, f"{vertex.id}", fontsize=8,color="red",ha="left", va="bottom")


    # Plot periodic links
    if show_periodic:    
        plotted_pairs = set()
        for vertex in tissue.vertices:    
            for partner in vertex.periodic_partners:    
                # avoid drawing the same line twice
                pair = tuple(sorted([vertex.id, partner.id]))

                if pair in plotted_pairs:
                    continue

                plotted_pairs.add(pair)

                x = [vertex.position[0],partner.position[0]]

                y = [vertex.position[1],partner.position[1]]

                ax.plot(x,y,'g--',linewidth=1)


    if show_edge_ids:

        for edge in tissue.edges:

            r1 = edge.v1.position[:2]
            r2 = edge.v2.position[:2]

            # Midpoint of the edge
            midpoint = 0.5 * (r1 + r2)

            # Edge vector
            edge_vector = r2 - r1

            # Length of edge
            length = np.linalg.norm(edge_vector)

            if length > 0:
                # Unit normal to the edge
                normal = np.array([-edge_vector[1],edge_vector[0]]) / length

                # Small offset so label does not sit directly on edge
                offset = 0.08 * normal
            else:
                offset = np.array([0.0, 0.0])

            label_position = midpoint + offset

            ax.text(label_position[0],label_position[1],f"E{edge.id}",
            fontsize=8,color="green",ha="center",va="center",bbox=dict(facecolor="white",edgecolor="none",alpha=0.7,pad=1))
    
    ax.set_aspect("equal")    
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title("Vertex model mesh")

    plt.show()
    
def plot_energy_landscape(dx_values,dy_values,energy_change,selectedVertexId):

    plt.figure(figsize=(7, 6))

    DX, DY = np.meshgrid(dx_values, dy_values)

    contour = plt.contourf(DX,DY,energy_change,levels=30)

    plt.colorbar(contour, label=r'$\Delta E$')

    plt.xlabel(r'$\Delta x$')
    plt.ylabel(r'$\Delta y$')

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.title(
        f'Local energy landscape — vertex {selectedVertexId}'
    )

    plt.show()