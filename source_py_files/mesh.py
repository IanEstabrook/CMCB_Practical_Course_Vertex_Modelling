import numpy as np
from .data_types import Vertex, Cell, Edge

def get_vertex(tissue, x, y, tol=8):
    #Returns an existing vertex if one already exists, otherwise creates a new one.

    key = (round(x, tol), round(y, tol))

    if key in tissue.vertex_lookup:
        return tissue.vertex_lookup[key]

    v = Vertex(id=len(tissue.vertices),position=np.array([x, y, 0.0]))

    tissue.vertices.append(v)
    tissue.vertex_lookup[key] = v

    return v

def get_cell_initialistion_only(tissue, i, j,ny):    
    #During intialisation, the cells are assigned values of d = i*ny +j, so just need to use the z

    return tissue.cells[i*ny + j]

def identify_boundary_positions(tissue, nx, ny, top_bottom_boundary_periodicity=True,side_boundary_periodicity=True):
    #For every cell in our mesh, determine if there are any top/bottom shared vertices, and record this appropriately so that such periodic boundaries can later be updated. 
    
    #Note that because we loop over every cell, we only have to carefully record shared vertices once.
    
    for cell in tissue.cells:    
        # Left boundary
        if cell.i == 0 and side_boundary_periodicity==True :#The first line of cells at the left boundary have id. value i=0.
            partner = get_cell_initialistion_only(tissue,nx-1, cell.j,ny)
            cell.vertices[2].periodic_partners.append(partner.vertices[0])
            cell.vertices[3].periodic_partners.append(partner.vertices[5])      
        
        # Right boundary
        if cell.i == nx-1 and side_boundary_periodicity==True: #The last line of cells to the right of our vertex mesh have index i= nx-1.            
            partner = get_cell_initialistion_only(tissue,0, cell.j,ny)
            cell.vertices[5].periodic_partners.append(partner.vertices[3])
            cell.vertices[0].periodic_partners.append(partner.vertices[2])        

        # Bottom boundary
        if cell.i>0 and cell.j==0 and top_bottom_boundary_periodicity==True:
            partner = get_cell_initialistion_only(tissue,(cell.i-1)%nx, ny-1,ny)
            cell.vertices[3].periodic_partners.append(partner.vertices[1])
            cell.vertices[4].periodic_partners.append(partner.vertices[0])
        elif cell.i==0 and cell.j==0 and top_bottom_boundary_periodicity==True:
            #Special case of this cell not being checked taken care of manually
            partner = get_cell_initialistion_only(tissue,0, ny-1,ny)            
            cell.vertices[4].periodic_partners.append(partner.vertices[2])            
        elif  cell.i==0 and cell.j==ny-1 and top_bottom_boundary_periodicity==True:
            partner = get_cell_initialistion_only(tissue,0, 0,ny)            
            cell.vertices[2].periodic_partners.append(partner.vertices[4])
        
        # Top boundary
        if (cell.i < nx-1 and cell.j == ny-1 and top_bottom_boundary_periodicity==True): 
            partner = get_cell_initialistion_only(tissue,(cell.i+1)%nx, 0,ny)
            cell.vertices[0].periodic_partners.append(partner.vertices[4])
            cell.vertices[1].periodic_partners.append(partner.vertices[3])
        elif (cell.i == nx-1 and cell.j == ny-1 and top_bottom_boundary_periodicity==True): 
            partner = get_cell_initialistion_only(tissue,nx-1, 0,ny)
            cell.vertices[1].periodic_partners.append(partner.vertices[5])
        elif (cell.i == nx-1 and cell.j == 0 and top_bottom_boundary_periodicity==True): 
            partner = get_cell_initialistion_only(tissue,nx-1, ny-1,ny)
            cell.vertices[5].periodic_partners.append(partner.vertices[1])
    
        #In addition to the above general conditions along the entire layer, the cells (0,0) (0,ny-1),(nx-1,0) and (nx-1,ny-1) potentially have additional connections diagonally which need to implemented separately.
        if  top_bottom_boundary_periodicity==True and side_boundary_periodicity==True:
            if cell.i == 0 and cell.j==0: 
                #cell vertex 3 is shared with nx-1,ny-1 vertex 1
                partner = get_cell_initialistion_only(tissue,nx-1, ny-1,ny)
                cell.vertices[3].periodic_partners.append(partner.vertices[1])      
                #cell vertex 4 is shared with nx-1,ny-1 vertex 0               
                cell.vertices[4].periodic_partners.append(partner.vertices[0])

            if cell.i == nx-1 and cell.j==ny-1:
                #Also add the reciprocal of the previous two conditions
                #cell vertex 0 is shared with nx-1,ny-1 vertex 4
                partner = get_cell_initialistion_only(tissue,0, 0,ny)
                cell.vertices[0].periodic_partners.append(partner.vertices[4])
                #cell vertex 1 is shared with nx-1,ny-1 vertex 3
                cell.vertices[1].periodic_partners.append(partner.vertices[3])        

                
def check_periodic_partners(tissue):
    #Because later we will update all points, it's very important that any vertices have correct lists of partners 
    #This function checks that all partner lists correctly reciprocate.
    #i.e. if vertex 1 is listed as shared with vertex 10 , then vertex 10 also is listed as shared with vertex 1.

    errors = 0

    for vertex in tissue.vertices:

        for partner in vertex.periodic_partners:

            if vertex not in partner.periodic_partners:
                print(f"ERROR: Vertex {vertex.id} lists {partner.id} as a periodic partner,but the reverse link is missing.")
                errors += 1

    if errors != 0:            
        print(f"{errors} missing reciprocal links found.")

def identify_periodic_edge_partners(tissue):
    #Because we have previously a function which identifies all periodic edges, we can 
    for edge in tissue.edges:
        v1 = edge.v1
        v2 = edge.v2

        # Find all possible periodic partners of the two vertices
        for p1 in v1.periodic_partners:
            for p2 in v2.periodic_partners:

                # Find an edge whose vertices are p1 and p2
                partner_edge = get_edge_from_vertices(tissue,p1,p2)                

                if partner_edge is not None:
                    if partner_edge not in edge.periodic_partners:
                        edge.periodic_partners.append(partner_edge)
                    if edge not in partner_edge.periodic_partners:
                        partner_edge.periodic_partners.append(edge)
                    
                    
                    for cell in partner_edge.cells:
                        if cell not in edge.cells:
                            edge.cells.append(cell)

                    for cell in edge.cells:
                        if cell not in partner_edge.cells:
                            partner_edge.cells.append(cell)
                            

def get_edge_from_vertices(tissue, v1, v2):

    for edge in tissue.edges:

        if ((edge.v1 is v1 and edge.v2 is v2) or
            (edge.v1 is v2 and edge.v2 is v1)):

            return edge

    return None                       
                        
def generate_hexagonal_sheet(tissue, nx, ny, edge_length,top_bottom_boundary_periodicity=True,side_boundary_periodicity=True):
    #nx, ny are the number of hexagons along x, y. For periodicity, both MUST be even numbers. a is the default edge length of the hexagons
    #The spacings of the centres of mass along an equally spaced hexagonal lattice are given by
    dx = np.sqrt(3) * edge_length
    dy = 1.5 * edge_length

    #For periodic surfaces, the initial condition along y must be even, otherwise the periodicity is nonsensical. This is not the case for x, which is periodic every cell unit.
    
    if(ny%2==1 and top_bottom_boundary_periodicity==True):
        ny=ny+1
    for i in range(nx):
        for j in range(ny):
            #Get the centre of mass positions
            xc = dx * i + (j % 2) * dx / 2
            yc = dy * j

            cell = Cell(id=len(tissue.cells),i=i,j=j,centre=np.array([xc,yc,0.0]))

            for k in range(6):
                #Now evenly space the points around the cell.
                theta = 2*np.pi*(k+0.5)/6

                xv = xc + edge_length*np.cos(theta)
                yv = yc + edge_length*np.sin(theta)

                vertex = tissue.get_vertex(xv, yv)

                cell.vertices.append(vertex)

                if cell not in vertex.cells:
                    vertex.cells.append(cell)

            for k in range(len(cell.vertices)):
                v1 = cell.vertices[k]
                v2 = cell.vertices[(k + 1) % len(cell.vertices)]

                edge = tissue.get_edge(v1, v2)

                if cell not in edge.cells:
                    edge.cells.append(cell)
                
                if edge not in v1.edges:
                    v1.edges.append(edge)

                if edge not in v2.edges:
                    v2.edges.append(edge)
            tissue.cells.append(cell)
            #tissue.cell_grid[i][j] = cell
    #tissue.identify_boundary_positions(nx, ny)
    identify_boundary_positions(tissue,nx,ny,top_bottom_boundary_periodicity,side_boundary_periodicity)
    #tissue.check_periodic_partners()
    check_periodic_partners(tissue)
    
    identify_periodic_edge_partners(tissue)

    
    