import dolfin as df
import numpy as np

#Extract an interface
def ExtractInterface(mesh, subdomains, marker_1, marker_2):
    tdim = mesh.topology().dim()
    fdim = tdim - 1

    mesh.init(fdim, tdim)
    mesh.init(fdim, 0)
    f2c = mesh.topology()(fdim, tdim)
    f2v = mesh.topology()(fdim, 0)

    # --- locate interface facets ---
    interface_facets = []
    for facet in df.facets(mesh):
        fid = facet.index()
        cells = f2c(fid)
        if len(cells) == 2:
            m0, m1 = subdomains[cells[0]], subdomains[cells[1]]
            if set([m0, m1]) == set([marker_1, marker_2]):
                interface_facets.append(fid)

    if not interface_facets:
        raise ValueError(f"No interface found between subdomains {marker_1} and {marker_2}.")

    # --- collect unique vertices ---
    old_vertex_indices = []
    for fid in interface_facets:
        old_vertex_indices.extend(f2v(fid).tolist())

    unique_old = sorted(set(old_vertex_indices))
    old_to_new = {old: new for new, old in enumerate(unique_old)}

    # --- determine interface cell type from string ---
    cell_desc = mesh.ufl_cell().cellname()  # stable across all FEniCS versions
    if cell_desc == "tetrahedron":
        iface_cell_type = "triangle"
        verts_per_cell  = 3
    elif cell_desc == "triangle":
        iface_cell_type = "interval"
        verts_per_cell  = 2
    elif cell_desc == "hexahedron":
        iface_cell_type = "quadrilateral"
        verts_per_cell  = 4
    else:
        raise NotImplementedError(f"Unsupported cell type: {cell_desc}")

    gdim = mesh.geometry().dim()
    coords = mesh.coordinates()

    # --- build mesh with MeshEditor ---
    iface_mesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(iface_mesh, iface_cell_type, fdim, gdim)

    editor.init_vertices(len(unique_old))
    for old_idx, new_idx in old_to_new.items():
        editor.add_vertex(new_idx, coords[old_idx])

    editor.init_cells(len(interface_facets))
    for cell_idx, fid in enumerate(interface_facets):
        new_verts = [old_to_new[v] for v in f2v(fid).tolist()]
        editor.add_cell(cell_idx, np.array(new_verts, dtype=np.uintp))

    editor.close()

    return iface_mesh, old_to_new

#functions to project exist points to polygonal fem mesh, works without volume
#returns a 2d array with l2-distances between each exit point and all boundary mesh vertices
def ComputeMeshDistance(Points, MyBoundaryMesh):
    GeoDim = MyBoundaryMesh.geometry().dim()
    #compute distances
    dists = np.zeros([Points.shape[0], MyBoundaryMesh.num_entities(0)])
    for i in range(Points.shape[0]):
        for v in df.vertices(MyBoundaryMesh):
            MyCoords = np.zeros(GeoDim)
            for j in range(GeoDim):
                MyCoords[j] = v.x(j)
            dists[i, v.index()] = np.linalg.norm(MyCoords - Points[i,:])
    return dists
    
def ProjectToBoundary(Points, MyBoundaryMesh):
    ProjectedPoints = []
    ProjectedCellIndices = []
    #2d array with l2-distances between each exit point and all boundary mesh vertices
    dists = ComputeMeshDistance(Points, MyBoundaryMesh)
    #sort the list so for each exit point, the index of the nearest mesh point is first
    dists_index = np.argsort(dists, axis=1)
    
    TopDim = MyBoundaryMesh.topology().dim()
    GeoDim = MyBoundaryMesh.geometry().dim()
    
    for i in range(Points.shape[0]):
        #v1 is now the boundary mesh vertex closest to exit point i
        v1 = df.Vertex(MyBoundaryMesh, dists_index[i, 0])
        #loop over the edges for which v1 is a corner...
        for f in df.cells(v1):
            found = 0
            for v in df.vertices(f):
                #from index 0 to index TopDim+1. So this gets the first TopDim entries.
                #Because it's sorted and TopDim=1, we check if v is the next closest vertex
                if v.index() in dists_index[i, :TopDim+1]:
                    found = found + 1
            if found == TopDim+1:
                break
        if found != TopDim+1:
            print("ProjectToBoundary: Error!")
            print("Does the IsInside Routine match the mesh??")
            print(Points[i])
            exit()
        #MyFacet is now the facet with the two vertices cloest to exit point i
        MyFacet = f
        #now project the exit point in normal direction onto this facet
        MyNormal = MyFacet.cell_normal().array()[:GeoDim]
        nnt = np.outer(MyNormal, MyNormal)
        MyCoords = np.zeros(GeoDim)
        for j in range(GeoDim):
            MyCoords[j] = v1.x(j)
        x_proj = (np.eye(GeoDim) - nnt)@Points[i,:] + nnt@MyCoords[:]
        ProjectedPoints.append(x_proj)
        ProjectedCellIndices.append(MyFacet.index())
    return np.array(ProjectedPoints), ProjectedCellIndices
        
class MyMesh():
    mesh = None
    subdomains = None
    boundaries = None
    dx = None
    ds = None
    
    def open(self, MeshName):
        self.mesh = df.Mesh("mesh/"+MeshName+".xml")
        self.subdomains = df.MeshFunction("size_t", self.mesh, "mesh/"+MeshName+"_physical_region.xml")
        self.boundaries = df.MeshFunction("size_t", self.mesh, "mesh/"+MeshName+"_facet_region.xml")

        #link boundary identifiers to integrals
        self.dx = df.Measure("dx", domain=self.mesh, subdomain_data=self.subdomains)
        self.ds = df.Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
    
    #make a function that can answer inside or outside?
    def MakeInsideOutside(self, inside_domains):
        InsideOutside = df.Function(df.FunctionSpace(self.mesh, "DG", 0))
        InsideOutside.rename("InsideIndicator", "label")
        d2c = [cell for cell in range(self.mesh.num_cells())
               for dof in InsideOutside.function_space().dofmap().cell_dofs(cell)]
        for i in range(len(InsideOutside.vector())):
            if self.subdomains[d2c[i]] in inside_domains:
                InsideOutside.vector()[i] = -1.0
            else:
                InsideOutside.vector()[i] = +1.0
        InsideOutside.vector().apply("")
        return InsideOutside

    def MakePINNSplit(self, interface_id, inside_domains, data_function):
        x_domain = []
        data_obs = []
        x_boundary = []
        x_subdomain_inside = []
        x_subdomain_outside = []
        d2v = df.dof_to_vertex_map(data_function.function_space())
        for i in range(len(data_function.vector())):
            v = df.Vertex(self.mesh, d2v[i])
            GeoDim = self.mesh.geometry().dim()
            Coordinate = []
            for j in range(GeoDim):
                Coordinate.append(v.x(j))
            #test if this vertex is on a boundary or not:
            on_boundary = False
            for f in df.facets(v):
                marker = self.boundaries[f.index()]
                #boundary is actually the inner interface only here
                if marker in interface_id:
                    on_boundary = True
                    break
            if on_boundary:
                x_boundary.append(Coordinate)
            else:
                #boundary does not belong to pde loss
                #x_domain.append(Coordinate)
                ##data for Laplace
                #data_obs.append(data_function.vector()[i])
                #also divide x into the two subdomains:
                is_inside = False
                for c in df.cells(v):
                    marker = self.subdomains[c.index()]
                    if marker in inside_domains:
                        is_inside = True
                        break
                if is_inside:
                    x_subdomain_inside.append(Coordinate)
                else:
                    x_subdomain_outside.append(Coordinate)
            #boundary belongs to data loss
            x_domain.append(Coordinate)
            #data for Laplace
            data_obs.append(data_function.vector()[i])
        return x_domain, x_boundary, x_subdomain_inside, x_subdomain_outside, data_obs
