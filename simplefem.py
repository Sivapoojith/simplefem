import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pandas as pd
import tetgen

def generate_elasticity_mat(youngs, poissons):
    E = youngs
    v = poissons
    mat = np.array([[1 - v, v, v, 0, 0, 0],
                    [v, 1 - v, v, 0, 0, 0],
                    [v, v, 1 - v, 0, 0, 0],
                    [0, 0, 0, 1 - 2*v, 0, 0],
                    [0, 0, 0, 0, 1 - 2*v, 0],
                    [0, 0, 0, 0, 0, 1 - 2*v],
                   ])
    elasticity_mat = E / ((1 + v) * (1 - 2 * v)) * mat
    return elasticity_mat


def add_local_stiffness(D, K_g, nodes, el_nodes):
    # https://academic.csuohio.edu/duffy_s/CVE_512_12.pdf
    x = np.array([nodes[el_nodes[0]][0], nodes[el_nodes[1]][0], nodes[el_nodes[2]][0], nodes[el_nodes[3]][0]])
    y = np.array([nodes[el_nodes[0]][1], nodes[el_nodes[1]][1], nodes[el_nodes[2]][1], nodes[el_nodes[3]][1]])
    z = np.array([nodes[el_nodes[0]][2], nodes[el_nodes[1]][2], nodes[el_nodes[2]][2], nodes[el_nodes[3]][2]])
    ones = np.array([1.0, 1.0, 1.0, 1.0])
    
    C = np.vstack((ones.T, x.T, y.T, z.T))
    #print("C: ", pd.DataFrame(C))
    IC = np.linalg.inv(C)
    B = np.zeros((6, 12))
    
    for i in range(4):
        B[0, i*3] = IC[1, i]
        B[1, i*3 + 1] = IC[2, i]
        B[2, i*3 + 2] = IC[3, i]
        
        B[3, i*3] = IC[2, i]
        B[3, i*3 + 1] = IC[1, i]
        
        B[4, i*3 + 1] = IC[3, i]
        B[4, i*3 + 2] = IC[2, i]
        
        B[5, i*3] = IC[3, i]
        B[5, i*3 + 2] = IC[1, i]
        
    B = B / (np.linalg.det(C))
    
    # TODO this is cheating, we should index the faces correctly, dunno if this will work!
    tet_volume = 1 / 6. * np.abs(np.linalg.det(C))
    K_loc_unscaled = np.dot(np.dot(B.T, D.T), B)
    K_l = K_loc_unscaled * tet_volume
    #print("tet vvolume: ", tet_volume)

    for i in range(4):
        for j in range(4):
            K_g.append([int(3 * el_nodes[i] + 0), int(3 * el_nodes[j] + 0), K_l[3 * i + 0, 3 * j + 0]])
            K_g.append([int(3 * el_nodes[i] + 0), int(3 * el_nodes[j] + 1), K_l[3 * i + 0, 3 * j + 1]])
            K_g.append([int(3 * el_nodes[i] + 0), int(3 * el_nodes[j] + 2), K_l[3 * i + 0, 3 * j + 2]])

            K_g.append([int(3 * el_nodes[i] + 1), int(3 * el_nodes[j] + 0), K_l[3 * i + 1, 3 * j + 0]])
            K_g.append([int(3 * el_nodes[i] + 1), int(3 * el_nodes[j] + 1), K_l[3 * i + 1, 3 * j + 1]])
            K_g.append([int(3 * el_nodes[i] + 1), int(3 * el_nodes[j] + 2), K_l[3 * i + 1, 3 * j + 2]])

            K_g.append([int(3 * el_nodes[i] + 2), int(3 * el_nodes[j] + 0), K_l[3 * i + 2, 3 * j + 0]])
            K_g.append([int(3 * el_nodes[i] + 2), int(3 * el_nodes[j] + 1), K_l[3 * i + 2, 3 * j + 1]])
            K_g.append([int(3 * el_nodes[i] + 2), int(3 * el_nodes[j] + 2), K_l[3 * i + 2, 3 * j + 2]])
    
    return B

def apply_constraints(K_g_sp, constraints):
    # constraint is of type [(node, [x, y, z]), ...] where xyz are 1 for constraint, 0 for no constraint
    for constraint in constraints:
        for i in range(len(constraint[1])):
            if constraint[1][i] == 1:
                idx = constraint[0]*3 + i
                K_g_sp[idx, :] = 0.0
                K_g_sp[:, idx] = 0.0
        for i in range(len(constraint[1])):
            if constraint[1][i] == 1:
                idx = constraint[0]*3 + i
                K_g_sp[idx, idx] = 1.0
                
def get_loads(loads, verts):
    # loads is of tpye [(node, x, y, z)]
    load_vec = np.zeros(len(verts)*3)
    for load in loads:
        node = load[0]
        load_vec[3*node + 0] = load[1]
        load_vec[3*node + 1] = load[2]
        load_vec[3*node + 2] = load[3]
    return load_vec
        
def solve_fem(K_g_sp, loads):
    d = spsolve(K_g_sp, loads)
    return d

def solve_full(elements, verts, poisson, youngs, constraints, loads):
    D = generate_elasticity_mat(youngs, poisson)
    load_arr = get_loads(loads, verts)
    K_g = []
    for el in elements:
        add_local_stiffness(D, K_g, verts, el)
    data = [k[2] for k in K_g]
    rows = [k[0] for k in K_g]
    cols = [k[1] for k in K_g]
    K_g_sp = coo_matrix((data, (rows, cols))).tolil()
    apply_constraints(K_g_sp, constraints)
    displacements = solve_fem(K_g_sp, load_arr)
    return displacements

def extract_tets(cells):
    cell_arr = []
    for i in range(len(cells) // 5):
        start = i*5 + 1
        cell_arr.append(cells[start:start + 4])
    return np.array(cell_arr)


if __name__ == "__main__":
    box = pv.Box((-1.0, 1.0, -5.0, 5.0, -1.0, 1.0))
    tet = tetgen.TetGen(box.triangulate())
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    verts = grid.points
    print("Cells: \n", grid.cells)
    print("Points: \n", verts)
    cell_arr = extract_tets(grid.cells)
    print(cell_arr)
    constraints = [[1, [1, 1, 1]],
                [3, [1, 1, 1]],
                [4, [1, 1, 1]],
                [6, [1, 1, 1]]]

    loads = [[0, 1., 0.0, 0],
            [2, 1., 0.0, 0],
            [5, 1., 0.0, 0],
            [7, 1., 0.0, 0]]
    poisson = 0.3
    youngs = 2000

    displacements = solve_full(cell_arr, verts, poisson, youngs, constraints, loads)
    print(pd.DataFrame(displacements))