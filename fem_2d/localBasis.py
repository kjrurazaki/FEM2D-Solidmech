import numpy as np

def localBasis(Nelem, triang, coord):
    '''
    Build local P1 basis functions on triangles
    only the coefficients (b,c) multiplying x and y are built
    phi(x, y) = a + bx + cy
    Built based on page 84 from notes
    b_i = y_j - y_k (i -> j -> k -> i) - convection (e.g. b_2 = y_3 - y_1)
    c_i = x_k - x_j (i -> j -> k -> i)
    '''

    Bloc = np.zeros((Nelem, 3))
    Cloc = np.zeros((Nelem, 3))
    Area = np.zeros((Nelem, 1))

    for iel in range(0, Nelem):
        nodes = triang[iel,:]
        p1 = coord[nodes[0] - 1,:].reshape(1,-1) # coordinate of first node
        p2 = coord[nodes[1] - 1,:].reshape(1,-1) # coordinate of second node
        p3 = coord[nodes[2] - 1,:].reshape(1,-1) # coordinate of third node
        A = np.concatenate((np.ones((3, 1)), (np.concatenate((p1, p2, p3), axis = 0))), axis = 1)
        DetA = np.linalg.det(A)
        Area[iel] = abs(DetA/2)
        
        for inod in range(1,4):
            n1 = mod_n(inod + 1,3)
            n2 = mod_n(inod + 2,3) 
            Bloc[iel, inod - 1] = (coord[nodes[n1 - 1] - 1, 1] - coord[nodes[n2 - 1] - 1, 1])/DetA
            Cloc[iel, inod - 1] = (coord[nodes[n2 - 1] - 1, 0] - coord[nodes[n1 - 1] - 1, 0])/DetA

    return Bloc, Cloc, Area

def mod_n(i, j):
    '''
    define the recirculation  (i -> j -> k -> i) - convection (e.g. b_2 = y_3 - y_1)
    '''
    aux = np.mod(i,j)
    if aux == 0:
        remainder = j
    else:
        remainder = aux
    return remainder