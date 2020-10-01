import numpy as np


gridRad = 0.027
OD = 0.028
ID = 0.027
dh = 0.0001
catGap = 0.004
catWid = 0.001
catL = 0.01
T0 = 298.15
L = 0.1


def RasterQuad(r): # sub-function of DiscretePipe and DiscreteCat. Rasterizes a 2D quadrant of radius r pixels
    inner = []
    outer = []
    x, y = r, 0     # midpoint circle algorithm
    while x >= y:
        while x <= np.sqrt(r**2 - y**2) + 0.5:
            inner.append((x, y))
            outer.append((x+1, y))
            y += 1
        x -= 1
    outer.append((x+1, y))
    inner = inner + [(y,x) for (x,y) in inner]
    outer = outer + [(y,x) for (x,y) in outer]
    return inner, outer # inner has the given radius. outer is the shell with radius r+1


def DiscreteCat(OD, ID, L, dh, catGap, catWid, catL, T0): # quarters and discretises the catalyst
    catGrid = np.zeros((gridRad+2, gridRad+2))
    inner, _ = RasterQuad(int(ID/dh/2-1))
    
    for point in inner:
        catGrid[:point[1]+1, point[0]] = 1
    
    for i in range(catGrid.shape[0]):
        if 0 <= (i*dh + catGap/2) % (catGap+catWid) < catGap:
            for j in range(catGrid.shape[1]):
                if 0 <= (j*dh + catGap/2) % (catGap+catWid) < catGap:
                    catGrid[i,j] = 0
    
    catNodes2D = np.argwhere(catGrid == 1)
    catNodes3D = np.vstack((np.hstack((catNodes2D, np.full((len(catNodes2D),1), z))) for z in range(int(catL/dh))))
    catNodes3D[:,2] += int((L-catL)/dh)
    catNodesT = dict(zip([tuple(row) for row in catNodes3D.tolist()], [T0 for i in range(len(catNodes3D))]))

    return catNodesT # a dictionary with spatial coordinates as keys and temperatures as values


def getNeighbours_3D(point):
    neighbours = np.array([
        (point[0], point[1]+1, point[2]),
        (point[0], point[1]-1, point[2]),
        (point[0]+1, point[1], point[2]),
        (point[0]-1, point[1], point[2]),
        (point[0], point[1], point[2]+1),  # Cell above
        (point[0], point[1], point[2]-1)  # Cell below
    ])
    
    return neighbours


def run_cata(grid, T_g, k_s, rho_s, T_Tot, dh, dt):

    for x in np.arange(1, int(ID/dh)):
            for y in np.arange(1, int(catL/dh)):
                for z in np.arange(1, int(ID/dh)):
                    if grid[(x,y,z)] == T_g:
                        pass
                    else:
                        grid_last = T_Tot[-1]
                        c_s = 450 + 0.28 * (grid[(x,y,z)]-273.15)
                        alpha = k_s / (rho_s*c_s)
                        neighbours = getNeighbours_3D((x, y, z))
                        coeff = alpha / (dh**2)
                        sum_T = 0
                        for pair in neighbours:
                            sum_T += grid_last[pair[0], pair[1]]

                        grid[(x, y, z)] = dt * (coeff * (sum_T - 6*grid_last[(x, y, z)])) + grid_last[(x, y, z)]
    
    return grid


def main():
    grid = DiscreteCat(OD, ID, L, dh, catGap, catWid, catL, T0)
    grid_T = []
    grid_T.append(grid)
    for t in time:
        grid = run_cata(grid, T_g, k_s, rho_s, T_Tot, dh, dt)
        grid_T.append(grid)
    
    return grid_T
