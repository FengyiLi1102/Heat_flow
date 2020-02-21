import numpy as np


def getNeighbours(point):

    neighours = np.array(
        [(point[0], point[1]+1),
         (point[0], point[1]-1),
         (point[0]+1, point[1]),
         (point[0]-1, point[1]),
         (point[0], point[1]),  # Cell above
         (point[0], point[1])]  # Cell below
    )

    return neighours