#!/usr/bin/env python
from tecio import TecplotFile
import numpy as np
import os

# open file but don't load the data immediately
is_github = os.getenv("GITHUB_ACTIONS") == "true"
test_file = 'test.dat'

tec = TecplotFile(test_file, read_data=False, tempFile='test.plt')

# show all properties
print(tec)

# print all variables
print("Variables in file:")
for v in tec.variables:
    print(" %s" % v)

# get variables for first zone
x = tec.get_data(0, var_prefix='X')
print(x)

# print geometries
for geom in tec.geometries:
    print('Coord %f, %f' % (geom.x0, geom.y0))
# print text
for t in tec.texts:
    print(t.text)


tec = TecplotFile("fem.plt")

x = tec.get_data(0, 0)
y = tec.get_data(0, 1)
v = tec.get_data(0, 2)

connect = tec.get_connect(0)

if not is_github:
    import pyvista as pv
    grid = pv.UnstructuredGrid(
        {
            pv.CellType.TRIANGLE: np.array(connect)
        },
        np.column_stack((x, y, np.zeros_like(x)))
    )
    grid[tec.variables[2]] = v
    pl = pv.Plotter()
    _ = pl.add_mesh(grid, show_edges=True, scalars=tec.variables[2])
    pl.view_xy()
    pl.show()
