#!/usr/bin/env python
import numpy as np
from construct import GreedyRange
from construct import Float32l as Float
from tecio import TecHeader, gen_data_struct, gen_zone_struct
from tecio import ZoneType

tec = dict(
    variables=["X", "Y", "P"],
    zones=[dict(zone_type=ZoneType.FETRIANGLE, num_pts=6, num_elems=4)],
)

zone_data = dict(
    connect=[
        [1, 2, 4],
        [2, 5, 4],
        [3, 5, 2],
        [5, 6, 4],
    ],
    data=[
        np.array([-1.0, 0.0, 1.0, -0.5, 0.5, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.8, 0.8, 1.6]),
        np.array([100.0, 125.0, 150.0, 150.0, 175.0, 200.0]),
    ],
)

for i in range(len(zone_data["connect"])):
    zone_data["connect"][i] = [j - 1 for j in zone_data["connect"][i]]

zone_data["min_max"] = [[np.min(d), np.max(d)] for d in zone_data["data"]]
with open("fem.plt", "wb") as f:
    TecHeader.build_stream(tec, f)
    GreedyRange(gen_zone_struct(len(tec["variables"]))).build_stream(tec["zones"], f)
    f.write(Float.build(357.0))
    gen_data_struct(tec["variables"], tec["zones"][0]).build_stream(zone_data, f)
