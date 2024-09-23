#!/usr/bin/env python
import numpy as np
from construct import GreedyRange
from construct import Float32l as Float
from tecio import TecHeader, gen_data_struct, gen_zone_struct
from tecio import ZoneType
from tecio import TecplotMarker, TecDatasetAux

tec = dict(
    variables=["X", "Y", "P"],
    zones=[
        dict(
            zone_type=ZoneType.FETRIANGLE,
            num_pts=6,
            num_elems=4,
            aux_vars=[
                {
                    "name": "aux_v1",
                    "value": 1.1,
                },
                {
                    "name": "aux_v2",
                    "value": 1.2,
                },
            ],
        )
    ],
    dataset_aux=[
        {
            "name": "var1",
            "value": "a random test",
        },
        {
            "name": "another_one",
            "value": 1.0e-5,
        },
    ],
    data=[
        dict(
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
    ],
)

for iz, data in enumerate(tec['data']):
    # ensure connect id starts from 0
    for icell in range(len(data['connect'])):
        data["connect"][icell] = [j - 1 for j in data["connect"][icell]]
    # calculate min_max
    data["min_max"] = [[np.min(d), np.max(d)] for d in data["data"]]


with open("fem.plt", "wb") as f:
    TecHeader.build_stream(tec, f)
    GreedyRange(gen_zone_struct(len(tec["variables"]))).build_stream(tec["zones"], f)
    GreedyRange(TecDatasetAux).build_stream(tec.get('dataset_aux', []), f)


    f.write(Float.build(TecplotMarker.EOH))
    for z, d in zip(tec['zones'], tec['data']):
        gen_data_struct(tec["variables"], z).build_stream(d, f)
