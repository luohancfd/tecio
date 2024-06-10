#!/usr/bin/env python3
'''Python script to load Tecplot binary file or gzipped binary file'''
__author__ = "Han Luo"
__copyright__ = "Copyright 2024, Han Luo"
__license__ = "GPL"
__repo__ = "https://github.com/luohancfd/tecio"

import numpy as np
import enum
import construct  # work with v2.10.67
import functools
import operator
import gzip
import logging
import sys
import re
import os
from glob import glob
import tempfile
from distutils.spawn import find_executable
from construct import Int8ul as Byte
from construct import Int16sl as Short
from construct import Int32sl as Int
from construct import Float32l as Float
from construct import Float64l as Double
from construct import Bit
from construct import Struct, Array, Const, GreedyRange
from construct import Index, Computed, Default, Rebuild
from construct import Check, Tell, Peek, singleton
from construct import this, len_
from construct import If, Switch, StopIf, Error
from io import SEEK_CUR
from typing import Union
from subprocess import Popen, PIPE
from sys import platform

__all__ = [
    'TecplotFile',
    'TecplotMatrix',
    'TecHeader',
    'TecGeom',
    'TecDatasetAux',
    'TecStr',
    'ZoneType',
    'VarLoc',
    'VarType',
    'gen_data_struct',
    'gen_zone_struct'
]

Logger = logging.getLogger('tecio')
Logger.setLevel(logging.WARNING)
Handler = logging.StreamHandler(sys.stdout)
Handler.setLevel(logging.WARNING)
Logger.addHandler(Handler)


def find_preplot():
    preplot = find_executable('preplot')
    if preplot is None and platform == 'linux':
        candidates = [
            '/usr/local/tecplot/360ex_2022r2/bin/preplot',
            '/sw/tecplot/360ex_2022r2/bin/preplot',
            '/sw/tecplot/360ex_2022r1/bin/preplot',
        ]

        for i in candidates:
            if os.path.isfile(i):
                preplot = i
                break
        tmp = glob('/usr/local/tecplot/*/bin/preplot')
        if len(tmp) > 0:
            preplot = tmp[0]

    if preplot is None:
        Logger.warn('preplot is not found and you can not read ASCII data')
    return preplot


PREPLOT = find_preplot()


class ZoneType(enum.IntEnum):
    ORDERED = 0
    FELINESEG = 1
    FETRIANGLE = 2
    FEQUADRILATERAL = 3
    FETETRAHEDRON = 4
    FEBRICK = 5
    FEPOLYGON = 6
    FEPOLYHEDRON = 7

    @property
    def npoints(self):
        d = [0, 2, 3, 4, 4, 8, -1, -1]
        return d[self.value]


class CoordSys(enum.IntEnum):
    Grid = 0
    Frame = 1
    FrameOffset = 2
    OldWindow = 3
    Grid3D = 4


class GeomScope(enum.IntEnum):
    Global = 0
    Local = 1


class DrawOrder(enum.IntEnum):
    AfterData = 0
    BeforeData = 1


class GeomType(enum.IntEnum):
    Line = 0
    Rectangle = 1
    Square = 2
    Circle = 3
    Ellipse = 4


class LinePattern(enum.IntEnum):
    Solid = 0
    Dashed = 1
    DashDot = 2
    DashDotDot = 3
    Dotted = 4
    LongDash = 5


class ArrowheadStyle(enum.IntEnum):
    Plain = 0
    Filled = 1
    Hollow = 2


class ArrowheadAttachment(enum.IntEnum):
    None_ = 0
    AtBeginning = 1
    AtEnd = 2
    AtBothEnds = 3


class PolyLineDataType(enum.IntEnum):
    Float = 1
    Double = 2


class GeomClipping(enum.IntEnum):
    ClipToAxes = 0
    ClipToViewport = 1
    ClipToFrame = 2


class Color(enum.IntEnum):
    Black = 0
    Red = 1
    Green = 2
    Blue = 3
    Cyan = 4
    Yellow = 5
    Purple = 6
    White = 7

    Custom1 = 8
    Custom2 = 9
    Custom3 = 10
    Custom4 = 11
    Custom5 = 12
    Custom6 = 13
    Custom7 = 14
    Custom8 = 15
    Custom9 = 16
    Custom10 = 17
    Custom11 = 18
    Custom12 = 19
    Custom13 = 20
    Custom14 = 21
    Custom15 = 22
    Custom16 = 23
    Custom17 = 24
    Custom18 = 25
    Custom19 = 26
    Custom20 = 27
    Custom21 = 28
    Custom22 = 29
    Custom23 = 30
    Custom24 = 31
    Custom25 = 32
    Custom26 = 33
    Custom27 = 34
    Custom28 = 35
    Custom29 = 36
    Custom30 = 37
    Custom31 = 38
    Custom32 = 39
    Custom33 = 40
    Custom34 = 41
    Custom35 = 42
    Custom36 = 43
    Custom37 = 44
    Custom38 = 45
    Custom39 = 46
    Custom40 = 47
    Custom41 = 48
    Custom42 = 49
    Custom43 = 50
    Custom44 = 51
    Custom45 = 52
    Custom46 = 53
    Custom47 = 54
    Custom48 = 55
    Custom49 = 56
    Custom50 = 57
    Custom51 = 58
    Custom52 = 59
    Custom53 = 60
    Custom54 = 61
    Custom55 = 62
    Custom56 = 63

    MultiColor = -1
    NoColor = -2
    MultiColor2 = -3
    MultiColor3 = -4
    MultiColor4 = -5

    RGBColor = -6

    MultiColor5 = -7
    MultiColor6 = -8
    MultiColor7 = -9
    MultiColor8 = -10

    InvalidColor = -255

    # These names are close, though not always
    # the mathematically closest, to the XKCD colors
    Grey = Custom1
    LightGrey = Custom2
    Orange = Custom3
    LimeGreen = Custom4
    AquaGreen = Custom5
    BrightBlue = Custom6
    Violet = Custom7
    HotPink = Custom8
    Mahogany = Custom9
    LightSalmon = Custom10
    LightOrange = Custom11
    LightGreen = Custom12
    SeaGreen = Custom13
    WarmBlue = Custom14
    LightPurple = Custom15
    Coral = Custom16
    Olive = Custom17
    Creme = Custom18
    Lemon = Custom19
    Spearmint = Custom20
    BrightCyan = Custom21
    BluePurple = Custom22
    LightMagenta = Custom23
    RedOrange = Custom24
    Forest = Custom25
    LightMintGreen = Custom26
    YellowGreen = Custom27
    Emerald = Custom28
    SkyBlue = Custom29
    Indigo = Custom30
    BubbleGum = Custom31
    Cinnamon = Custom32
    DarkTurquoise = Custom33
    LightCyan = Custom34
    LemonGreen = Custom35
    Chartreuse = Custom36
    Azure = Custom37
    RoyalBlue = Custom38
    BrightPink = Custom39
    DeepRed = Custom40
    DarkBlue = Custom41
    LightBlue = Custom42
    MustardGreen = Custom43
    LeafGreen = Custom44
    Turquoise = Custom45
    OceanBlue = Custom46
    Magenta = Custom47
    Raspberry = Custom48
    DeepViolet = Custom49
    Lilac = Custom50
    Khaki = Custom51
    Fern = Custom52
    GreyTeal = Custom53
    DuskyBlue = Custom54
    MediumPurple = Custom55
    LightMaroon = Custom56


class EnumInt(construct.Adapter):
    def __init__(self, penum: enum.IntEnum, subcon=Int):
        super().__init__(subcon)
        if isinstance(penum, enum.IntEnum):
            raise ValueError("penum should be an IntEnum")
        self.penum = penum

    def _encode(self, obj, context, path):
        if isinstance(obj, int):
            return obj
        return int(obj)

    def _decode(self, obj, context, path):
        return self.penum(obj)


class TecplotString(construct.Adapter):
    """construct type for Tecplot string
    """

    def __init__(self, encoder=construct.Int32ul):
        self.encoder = encoder
        subcon = construct.NullTerminated(
            construct.GreedyBytes,
            term=self.encoder.build(0)
        )
        super().__init__(subcon)
        self.nbyte = self.encoder.sizeof()

    def _decode(self, obj, context, path):
        return "".join([chr(self.encoder.parse(obj[i:])) for i in range(0, len(obj), self.nbyte)])

    def _encode(self, obj, context, path):
        if obj == "":
            return b""
        return b"".join([self.encoder.build(ord(i)) for i in obj])

    def _emitparse(self, code):
        raise NotImplementedError

    def _emitbuild(self, code):
        raise NotImplementedError
        # This is not a valid implementation. obj.encode() should be inserted into subcon
        # return f"({self.subcon._compilebuild(code)}).encode({repr(self.encoding)})"

    def _emitfulltype(self, ksy, bitwise):
        return dict(type="strz", encoding="ascii")


TecStr = TecplotString()


def squeeze_ijk(ijk, d_ijk=None):
    if d_ijk is None:
        d_ijk = ijk
    for i in range(len(ijk) - 1, -1, -1):
        if d_ijk[i] != 1:
            return ijk[:i + 1] if i > 0 else [ijk[0]]
    return ijk[:]


class VarLoc(enum.IntEnum):
    Node = 0,
    CellCentered = 1


class VarType(enum.IntEnum):
    Float = 1,
    Double = 2,
    LongInt = 3,
    ShortInt = 4,
    Byte = 5,
    Bit = 6


@singleton
class Bool(construct.Adapter):
    def __init__(self):
        super().__init__(Int)

    def _decode(self, obj, context, path):
        return bool(obj)

    def _encode(self, obj, context, path):
        return obj != 0


VarTypeConstruct = [None, Float, Double, Int, Short, Byte, Bit]


class TecplotMatrix(construct.Construct):
    """Matrix"""

    def __init__(self, subcon: Union[construct.FormatField, None], cell_centered=False, ijk: Union[list[int], None] = None, discard=False, zone_type: ZoneType = ZoneType.ORDERED):
        """_summary_
        For zone_type == ZoneType.ORDERED:
            ijk = [i, j, k]
        For zone_type != ZoneType.ORDERED:
            ijk = [npoints] or [nelems]

        Args:
            subcon (construct.FormatField): one of the VarTypeConstruct
            cell_centered (bool, optional): whether the data is cell centered. Defaults to False.
            ijk (list[int] | None, optional): check the above document. Defaults to None.
            discard (bool, optional): whether to load the data instantaneously. Defaults to False.
            zone_type (ZoneType, optional): type of the zone. Defaults to ZoneType.ORDERED.
        """

        '''
        For zone_type == ZoneType.ORDERED:
            ijk = [i, j, k]
        For zone_type != ZoneType.ORDERED:
            ijk = [npoints] or [nelems]
        '''
        super().__init__()
        self.name = "TecplotMatrix"
        self.subcon = subcon
        self.dtype = None
        if not callable(subcon):
            self.dtype = np.dtype(subcon.fmtstr)
        self.discard = discard
        self.cell_centered = cell_centered
        self.zone_type = zone_type
        self.ijk = ijk
        if isinstance(ijk, int):
            self.ijk = [ijk]
        self.order = 'F'

    @staticmethod
    def ijk_to_fshape(ijk, cell_centered=False, zone_type=ZoneType.ORDERED):
        '''
        Convert ijk to the index in file storage
        '''
        if zone_type == ZoneType.ORDERED:
            if cell_centered:
                if ijk[2] > 1:
                    # check manual Binary Data File Format > Note 5
                    return [ijk[0], ijk[1], ijk[2] - 1]
                elif ijk[1] > 1:
                    return [ijk[0], ijk[1] - 1]
                else:
                    return [ijk[0] - 1]
            return ijk[:]
        else:
            return [ijk[0]]

    @staticmethod
    def ijk_to_mshape(ijk, cell_centered=False, zone_type=ZoneType.ORDERED):
        '''
        Convert ijk to a numpy shape
        '''
        if zone_type == ZoneType.ORDERED:
            if cell_centered:
                t = []
                for i in range(3):
                    if ijk[i] > 1:
                        t.append(ijk[i])
                    else:
                        return t
                return t
            return ijk[:]
        else:
            return [ijk[0]]

    @staticmethod
    def mshape_to_ijk(mshape, cell_centered=False, zone_type=ZoneType.ORDERED):
        t = [i for i in mshape]
        if len(t) > 3:
            raise ValueError("Dimension should be less than 3")
        if zone_type == ZoneType.ORDERED:
            if cell_centered:
                t = [i + 1 for i in t]
            for i in range(len(t), 3):
                t.append(1)
            return t
        else:
            return [t[0]]

    @staticmethod
    def fdata_to_mdata(data: np.ndarray, ijk, cell_centered=False, zone_type=ZoneType.ORDERED):
        if zone_type == ZoneType.ORDERED:
            if cell_centered:
                if ijk[2] > 1:
                    return data[:-1, :-1]
                elif ijk[1] > 1:
                    return data[:-1, :]
                else:
                    return data
        return data

    @staticmethod
    def mdata_to_fdata(data: np.ndarray, cell_centered=False, zone_type=ZoneType.ORDERED):
        if zone_type == ZoneType.ORDERED:
            if cell_centered:
                if data.ndim == 1:
                    return data.copy()
                elif data.ndim == 2:
                    return np.vstack((data, np.zeros(data.shape[1], dtype=data.dtype)))
                elif data.ndim == 3:
                    data = np.append(
                        data,
                        np.zeros_like(data[1, :, :]).reshape((1, data.shape[1], data.shape[2])),
                        axis=0
                    )
                    data = np.append(
                        data,
                        np.zeros_like(data[:, 1, :]).reshape((data.shape[0], 1, data.shape[2])),
                        axis=1
                    )
                    return data
                else:
                    raise ValueError("cell centered only accepts dim <= 3")
            return data.copy()
        else:
            return data.copy()

    def _parse(self, stream, context, path):
        ijk = self.ijk
        dtype = self.dtype

        subcon = self.subcon
        if 'subcon' in context:
            subcon = context.subcon
        if callable(subcon):
            subcon = construct.evaluate(subcon, context)
            dtype = np.dtype(subcon.fmtstr)

        cell_centered = self.cell_centered
        if 'cell_centered' in context:
            cell_centered = context.cell_centered
        cell_centered = bool(construct.evaluate(cell_centered, context))
        context._cell_centered = cell_centered

        zone_type = self.zone_type
        if 'zone_type' in context:
            zone_type = context.zone_type
        zone_type = construct.evaluate(zone_type, context)

        discard = self.discard
        if 'discard' in context:
            discard = context.discard
        discard = construct.evaluate(discard, context)

        ijk = construct.evaluate(ijk, context)
        if isinstance(ijk, int):
            ijk = [ijk]
        if zone_type == ZoneType.ORDERED:
            if len(ijk) < 3:
                for i in range(len(ijk), 3):
                    ijk.append(1)

        del context._cell_centered
        fshape = self.ijk_to_fshape(ijk, cell_centered, zone_type)
        length = functools.reduce(operator.mul, fshape) * dtype.itemsize
        if length <= 0:
            raise construct.RangeError("invalid length")
        context._offset = construct.stream_tell(stream, path)
        if 'data_offset' in context:
            # inject data_offset if it exists
            context.data_offset[context._index] = context._offset
        del context._offset
        if discard:
            construct.stream_seek(stream, length, SEEK_CUR, path)
            return None
        else:
            obj = np.frombuffer(
                construct.stream_read(stream, length, path),
                dtype=dtype,
            ).reshape(fshape, order=self.order)
            mdata = self.fdata_to_mdata(obj, ijk, cell_centered, zone_type)
            if path[-7:] == "min_max":
                return np.reshape(mdata, (2, -1))
            if zone_type == ZoneType.ORDERED:
                npshape = squeeze_ijk(mdata.shape, fshape)
                return mdata.reshape(npshape)
            else:
                return mdata

    def _build(self, obj: np.ndarray, stream, context, path):
        subcon = self.subcon
        if callable(subcon):
            subcon = subcon(context)
        dtype = np.dtype(subcon.fmtstr)
        obj = np.array(obj).astype(dtype)

        cell_centered = self.cell_centered
        if callable(cell_centered):
            cell_centered = cell_centered(context)
        if not isinstance(cell_centered, bool):
            raise ValueError("cell_centered should be a bool")
        context._cell_centered = cell_centered

        zone_type = self.zone_type
        if callable(zone_type):
            zone_type = zone_type(context)
        if not isinstance(zone_type, ZoneType):
            raise ValueError("zone_type should be a ZoneType")

        discard = self.discard
        if 'discard' in context:
            discard = context.discard

        del context._cell_centered

        mshape = list(obj.shape)
        ijk = self.mshape_to_ijk(mshape, cell_centered, zone_type)
        fshape = self.ijk_to_fshape(ijk, cell_centered, zone_type)
        length = functools.reduce(operator.mul, fshape) * dtype.itemsize
        _ = construct.stream_tell(stream, path)
        if discard:
            buf = b'\xff' * length
        else:
            buf = self.mdata_to_fdata(obj, cell_centered).tobytes(order='F')
        construct.stream_write(stream, buf, length, path)
        return buf

    def _sizeof(self, context, path):
        raise construct.SizeofError(path=path)

    def _emitfulltype(self, ksy, bitwise):
        return dict(type=self.subcon._compileprimitivetype(ksy, bitwise), repeat="eos")


def calculate_data_length(zone_type: ZoneType, var_loc: VarLoc, ijk: list[int], num_pts: int, num_elems: int):
    if zone_type == ZoneType.ORDERED:
        if var_loc == VarLoc.Node:
            return ijk[0] * ijk[1] * ijk[2]
        else:
            return ijk[0] * (1 if ijk[1] <= 1 else (ijk[1] if ijk[2] <= 1 else ijk[1] * ijk[2]))
    else:
        return num_pts if var_loc == VarLoc.Node else num_elems


def calculate_nvar_zone(has_passive_var: bool, passive_var: list[int], has_shared_var: bool, shared_var: list[int], nvar: int):
    return len([1 for i, j in zip(passive_var if has_passive_var else [
        0] * nvar, shared_var if has_shared_var else [-1] * nvar) if i == 0 and j == -1])


def find_var_zone(has_passive_var: bool, passive_var: list[int], has_shared_var: bool, shared_var: list[int], nvar: int):
    return [i for i, (j, k) in enumerate(zip(passive_var if has_passive_var else [
        0] * nvar, shared_var if has_shared_var else [-1] * nvar))
        if j == 0 and k == -1]


def aux_parse(x, lst, ctx):
    if x == '':
        lst.pop()
        return True
    return False


# ===================== Tecplot Sections ======================

TecHeader = Struct(
    Const(b"#!TDV112"),  # Magic number
    Const(Int.build(1)),  # Integer value of 1
    Const(Int.build(0)),  # FileType: 0 = FULL
    "title" / Default(TecStr, ""),  # File Title
    "nvar" / Rebuild(Int, len_(this.variables)),  # Number of variables
    "variables" / Array(this.nvar, TecStr),  # Variable names
)


class TypeTecplotAuxVar:
    name: str
    value: str


TecDatasetAux = Struct(
    Const(Float.build(799.0)),
    "name" / TecStr,  # Variable names
    Const(Int.build(0)),
    "value" / TecStr,
)


class TypeTecplotAuxVar2:
    name: str
    variable: int
    value: str


TecVariableAux = Struct(
    Const(Float.build(899.0)),
    "variable" / Int,
    "name" / TecStr,  # Variable name,
    Const(Int.build(0)),
    "value" / TecStr
)


class TypeTecplotGeomArrow:
    style: ArrowheadStyle
    attachment: ArrowheadAttachment
    size: float
    angle: float
    macro_function: str


class TypeTecplotGeomPolylineOne:
    num_points: int
    x: list[float]
    y: list[float]
    z: list[float]


class TypeTecplotGeomPolyline:
    num_polylines: int
    lines: list[TypeTecplotGeomPolylineOne]


class TypeTecplotGeomRectangle:
    width: float
    height: float


class TypeTecplotGeomCircle:
    radius: float


class TypeTecplotGeomSquare:
    width: float


class TypeTecplotGeomEllipse:
    radii_x: float
    radii_y: float


class TypeTecplotGeom:
    igeom: int
    coord_sys: CoordSys
    scope: GeomScope
    draw_order: DrawOrder
    x0: float
    y0: float
    z0: float
    zone: int
    color: Color
    fill_color: Color
    is_filled: int
    type: int
    line_pattern: int
    pattern_length: float
    line_thickness: float
    num_ellipse_pts: int
    arrowhead: TypeTecplotGeomArrow
    geom: Union[TypeTecplotGeomRectangle, TypeTecplotGeomCircle, TypeTecplotGeomSquare, TypeTecplotGeomPolyline, TypeTecplotGeomEllipse]


TecGeom = Struct(
    Const(Float.build(399.0)),
    "igeom" / Index,
    "coord_sys" / Default(EnumInt(CoordSys), CoordSys.Grid),
    "scope" / Default(EnumInt(GeomScope), GeomScope.Global),
    "draw_order" / Default(EnumInt(DrawOrder), DrawOrder.AfterData),
    "x0" / Default(Double, 0.0),
    "y0" / Default(Double, 0.0),
    "z0" / Default(Double, 0.0),
    "zone" / Default(Int, 0),
    "color" / Default(EnumInt(Color), Color.Black),
    "fill_color" / Default(EnumInt(Color), Color.Black),
    "is_filled" / Default(Bool, 0),
    "type" / Default(EnumInt(GeomType), GeomType.Line),
    "line_pattern" / Default(EnumInt(LinePattern), LinePattern.Solid),
    "pattern_length" / Default(Double, 0),  # GUI's value divided by 100
    "line_thickness" / Default(Double, 0.04),  # GUI's value divided by 100
    "num_ellipse_pts" / Default(Int, 1),
    "arrowhead" / Struct(
        "style" / Default(EnumInt(ArrowheadStyle), ArrowheadStyle.Plain),
        "attachment" / Default(EnumInt(ArrowheadAttachment), ArrowheadAttachment.None_),
        "size" / Default(Double, 5.0),
        "angle" / Default(Double, 15.0),  # GUI's value converted to radian
        "macro_function" / Default(TecStr, ""),
    ),
    "data_type" / Default(EnumInt(PolyLineDataType), PolyLineDataType.Float),  # 1=Float, 2=Double
    "clipping" / Default(EnumInt(GeomClipping), GeomClipping.ClipToAxes),
    "__integrity__" / Check(this.type == 0),
    "geom" / Switch(
        this.type,
        {
            GeomType.Line: Struct(
                "num_polylines" / Rebuild(Int, len_(this.lines)),
                "lines" / Array(
                    this.num_polylines,
                    Struct(
                        "num_points" / Rebuild(Int, len_(this.x)),
                        "x" / Array(this.num_points, Switch(this._._.data_type, {VarType.Float: Float, VarType.Double: Double})),
                        "y" / Array(this.num_points, Switch(this._._.data_type, {VarType.Float: Float, VarType.Double: Double})),
                        StopIf(this._._.coord_sys != CoordSys.Grid3D),
                        "z" / Array(this.num_points, Switch(this._._.data_type, {VarType.Float: Float, VarType.Double: Double})),
                    )
                )
            ),
            GeomType.Rectangle: Struct(
                "width" / Float,
                "height" / Float
            ),
            GeomType.Circle: Struct(
                "radius" / Float
            ),
            GeomType.Square: Struct(
                "width" / Float
            ),
            GeomType.Ellipse: Struct(
                "radii_x" / Float,
                "radii_y" / Float
            )
        },
        default=Error
    )
)


class Font(enum.IntEnum):
    Helvetica = 0
    HelveticaBold = 1
    Greek = 2
    Math = 3
    UserDefined = 4
    Times = 5
    TimesItalic = 6
    TimesBold = 7
    TimesItalicBold = 8
    Courier = 9
    CourierBold = 10
    Extended = 11
    HelveticaItalic = 12
    HelveticaItalicBold = 13
    CourierItalic = 14
    CourierItalicBold = 15


class Units(enum.IntEnum):
    Grid = 0
    Frame = 1
    Point = 2
    Screen = 3
    AxisPercentage = 4


class TextBox(enum.IntEnum):
    None_ = 0
    Filled = 1
    Hollow = 2


class TextAnchor(enum.IntEnum):
    Left = 0
    Center = 1
    Right = 2
    MidLeft = 3
    MidCenter = 4
    MidRight = 5
    HeadLeft = 6
    HeadCenter = 7
    HeadRight = 8
    OnSide = 9


class TypeTecplotText:
    x0: float
    y0: float
    z0: float
    text: str
    height_unit: Units
    height: float
    color: Color
    coord_sys: CoordSys
    scope: GeomScope
    font: Font
    box_type: TextBox
    box_margin: float
    box_linewidth: float
    box_color: Color
    box_fill_color: Color
    angle: float
    line_spacing: float
    fill_color: Color
    anchor: TextAnchor
    zone: int
    macro: str
    clipping: GeomClipping


TecText = Struct(
    Const(Float.build(499.0)),
    "coord_sys" / Default(EnumInt(CoordSys), CoordSys.Grid),
    "scope" / Default(EnumInt(GeomScope), GeomScope.Global),
    "x0" / Default(Double, 0.0),
    "y0" / Default(Double, 0.0),
    "z0" / Default(Double, 0.0),
    "font" / Default(EnumInt(Font), Font.Helvetica),
    "height_unit" / Default(EnumInt(Units), Units.Frame),
    "height" / Default(Double, 0.04),  # ASCII/GUI value divided by 100
    "box_type" / Default(EnumInt(TextBox), TextBox.None_),
    "box_margin" / Default(Double, 0.0),
    "box_linewidth" / Default(Double, 0.04),
    "box_color" / Default(EnumInt(Color), Color.Black),
    "box_fill_color" / Default(EnumInt(Color), Color.NoColor),
    "angle" / Default(Double, 0.0),
    "line_spacing" / Default(Double, 1),
    "anchor" / Default(EnumInt(TextAnchor), TextAnchor.Left),
    "zone" / Default(Int, 0),
    "color" / Default(EnumInt(Color), Color.Black),
    "macro" / Default(TecStr, ""),
    "clipping" / Default(EnumInt(GeomClipping), GeomClipping.ClipToAxes),
    "text" / TecStr
)


class TypeTecplotZone:
    offset_start: int
    izone: int
    title: str
    time_strand: int
    solution_time: float
    zone_type: ZoneType
    has_var_loc: int
    var_loc: int
    ijk: Union[list[int], None]
    num_pts: Union[int, None]
    num_faces: Union[int, None]
    face_nodes: Union[int, None]
    boundary_faces: Union[int, None]
    boundary_connections: Union[int, None]
    num_elems: Union[int, None]
    cell_dim: Union[list[int], None]
    aux_vars: list[TypeTecplotAuxVar]
    offset_end: int


def gen_zone_struct(nvar: int):
    return Struct(
        "offset_start" / Tell,
        "izone" / Index,
        Const(Float.build(299.0)),  # Zone marker
        "title" / Default(TecStr, "ZONE 001"),  # Zone name
        Const(Int.build(-1)),  # Parent Zone
        "time_strand" / Default(Int, -2),  # StrandID
        "solution_time" / Default(Double, 0.0),  # Solution Time
        Const(Int.build(-1)),  # Default Zone Color
        "zone_type" / Default(EnumInt(ZoneType), ZoneType.ORDERED),  # Zone Type
        "has_var_loc" / Default(Bool, True),  # Var Location = 1
        "var_loc" / Default(If(this.has_var_loc, Array(nvar, EnumInt(VarLoc))), [VarLoc.Node] * nvar),  # Var Loc * nvar_file
        Const(Int.build(0)),  # raw local 1-to-1
        Const(Int.build(0)),  # mcs user-defined face = 0,
        "ijk" / Default(If(this.zone_type == ZoneType.ORDERED, Array(3, Int)), None),
        "num_pts" / Default(If(this.zone_type != ZoneType.ORDERED, Int), None),
        "num_faces" / Default(If(this.zone_type == ZoneType.FEPOLYGON or this.zone_type == ZoneType.FEPOLYHEDRON, Int), None),
        "face_nodes" / Default(If(this.zone_type == ZoneType.FEPOLYGON or this.zone_type == ZoneType.FEPOLYHEDRON, Int), None),
        "boundary_faces" / Default(If(this.zone_type == ZoneType.FEPOLYGON or this.zone_type == ZoneType.FEPOLYHEDRON, Int), None),
        "boundary_connections" / Default(If(this.zone_type == ZoneType.FEPOLYGON or this.zone_type == ZoneType.FEPOLYHEDRON, Int), None),
        "num_elems" / Default(If(this.zone_type != ZoneType.ORDERED, Int), None),
        "cell_dim" / Default(If(this.zone_type != ZoneType.ORDERED, Array(3, Int)), [0, 0, 0]),
        "__integrity__" / Check(lambda this: bool(this.ijk is None) != bool(this.num_elems is None)),
        "aux_vars" / Default(GreedyRange(
            Struct(
                StopIf(Peek(Int) == 0),
                Const(Int.build(1)),
                "name" / TecStr,
                Const(Int.build(0)),
                "value" / TecStr,
            ),
        ), []),
        Const(Int.build(0)),
        "offset_end" / Tell,
    )


class TypeTecplotDataBlock:
    offset_start: int
    izone: int
    data_type: list[VarType]
    has_passive_var: bool
    passive_var: list[int]
    has_shared_var: bool
    shared_var: int
    shared_connectivity: int
    ivar_zone: list[int]
    nvar_zone: int
    min_max: list[list[float]]
    data_offset: list[int]
    data: list[np.ndarray]
    connect: Union[list[list[int]], None]
    offset_end: int


def gen_data_struct(variables: list[str], zone: Union[dict, construct.Container], read_data_=True):
    if isinstance(zone, dict):
        zone = construct.Container(**zone)
    nvar = len(variables)
    tmp = gen_zone_struct(nvar)
    zone = tmp.parse(tmp.build(zone)) # ensure there is no missing values in zone
    return Struct(
        "offset_start" / Tell,
        Const(Float.build(299.0)),
        "data_type" / Default(Array(nvar, EnumInt(VarType)), [VarType.Float] * nvar),
        "has_passive_var" / Default(Bool, True),
        "passive_var" / Default(If(this.has_passive_var, Array(nvar, Int)), [0] * nvar),
        "has_shared_var" / Default(Bool, True),
        "shared_var" / Default(If(this.has_shared_var, Array(nvar, Int)), [-1] * nvar),
        "shared_connectivity" / Default(Int, -1),
        "ivar_zone" / Computed(lambda this: find_var_zone(this.has_passive_var, this.passive_var, this.has_shared_var, this.shared_var, nvar)),
        "nvar_zone" / Computed(lambda this: len(this.ivar_zone)),
        "min_max" / Array(this.nvar_zone, Array(2, Double)),
        "data_offset" / Computed(lambda this: [-1] * this.nvar_zone),
        "data" / Array(
            this.nvar_zone,
            TecplotMatrix(
                lambda this: VarTypeConstruct[this.data_type[this.ivar_zone[this._index]]],
                cell_centered=lambda this: zone.var_loc[this.ivar_zone[this._index]] == VarLoc.CellCentered if 'has_var_loc' in zone and zone.has_var_loc else False,
                ijk=lambda this: zone.ijk if zone.zone_type == ZoneType.ORDERED else [zone.num_pts if not this._cell_centered else zone.num_elems],
                discard=bool(not read_data_),
                zone_type=zone.zone_type
            )
        ),
        # connect starts from 0
        "connect" / Default(
            If(
                lambda this: zone.zone_type != ZoneType.ORDERED and this.shared_connectivity == -1,
                # Byte[lambda this: zone.num_elems * ZoneType(zone.zone_type).npoints * 4],
                Array(lambda this: zone.num_elems, Array(lambda this: ZoneType(zone.zone_type).npoints, Int))

            ),
            None),
        "offset_end" / Tell,
        Check(lambda this: True if zone.zone_type == ZoneType.ORDERED else max([max(i) for i in this.connect]) < zone.num_pts),
    )


class TecplotFile(construct.Container):
    """
    Tecplot Handler
    """
    # from TecHeader
    title: str
    variables: list[str]
    zones: list[TypeTecplotZone]
    data: list[TypeTecplotDataBlock]

    def __init__(self, filePath: str, read_data: bool = True, tempFile: str = ''):
        """Open the tecplot file

        Args:
            filePath (str): path of the file
            read_data (bool, optional): whether to read all data into memory immediately. Defaults to True.
            tempFile (str, optional): temporarily converted binary file if the input file is ASCII data. Defaults to ''.

        Raises:
            RuntimeError: Preplot fail to convert the ASCII data to binary
            ValueError: _description_
            ValueError: _description_
        """
        import time
        super().__init__()
        self.file: str = filePath
        self._read_data: bool = read_data
        self._compressed: bool = False
        self.__open = open

        self.binaryFile: str = ''
        self.__keepTemp: bool = False
        if tempFile:
            self.binaryFile = tempFile
            self.__keepTemp = True

        # from TecHeader
        self.title: str = ''
        self.nvar: int = 0
        self.variables: list[str] = []

        self.has_dataset_aux: bool = False
        self.dataset_aux: list[TypeTecplotAuxVar] = []

        self.has_variable_aux: bool = False
        self.variable_aux: list[TypeTecplotAuxVar2] = []

        self.has_geometry: bool = False
        self.geometries: list[TypeTecplotGeom] = []

        self.has_text: bool = False
        self.texts: list[TypeTecplotText] = []

        self.nzones: int = 0
        self.zones: list[TypeTecplotZone] = []

        self.has_data: bool = False
        self.data: list[TypeTecplotDataBlock] = []

        with open(self.file, 'rb') as f:
            b = f.read(8)
            if b[:2] == b'\x1f\x8b':
                self._compressed = True
                self.__open = gzip.open
            elif b != b'#!TDV112':
                if not self.__keepTemp:
                    with tempfile.NamedTemporaryFile(prefix=os.path.basename(self.file) + '.') as f:
                        self.binaryFile = f.name
                p = Popen([PREPLOT, self.file, self.binaryFile], stdout=PIPE, stderr=PIPE)
                o, _ = p.communicate()
                for i in o.decode().split():
                    if 'Err' in i:
                        raise RuntimeError(f'Preplot: {i}')
                Logger.info(f'Convert {self.file} => {self.binaryFile}')
                self.file = self.binaryFile

        float_peek = Peek(Float)
        start_time = time.time()
        with self.__open(self.file, 'rb') as f:
            self.update(TecHeader.parse_stream(f))
            reachingEOHM = False
            zone_struct = gen_zone_struct(self.nvar)
            while not reachingEOHM:
                marker = float_peek.parse_stream(f)
                if marker == 399.0:
                    Peek(TecGeom).parse_stream(f)
                    self.has_geometry = True
                    self.geometries += GreedyRange(TecGeom).parse_stream(f)
                elif marker == 499.0:
                    Peek(TecText).parse_stream(f)
                    self.has_text = True
                    self.texts += GreedyRange(TecText).parse_stream(f)
                elif marker == 299.0:
                    Peek(zone_struct).parse_stream(f)
                    self.zones += GreedyRange(zone_struct).parse_stream(f)
                elif marker == 799.0:
                    Peek(TecDatasetAux).parse_stream(f)
                    self.has_dataset_aux = True
                    self.dataset_aux += GreedyRange(TecDatasetAux).parse_stream(f)
                elif marker == 357.0:
                    self.has_data = True
                    Const(Float.build(357.0)).parse_stream(f)
                    break
                elif marker == 899.0:
                    Peek(TecVariableAux).parse_stream(f)
                    self.has_variable_aux = True
                    self.variable_aux += GreedyRange(TecVariableAux).parse_stream(f)
                else:
                    raise NotImplementedError("Unknown marker = %d" % marker)

            self.nzones = len(self.zones)
            for iz, z in enumerate(self.zones):
                self.data.append(gen_data_struct(self.variables, z, self._read_data).parse_stream(f))
            if len(self.data) == 0:
                raise ValueError("Fail to parse data")
        Logger.info(f"Finish loading {filePath:s} with {self.nzones} zones in {time.time() - start_time:f}(s)")

    def __del__(self):
        if not self.__keepTemp and self.binaryFile and os.path.isfile(self.binaryFile):
            os.remove(self.binaryFile)

    def get_connect(self, izone: int, start_from=0):
        if self.zones[izone].zone_type == ZoneType.ORDERED:
            return None
        elif self.data[izone].shared_connectivity == -1:
            if start_from != 0:
                return [[j+1 for j in i] for i in self.data[izone].connect]
            return [[j for j in i] for i in self.data[izone].connect]
        else:
            return self.get_connect(self.data[izone].shared_connectivity)

    def get_data(self, izone: int, ivar: int = None,
                 var_prefix: str = '',
                 var_partial: str = '',
                 var_end: str = '',
                 var_re: re.Pattern = None) -> np.ndarray:
        import time
        """
            Get the data from izone for variable ivar
            izone and ivar should start from 0
        """
        izone = self.nzones + izone if izone < 0 else izone
        if ivar is not None:
            pass
        elif var_prefix != '':
            ivar = next(iv for iv, v in enumerate(self.variables) if v.startswith(var_prefix))
        elif var_partial != '':
            ivar = next(iv for iv, v in enumerate(self.variables) if var_partial in v)
        elif var_end != '':
            ivar = next(iv for iv, v in enumerate(self.variables) if v.endswith(var_end))
        elif var_re is not None:
            ivar = next(iv for iv, v in enumerate(self.variables) if var_re.search(v))
        ivar = self.nvar + ivar if ivar < 0 else ivar
        offset = None
        d = self.data[izone]
        if ivar in d.ivar_zone:
            if self._read_data or d.data[d.ivar_zone.index(ivar)] is not None:
                return d.data[d.ivar_zone.index(ivar)].copy()
            else:
                offset = d.data_offset[d.ivar_zone.index(ivar)]

        if d.has_shared_var and d.shared_var[ivar] != -1:
            jzone = d.shared_var[ivar]
            return self.get_data(jzone, ivar)

        z = self.zones[izone]
        zt = z.zone_type
        vt = VarLoc.Node if not z.has_var_loc else z.var_loc[ivar]

        if offset is None:
            assert d.has_passive_var and d.passive_var[ivar] == 1
            # It's a passive variable
            vdt = np.dtype(VarTypeConstruct[d.data_type[ivar]].fmtstr)
            if zt == ZoneType.ORDERED:
                shape = squeeze_ijk(z.ijk)
                if vt == VarLoc.CellCentered:
                    shape = [i - 1 for i in shape if i > 1]
            else:
                shape = [z.num_elems if vt == VarLoc.CellCentered else z.num_pts]
            return np.zeros(shape, dtype=vdt)
        else:
            s = TecplotMatrix(
                VarTypeConstruct[d.data_type[ivar]],
                cell_centered=vt == VarLoc.CellCentered,
                ijk=z.ijk if zt == ZoneType.ORDERED else [z.num_pts if vt != VarLoc.CellCentered else z.num_elems],
                zone_type=zt
            )
            start_time = time.time()
            vname = self.variables[ivar]
            with self.__open(self.file, 'rb') as f:
                f.seek(offset)
                d.data[d.ivar_zone.index(ivar)] = s.parse_stream(f)
            Logger.info(f"Finish on-demand loading of Zone {izone:d} Variable \"{vname}\" in {time.time() - start_time:f} (s)")
            return d.data[d.ivar_zone.index(ivar)].copy()

    def get_solution_time(self, izone: int) -> float:
        return self.zones[izone].solution_time

    def get_min(self, izone: int, ivar: int) -> Union[int, float]:
        return self.data[izone].data[ivar].min

    def get_max(self, izone: int, ivar: int) -> Union[int, float]:
        return self.data[izone].data[ivar].max

    def get_dataset_aux(self, name):
        if self.has_dataset_aux:
            for i in self.dataset_aux:
                if i.name == name:
                    return i.value
        raise ValueError(f'{name} not found in dataset auxiliary data')
