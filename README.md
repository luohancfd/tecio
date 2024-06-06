# TecIO

![pypi](https://img.shields.io/pypi/v/tecio.svg?logo=python&logoColor=white) ![GitHub Tag](https://img.shields.io/github/v/tag/luohancfd/tecio?color=green)
![python](https://img.shields.io/badge/python>=3.9-blue.svg) ![GitHub License](https://img.shields.io/github/license/luohancfd/tecio)


## Introduction
TecIO is a pure Python package licensed under GPL v3, designed to facilitate the reading and writing of data files in the Tecplot&reg; binary format. Please note that Tecplot&reg; is a registered trademark belonging to Tecplot, Inc in the United States and other countries. This package is not affiliated with or endorsed by Tecplot, Inc.

TecIO is written entirely in Python and does not require the installation of Tecplot products; or possession of a Tecplot license. However, you need to have [Preplot&trade;](https://tecplot.com/2017/01/05/preplot-szl-convert-overview/) to read ASCII format file.

Tecplot binary file format can be found at [360 data format guid](https://raw.githubusercontent.com/su2code/SU2/master/externals/tecio/360_data_format_guide.pdf) Appendix A.

## Installation
You can install TecIO using pip:
```bash
pip install tecio
```

## Dependencies
TecIO depends on the following Python packages:
- [NumPy](https://numpy.org/): For efficient numerical operations.
- [Construct](https://construct.readthedocs.io/): For parsing and building binary data structures.

These dependencies will be automatically installed when installing TecIO.

## Usage
Check `test` folder

## License
TecIO is licensed under the GPL v3 license. See [LICENSE](LICENSE) for more details.

![GPL v3 License](https://www.gnu.org/graphics/gplv3-or-later.svg)

## Disclaimer

Tecplot®, Tecplot 360,™ Tecplot 360 EX,™ Tecplot Focus, the Tecplot product logos, Preplot,™ Enjoy the View,™ Master the View,™ SZL,™ Sizzle,™ and Framer™ are registered trademarks or trademarks of Tecplot, Inc. in the United States and other countries. All other product names mentioned herein are trademarks or registered trademarks of their respective owners. This package is not affiliated with or endorsed by Tecplot, Inc.

## Support and Contributions
For any issues or feature requests, please open an issue on [GitHub](https://github.com/luohancfd/tecio).

Contributions are welcome! Feel free to fork the repository and submit pull requests.
