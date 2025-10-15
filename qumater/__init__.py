"""Top level package for QuMater.

This module exposes high level namespaces for materials and qsim. Most
functionality is available from subpackages and the CLI; see
`help(qumater)` for details.
"""

from importlib import import_module as _import_module

__all__ = ['materials', 'qsim', 'platform']
materials = _import_module('qumater.materials')
qsim = _import_module('qumater.qsim')
platform = _import_module('qumater.platform')
