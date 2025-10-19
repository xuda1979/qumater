"""Top level package for QuMater.

This module exposes high level namespaces for materials, quantum simulation
primitives, orchestration utilities, and platform integrations. Most
functionality is available from subpackages and the CLI; see
``help(qumater)`` for details.
"""

from importlib import import_module as _import_module

__all__ = ['materials', 'qsim', 'platform', 'core', 'workflows', 'cli']
materials = _import_module('qumater.materials')
qsim = _import_module('qumater.qsim')
platform = _import_module('qumater.platform')
core = _import_module('qumater.core')
workflows = _import_module('qumater.workflows')


def cli(argv=None):
    """Entry point for invoking :mod:`qumater.cli` from Python code."""

    from .cli import main as _main

    return _main(argv)
