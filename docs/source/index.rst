.. pybkgmodel documentation master file, created by
   sphinx-quickstart on Fri Jul 14 11:55:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**pybkgmodel** documentation
============================

.. |ci| image:: https://github.com/cta-observatory/pybkgmodel/actions/workflows/ci.yml/badge.svg?branch=main
  :target: https://github.com/cta-observatory/pybkgmodel/actions/workflows/ci.yml

.. |coverage| image:: https://codecov.io/gh/cta-observatory/pybkgmodel/branch/main/graph/badge.svg
  :target: https://codecov.io/gh/cta-observatory/pybkgmodel

|ci| |coverage|

.. _CTA: https://www.cta-observatory.org/
.. _LST: https://www.cta-observatory.org/project/technology/lst/
.. _MAGIC: https://magic.mpp.mpg.de/
.. _pybkgmodel: https://github.com/cta-observatory/pybkgmodel

.. warning::

  The project is under active development, so major changes are possible without notice.

  Please use
  `issues <https://gitlab.pic.es/mstrzys/pybkgmodel/issues>`_
  to report problems or make suggestions.

`pybkgmodel`_ is a background model generation tool for Imaging Atmospheric Cherenkov Telescopes (IACTs).
It allows building background maps from the available data excluding the region of interest from the process.

Supported **background generation methods**:

- "wobble map" - assumes IACT observations were performed wobbling around the target position. For each telescope pointing,
  the background is generated from the IACT camera half, which does not include the source position;
- "exclusion map" - excludes the specified sky region from consideration and generates the background model from the remaining data.

Supported **background generation modes**:

- "run-wise": for each telescope "data run" (observation session unit) identifies other runs close to it in time
  and constructs the individual background model from them only;
- "stacked": add the "run-wise" models together, resulting in an observation-averaged background model.

The latter is in general less noisy than the individual "run-wise" models at the cost of losing information
on the potential background variation during the observations.

.. note::

  Despite the initial focus on
  `CTA`_/`LST`_ and `MAGIC`_ data,
  the project may be extended to any other IACTs (e.g. other CTA instruments).

.. toctree::
   :hidden:
   :maxdepth: 2

   install
   userguide
   reference
   contribute
   authors
