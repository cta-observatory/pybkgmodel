.. _userguide:

User Guide
==========

Generate a background model
---------------------------

The background model generation is controlled via a configuration file in YAML format (see :ref:`userguide-cfg`).

Execute :ref:`bkgmodel` to run the code, specifying the corresponding configuration file, e.g.:

.. code-block::

    bkgmodel --config examples/config_example.yaml


.. _userguide-cfg:

Configuration
-------------

In the configuration file, it is possible to specify the input data,
output folder, background model generation method, maps binning and exclusion regions to apply.

.. literalinclude:: ../../examples/config_example.yaml
   :language: yaml