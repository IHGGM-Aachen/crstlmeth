Getting Started
===============

Install from source:

.. code-block:: bash

   pip install -e .

Run the CLI:

.. code-block:: bash

   crstlmeth calculate methylation myfile.bedmethyl.gz chr1 1000000 1001000

Or launch the web UI:

.. code-block:: bash

   crstlmeth web

Input expectations:

- Input: bgzipped + indexed `.bedmethyl.gz`
- Optional: MLPA BED definitions
- Output: plots, reference `.cmeth` files
