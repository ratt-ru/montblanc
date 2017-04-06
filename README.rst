Montblanc
=========

A PyCUDA implementation of the Radio Interferometry Measurement
Equation, and a foothill of Mount Exaflop.

License
-------

Montblanc is licensed under the GNU GPL v2.0 License.

.. include:: docs/installation.rst

Running Tests
-------------

Once the libraries have been compiled you should be able to run the

::

    $ cd tests
    $ python -c 'import montblanc; montblanc.test()'
    $ python -m unittest test_rime_v4.TestRimeV4.test_sum_coherencies_double

which will run the entire test suite or only the specified test case,
respectively. The reported times are for the entire test case with numpy
code, and not just the CUDA kernels.

If you're running on an ubuntu laptop with optimus technology, you may
have to install bumblebee and run

::

    $ optirun python -c 'import montblanc; montblanc.test()'

Playing with a Measurement Set
------------------------------

You could also try run

::

    $ cd examples
    $ python MS_example.py /home/user/data/WSRT.MS -np 10 -ng 10 -c 100

which sets up things based on the supplied Measurement Set, with 10
point and 10 gaussian sources. It performs 100 iterations of the
pipeline.

Citing Montblanc
----------------

If you use Montblanc and find it useful, please consider citing the
related
`paper <http://www.sciencedirect.com/science/article/pii/S2213133715000633>`__.
A `arXiv <http://arxiv.org/abs/1501.07719>`__ preprint is available.

The BIRO paper is available at
`MNRAS <http://mnras.oxfordjournals.org/content/450/2/1308.abstract>`__,
and a `arXiv <http://arxiv.org/abs/1501.05304>`__ is also available.

Caveats
-------

Montblanc is an experimental package, undergoing rapid development. The
plan for 2015 is to iterate on new versions of the BIRO pipeline.

In general, I will avoid making changes to BIRO v2 and v3, but
everything beyond that may be changed, including the basic API residing
in BaseSolver.py. In practice, this means that the interfaces in the
base montblanc package will remain stable. For example:

.. code:: python

    import montblanc
    montblanc.rime_solver(...)

Everything should be considered unstable and subject to change. I will
make an effort to maintain the CHANGELOG.md, to record any breaking API
changes.
