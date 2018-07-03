from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask.array as da

from xarrayms import xds_from_ms, xds_from_table

from montblanc.impl.rime.tensorflow.datasets.dataset import Dataset


class MeasurementSet(Dataset):
    def __init__(self, ms, **kwargs):
        self._ms = ms
        self._kwargs = kwargs

        self._dim_sizes = None
        self._dim_chunks = None

    def _inspect_ms(self):
        """
        Computes dimension sizes and chunking strategies for
        the Measurement Set.
        """
        # Perform inspection
        kwargs = self._kwargs.copy()
        kwargs['columns'] = ["TIME"]

        xds = list(xds_from_ms(self._ms, **kwargs))

        # Get the antenna dataset
        ant_ds = list(xds_from_table('::'.join((self._ms, "ANTENNA"))))
        assert len(ant_ds) == 1
        ant_ds = ant_ds[0].rename({'row': 'antenna'}).drop('table_row')

        # Get datasets for DATA_DESCRIPTION, SPECTRAL_WINDOW
        # POLARIZATION and FIELD, partitioned by row
        ddid_tab = '::'.join((self._ms, "DATA_DESCRIPTION"))
        spw_tab = '::'.join((self._ms, "SPECTRAL_WINDOW"))
        pol_tab = '::'.join((self._ms, "POLARIZATION"))

        ddid_ds = list(xds_from_table(ddid_tab, group_cols="__row__"))
        spwds = list(xds_from_table(spw_tab, group_cols="__row__"))
        pds = list(xds_from_table(pol_tab, group_cols="__row__"))

        def _join_subtables(ds):
            """
            Join Spectral Window and Polarization
            datasets, given the Data Descriptor ID
            """
            ddid = ddid_ds[ds.attrs['DATA_DESC_ID']].drop('table_row')
            spw = spwds[ddid.SPECTRAL_WINDOW_ID.values].drop('table_row')
            pol = pds[ddid.POLARIZATION_ID.values].drop('table_row')

            return ds.assign(ANTENNA_POSITION=ant_ds.POSITION,
                             FREQUENCY=spw.CHAN_FREQ,
                             CORRELATION_TYPE=pol.CORR_TYPE,
                             CORRELATION_PRODUCT=pol.CORR_PRODUCT)

        xds = [_join_subtables(ds) for ds in xds]

        # Get the unique times and their counts for each grouping
        # We use the counts (number of occurrences of a unique time
        # over consecutive rows) as the row chunking strategy
        utime_counts = [da.unique(ds.TIME.data, return_counts=True)
                        for ds in xds]
        utime_counts = da.compute(utime_counts)[0]

        # Dimensions for each group
        ds_dims = [ds.dims for ds in xds]

        # Calculate dimension sizes and chunks for each group
        self._dim_sizes = [{
                'time': len(counts),
                'ant': dims['antenna'],
                'row': dims['row'],
                'corr': dims['corr'],
                'chan': dims['chan'],
            }
            for dims, (times, counts) in zip(ds_dims, utime_counts)]

        self._dim_chunks = [{
                'time': (1,) * len(counts),
                'ant': (dims['antenna'],),
                'row': tuple(counts),
                'corr': (dims['corr'],),
                'chan': (dims['chan'],),
            }
            for dims, (times, counts) in zip(ds_dims, utime_counts)]

        # Check that chunk sums equal dimension sizes
        for chunks, sizes in zip(self._dim_chunks, self._dim_sizes):
            assert chunks.keys() == sizes.keys()
            for dim in sizes.keys():
                if not sum(chunks[dim]) == sizes[dim]:
                    raise ValueError("%s sum(%s) != %d" %
                                     (dim, sum(chunks[dim]), sizes[dim]))

    def dim_sizes(self):
        # Get sizes lazily
        if self._dim_sizes is None:
            self._inspect_ms()

        return self._dim_sizes

    def dim_chunks(self):
        # Get chunks lazily
        if self._dim_chunks is None:
            self._inspect_ms()

        return self._dim_chunks

    def dataset(self, chunks=None):
        if chunks is None:
            if self._dim_chunks is None:
                self._inspect_ms()

            chunks = self._dim_chunks

        if isinstance(chunks, tuple):
            chunks = list(chunks)
        if not isinstance(chunks, list):
            chunks = [chunks]

        if not all(isinstance(c, dict) for c in chunks):
            raise ValueError("All chunks must be dictionaries")

        diff = len(self._dim_chunks) - len(chunks)

        if diff > 0:
            chunks = chunks + [chunks[-1]] * diff

        kwargs = self._kwargs.copy()
        kwargs['chunks'] = chunks

        return list(xds_from_ms(self._ms, **kwargs))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("ms")
    args = p.parse_args()

    ds = MeasurementSet(args.ms)

    from pprint import pprint

    # pprint(ds.dim_sizes())
    print([{k: sum(v) for k, v in elem.items()} for elem in ds.dim_chunks()])
    print(ds.dataset())
