


def _rime_factory(wrapper, output_schema):
    # Establish a sorted sequence of inputs that will correspond
    # to the arguments in the factory function
    phs = wrapper.placeholders.copy()

    main_phs = phs.pop("inputs")
    main_inputs = list(sorted(main_phs.keys()))

    source_inputs = {dsn: (_key_from_dsn(dsn), list(sorted(sphs.keys())))
                     for dsn, sphs in phs.items()}

    oreshapes = output_shapes(wrapper, output_schema, reshapes=True)

    def _rime(*args):
        main_args = args[0:len(main_inputs)]
        main_feed = {}
        main_key = _key_pool.get(1)
        source_keys = []

        dequeue_dict = {"inputs": main_key[0]}

        key_lists = []
        start = end = len(main_inputs)

        # Determine keys for our source inputs
        for dsn, (source_key, inputs) in source_inputs.items():
            # Extract argument range for this source type
            end += len(inputs)
            ds_args = args[start:end]

            if not all(isinstance(a, type(ds_args[0])) for a in ds_args[1:]):
                raise TypeError("Argument types were not all the same "
                                "type for dataset %s" % dsn)

            if isinstance(ds_args[0], list):
                nentries = len(ds_args[0])

                if not all(nentries == len(a) for a in ds_args[1:]):
                    raise ValueError("Expected lists of the same length")

                main_feed[source_key] = keys = _key_pool.get(nentries)
            elif isinstance(ds_args[0], np.ndarray):
                main_feed[source_key] = keys = _key_pool.get(1)
            else:
                raise ValueError("Unhandled input type '%s'"
                                 % type(ds_args[0]))

            key_lists.append(keys)
            source_keys.extend(keys)
            dequeue_dict[dsn] = keys
            start = end

        inputs = {n: a for n, a in zip(main_inputs, main_args)}
        inputs["time_index"].fill(0)
        inputs["antenna1"][:] = 0
        inputs["antenna2"][:] = 1

        main_feed.update(inputs)
        print("Enqueueing main inputs %s" % main_key[0])
        wrapper.enqueue("inputs", main_key[0], main_feed)
        print("Enqueueing main inputs %s done" % main_key[0])

        start = end = len(main_inputs)

        # Iteration producing something like
        # "point_inputs", ("__point_keys__", ["point_lm", "point_stokes"])
        for (dsn, (_, inputs)), keys in zip(source_inputs.items(), key_lists):
            # Extract argument range for this source type
            end += len(inputs)
            ds_args = args[start:end]

            print("Enqueueing %s inputs %s" % (dsn, keys))

            # Handle lists of source chunks
            if isinstance(ds_args[0], list):
                for e, k in enumerate(keys):
                    wrapper.enqueue(dsn, k, {n: a[e] for n, a
                                             in zip(inputs, ds_args)})
            # Handle a single source chunk
            elif isinstance(ds_args[0], np.ndarray):
                wrapper.enqueue(dsn, keys[0], {n: a for n, a
                                               in zip(inputs, ds_args)})
            else:
                raise ValueError("Unhandled input type '%s'"
                                 % type(ds_args[0]))

            print("Enqueueing %s inputs %s done" % (dsn, keys))

            start = end

        res = wrapper.dequeue(dequeue_dict)
        _key_pool.release(source_keys)
        _key_pool.release(main_key)

        # Return data, reshaping into shapes that dask will understand
        return tuple(out[r] for out, r in zip(res, oreshapes))

    return _rime
