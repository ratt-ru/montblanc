import csv
import re

class ParseResults(object):
    def __init__(self):
        self.arrays = {}
        self.src_counts = {}

    def group_arys(self, l):
        """

        """
        rl = []

        for a in l:
            if not self.arrays.has_key(a):
                raise ValueError('ParseResult object '
                    'does not have a %s array' % a)

            rl.append(self.arrays[a])

        return rl


__sky_model_re = re.compile('^\s*?#\s*?format\s+?(?P<src_count>\S*?)\s*?:\s*?(?P<fmt_spec>\S*?)$')

def parse_sky_model(filename):
    """
    Parse a sky model file. Hacky.
    """

    results = ParseResults()

    fmt_specs = []
    src_count = None

    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            # Check for a format string in
            # these values. Do a cheap comparison
            # before using expensive regular expressions
            if row[0].startswith('# format'):
                match = __sky_model_re.search(row[0])

                # Update the format list array if we match
                if match:
                    src_count = match.group('src_count')
                    fmt_specs = [match.group('fmt_spec')]
                    fmt_specs.extend([v.strip() for v in row[1:]])
                    continue

            # CSV file returns stuff separated
            # by commas in a list
            for idx, value in enumerate(row):
                # Only handle values if we have a
                # format specifier for them
                if idx >= len(fmt_specs):
                    break

                fmt_spec = fmt_specs[idx]
                l = results.arrays.get(fmt_spec, [])
                l.append(value.strip())
                results.arrays[fmt_spec] = l

            # If we've reached this point, we've
            # handled a row defining a source,
            # increment the associated source type.
            if src_count is not None:
                t = results.src_counts.get(src_count, 0)
                results.src_counts[src_count] = t+1

    return results