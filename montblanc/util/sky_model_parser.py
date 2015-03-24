import csv
import re

class ParseResults(object):
    def __init__(self):
        self.arrays = {}
        self.src_types = {}

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


__sky_model_re = re.compile('^\s*?#\s*?format\s+(?P<src_type>.*?):(?P<format_specifiers>.*?)$')

def parse_sky_model(filename):
    """
    Parse a sky model file. Hacky.
    """

    results = ParseResults()

    format_specifiers = []
    src_type = None

    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            force_next_row = False
            # CSV file returns stuff separated
            # by commas in a list
            for idx, value in enumerate(row):
                # We've been told to bail out of
                # this loop lower down
                if force_next_row is True:
                    break

                # Check for a format string in
                # these values. Do a cheap comparison
                # before using expensive regular expressions
                if value.startswith('# format'):
                    match = __sky_model_re.search(value)

                    # Update the format list array and continue
                    # to the next row
                    if match:
                        src_type = match.group('src_type')
                        format_specifiers = match.group('format_specifiers').split()
                        force_next_row = True
                        continue
                # Handle the value if we have a
                # format specifier for it
                elif idx < len(format_specifiers):
                    format_specifier = format_specifiers[idx]
                    l = results.arrays.get(format_specifier, [])
                    l.append(value.strip())
                    results.arrays[format_specifier] = l

            if force_next_row is True:
                continue

            # If we've reached this point, we've
            # handled a row defining a source,
            # increment the associated source type.
            if src_type is not None:
                t = results.src_types.get(src_type, 0)
                results.src_types[src_type] = t+1

    return results