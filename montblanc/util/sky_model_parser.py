import csv
import re

__sky_model_re = re.compile('^\s*?#\s*?format:(.*?)$')

class ParseResults(object):
    def __init__(self):
        self.arrays = {}

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

def parse_sky_model(filename):
    """
    Parse a sky model file
    """

    results = ParseResults()

    format_specifiers = []

    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            force_next_row = False
            # CSV file returns comma separated stuff
            for idx, value in enumerate(row):
                if force_next_row is True:
                    force_next_row = False
                    continue

                # Check for a format string in
                # these values. Do a cheap comparison
                # before using expensive regular expressions
                if value.startswith('#'):
                    match = __sky_model_re.search(value)

                    if match:
                        # Update the format list array and continue
                        # to the next row
                        format_specifiers = match.group(1).split()
                        force_next_row = True
                        continue
                else:
                    # Next line if we don't have a format
                    # specifier for this value
                    if idx >= len(format_specifiers):
                        continue

                    format_specifier = format_specifiers[idx]
                    l = results.arrays.get(format_specifier, [])
                    l.append(value.strip())
                    results.arrays[format_specifier] = l

    return results