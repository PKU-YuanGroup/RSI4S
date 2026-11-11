"""Lossless column selection for large native daily-window exports."""
from collections import OrderedDict
import json

from .native_reader import PROFILES


class TableColumns:
    """Avoid a pandas DataFrame allocation for every overlapping daily window.

    Cache only explicitly selected columns in organizer memory. Exported arrays
    contain exactly the requested rows and columns; no source table is exposed.
    Selectors with transformations retain the original reader implementation.
    """
    def __init__(self, reader):
        self.reader = reader
        self.tables = OrderedDict()

    def load(self, path, selector):
        supported = {'kind', 'profile', 'columns', 'read_csv', 'row', 'rows', 'slice',
                     'warmup_days', 'target_slice'}
        positions = [key for key in ('row', 'rows', 'slice') if key in selector]
        if (selector['kind'] != 'table' or set(selector)-supported
                or len(positions) != 1):
            table = self.reader.load(path, selector)
            return {name: table[name].to_numpy() for name in table.columns}
        columns = tuple(PROFILES[selector['profile']] if 'profile' in selector else selector['columns'])
        options = selector.get('read_csv', {})
        key = (path, json.dumps(options, sort_keys=True), columns)
        if key not in self.tables:
            table = self.reader.table(path, options)
            self.tables[key] = {name: table[name].to_numpy(copy=False) for name in columns}
            if len(self.tables) > 6:
                self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        position = positions[0]
        indices = ([selector['row']] if position == 'row' else
                   selector['rows'] if position == 'rows' else slice(*selector['slice']))
        # Copies prevent a small exported window from retaining an entire source
        # table after it leaves the bounded cache. No preprocessing is performed.
        return {name: values[indices].copy() for name, values in self.tables[key].items()}

    def inputs(self, row):
        if len(row['input_path']) != len(row['input_selector']):
            raise ValueError('Path/selector count mismatch')
        return [self.load(path, selector) for path, selector in
                zip(row['input_path'], row['input_selector'])]
