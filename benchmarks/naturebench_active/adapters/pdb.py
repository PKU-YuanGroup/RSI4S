"""Repair widened PDB coordinate fields without changing the frozen predictor."""
import math
import re


ATOM=re.compile(r'^ATOM  (\s*\d+) (.{4}) (.{3}) (.)(\s*\d+)    '
                r'\s*([+-]?\d+\.\d{3})\s*([+-]?\d+\.\d{3})\s*([+-]?\d+\.\d{3})(.*)$')
HEADER=re.compile(r'^ATOM  (\s*\d+) (.{4}) (.{3}) (.)(\s*\d+)    ')
NUMBER=re.compile(r' *[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)? *')


def fixed_columns(text):
    """Use the most decimal places fitting each native eight-column float.

    Python's :8.3f expands past eight columns for large coordinates, displacing
    later fields. Keep raw predictions unchanged; this serializer only rounds
    the native output copy when its fixed-width format requires fewer decimals.
    """
    lines=[];rounded=0;changed=0;wrapped=0;maximum=0.
    for line in text.splitlines():
        if not line.startswith('ATOM  '):
            lines.append(line);continue
        match=ATOM.fullmatch(line)
        if match is None:
            # The native reader uses float() on eight-column fields; integers,
            # shorter decimals and exponent notation are already valid. Keep
            # their exact representation and coordinates, without re-rounding.
            header=HEADER.match(line)
            if header is None:raise ValueError('Unrecognized generated PDB atom layout')
            tail=line[header.end():]
            fields=[tail[start:start+8] for start in (0,8,16)]
            if any(len(field)!=8 or NUMBER.fullmatch(field) is None for field in fields):
                raise ValueError('Unrecognized generated PDB atom layout')
            if not all(math.isfinite(float(field)) for field in fields):
                raise ValueError('Nonfinite PDB coordinate')
            serial,residue=int(header[1]),int(header[5])
            if residue>9999:raise ValueError('PDB residue number exceeds fixed-width range')
            wrapped+=serial>99999;serial%=100000
            fixed=f'ATOM  {serial:5d} {header[2]} {header[3]} {header[4]}{residue:4d}    '+tail
            changed+=fixed!=line;lines.append(fixed)
            continue
        serial,residue=int(match[1]),int(match[5])
        if residue>9999:raise ValueError('PDB residue number exceeds fixed-width range')
        # Native parsing never reads atom serials. Wrap them, as PDB writers do
        # above 99,999 atoms, instead of rejecting a complete structure.
        wrapped+=serial>99999;serial%=100000
        fields=[]
        for field in match.group(6,7,8):
            value=float(field)
            if not math.isfinite(value):raise ValueError('Nonfinite PDB coordinate')
            for decimals in (3,2,1,0):
                encoded=f'{value:8.{decimals}f}'
                if len(encoded)==8:break
            else:raise ValueError('PDB coordinate exceeds fixed-width range')
            error=abs(float(encoded)-value)
            rounded+=error>0;maximum=max(maximum,error);fields.append(encoded)
        fixed=f'ATOM  {serial:5d} {match[2]} {match[3]} {match[4]}{residue:4d}    '+''.join(fields)+match[9]
        changed+=fixed!=line;lines.append(fixed)
    return '\n'.join(lines)+'\n',dict(reformatted_atoms=changed,rounded_coordinate_values=rounded,
        wrapped_atom_serials=wrapped,
        maximum_absolute_rounding_angstrom=round(maximum,6),raw_prediction_changed=False)
