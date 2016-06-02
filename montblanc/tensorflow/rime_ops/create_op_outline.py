import argparse
import re

from op_source_templates import (MAIN_HEADER_TEMPLATE,
    CPP_HEADER_TEMPLATE,
    CPP_SOURCE_TEMPLATE,
    CUDA_HEADER_TEMPLATE,
    CUDA_SOURCE_TEMPLATE,
    PYTHON_SOURCE_TEMPLATE)

MODULE='rime'
PROJECT='montblanc'

# Convert CamelCase op names to snake_case
def camel_to_snake_case(name):
    FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
    ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')
    s1 = FIRST_CAP_RE.sub(r'\1_\2', name)
    return ALL_CAP_RE.sub(r'\1_\2', s1).lower()

# Derive a C++ header guard from the header name
def header_guard(header_name):
    guard_str = header_name.replace('.', '_')
    return ''.join([MODULE, '_', guard_str]).upper()

parser = argparse.ArgumentParser()
parser.add_argument('opname')
args = parser.parse_args()

snake_case = camel_to_snake_case(args.opname)

# Create dictionary with variables required for creating the templates
D = {
    'opname' : args.opname,
    'project' : PROJECT,
    'module' : MODULE,
    'snake_case' : snake_case,
    'library' : ''.join([MODULE, '.so']),
}

# Filenames
D.update({
    'main_header_file' : ''.join([snake_case, '_op.h']),
    'cpp_header_file' : ''.join([snake_case, '_op_cpu.h']),
    'cpp_source_file' : ''.join([snake_case, '_op_cpu.cpp']),
    'cuda_header_file' : ''.join([snake_case, '_op_gpu.cuh']),
    'cuda_source_file' : ''.join([snake_case, '_op_gpu.cu']),
    'python_test_file' : ''.join(['test_', snake_case, '.py'])
})

# C++ header guards
D.update({
    'main_header_guard' : header_guard(D['main_header_file']),
    'cpp_header_guard' : header_guard(D['cpp_header_file']),
    'cuda_header_guard' : header_guard(D['cuda_header_file']),
})

# C++ namespace
D.update({
    'project_namespace_start' : ''.join([PROJECT, '_namespace_begin']).upper(),
    'project_namespace_stop' : ''.join([PROJECT, '_namespace_stop']).upper(),
    'op_namespace_start' : ''.join([PROJECT, '_', snake_case, '_namespace_begin']).upper(),
    'op_namespace_stop' : ''.join([PROJECT, '_', snake_case, '_namespace_stop']).upper(),
})

# kernel names
D.update({
    'kernel_name' : ''.join([MODULE, '_', snake_case])
})

# Write out each file, substituting template variables
with open(D['main_header_file'], 'w') as f:
    f.write(MAIN_HEADER_TEMPLATE.substitute(**D))

with open(D['cpp_header_file'], 'w') as f:
    f.write(CPP_HEADER_TEMPLATE.substitute(**D))

with open(D['cpp_source_file'], 'w') as f:
    f.write(CPP_SOURCE_TEMPLATE.substitute(**D))

with open(D['cuda_header_file'], 'w') as f:
    f.write(CUDA_HEADER_TEMPLATE.substitute(**D))

with open(D['cuda_source_file'], 'w') as f:
    f.write(CUDA_SOURCE_TEMPLATE.substitute(**D))

with open(D['python_test_file'], 'w') as f:
    f.write(PYTHON_SOURCE_TEMPLATE.substitute(**D))
