import sys
import site
import os
import sysconfig  # Make sure to import sysconfig

print("Python Executable:", sys.executable)
print("Python Version:", sys.version)
print("Python Include Path:", sysconfig.get_paths()['include'])
print("Site-packages:", site.getsitepackages())
print("sys.path:")
for path in sys.path:
    print("  ", path)

