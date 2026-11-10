import sys
from build_capture import install, mark_terminal_error
from torch.utils.cpp_extension import load_inline
install()
mode=sys.argv[1]
try:
    source = ('int fixture() { return missing_fixture_symbol; }' if mode in ('quiet','verbose') else ('extern int missing_link_fixture(); int fixture() { return missing_link_fixture(); }' if mode == 'undefined' else 'int fixture() { return 1; }'))
    load_inline(name='repair_fixture_'+mode, cpp_sources=source, functions=['fixture'],
                extra_ldflags=(['-lrepair_nonexistent_library'] if mode=='link' else ['-Wl,--no-undefined'] if mode=='undefined' else []), verbose=mode!='quiet')
except RuntimeError as error:
    # Deliberately wrap to reproduce the previous loss of the nested diagnostic.
    try: raise RuntimeError('wrapped extension failure') from error
    except RuntimeError as wrapped: mark_terminal_error(wrapped)
    sys.exit(0)
sys.exit(1)
