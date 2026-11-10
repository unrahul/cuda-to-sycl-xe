# Managed assessment contract

Each shell command starts in the task root; `cd` does not persist between commands. The assessment image already configures the compiler and SDK paths; do not mutate the environment. Submit `submission/model_new.py` and local sources. The free static validator snapshots the same files and calls the same scanner as assessment. Run it before the first assessment and after source edits until it passes. To check without using an assessment, write `{"id":1,"action":"validate"}` to `requests/validate-1.json` and read `feedback/validate-1.json`; use a fresh integer 1..999999 for each new check. For assessment number 1..6, write `{"id":1}` to `requests/1.json` and read `feedback/1.json`. Use an atomic rename when creating request files, wait for feedback, and do not reuse IDs. Validation checks syntax and static policy only: it does not compile or prove correctness.

The submission must contain a regular `model_new.py`; no symlinks. Broker limits: 128 files, 10 MiB per file, 64 MiB total, 512 entries, depth 16. Only .py, .cpp, .cc, .cxx, .c, .hpp, .h, .sycl, .json, .txt are copied. A class named `ModelNew` is required. The scanner parses only `model_new.py` as Python AST. It scans that file and local .c, .cc, .cpp, .cxx, .h, .hh, .hpp, and .sycl files; it skips hidden directories and .git, .torch_extensions, build, __pycache__, and .cache. Note the source-only broker does not copy .hh files. It requires a recognized torch extension loader or colocated native module and a visible SYCL marker (or oneMKL/oneDNN marker for library backend). `SYCL_BACKEND` assignments must be literal strings `custom` or `library`, with no conflicting values; omitted means custom. The native-module alternative requires an actual colocated module and cannot pass through this source-only broker. Comments are removed for banned-pattern scans; backend marker search includes comments.

Python AST rules (matching terminal call names, not merely builtins):
- Blocked import roots: `aiohttp`, `asyncio`, `builtins`, `concurrent`, `ctypes`, `dis`, `ftplib`, `http`, `httpx`, `importlib`, `inspect`, `marshal`, `multiprocessing`, `pickle`, `requests`, `resource`, `runpy`, `signal`, `smtplib`, `socket`, `ssl`, `subprocess`, `sys`, `telnetlib`, `urllib`, `websockets`.
- Blocked call names: `__import__`, `breakpoint`, `callable`, `compile`, `delattr`, `dir`, `eval`, `exec`, `getattr`, `globals`, `hasattr`, `help`, `id`, `input`, `isinstance`, `locals`, `open`, `setattr`, `type`, `vars`.
- Blocked os process call names: `execv`, `execve`, `execvp`, `execvpe`, `fork`, `forkpty`, `kill`, `popen`, `posix_spawn`, `posix_spawnp`, `spawnl`, `spawnle`, `spawnlp`, `spawnlpe`, `spawnv`, `spawnve`, `spawnvp`, `spawnvpe`, `startfile`, `system`.
- Recognized loader names: `load`, `load_inline`, `load_library`, `load_module`.
- Any import path containing `kernelbench`, `reference`, or `task` is rejected. `os.environ`, `os.putenv`, `os.unsetenv` access and calls under `os.environ` are rejected. Dunder attributes are rejected except `__init__`, `__file__`, `__name__`.
- Recognized `load` must contain `cpp_extension` in its call name, or be imported directly from `torch.utils.cpp_extension`; `load_inline`, `load_library`, `load_module` are recognized by terminal call name.

The precise case-insensitive regular expressions for static source scans follow. They are applied to each scanned source after comments are stripped (Python token comments; C-style comments for other files).

Low precision rejected:

- `\b(?:torch\s*\.\s*)?(?:float16|bfloat16|float8_e[45]m[23])\b`
- `\b(?:__half2?|sycl::half2?|cl::sycl::half|half_t|_Float16|float16_t)\b`
- `\.\s*half\s*\(`
- `\b(?:const\s+|volatile\s+)?half(?:\s*[*&]+\s*|\s+)[A-Za-z_]\w*\b`
- `\bstatic_cast\s*<\s*half\s*>`

Process/network rejected:

- `\bsys/socket\.h\b|\bnetinet/|\bsys/un\.h\b`
- `\b(?:fork|execve|execvp|posix_spawn|curl_easy_init|socket|connect)\s*\(`

Torch compute rejected:

- `\b(?:torch|at|torch::)\s*(?:\.|::)\s*(?:matmul|mm|bmm|add|sub|mul|div|sum|mean|softmax|relu|sigmoid|tanh|einsum|convolution|conv2d)\s*\(`

Custom marker required:

- `\bsycl::|\bcl::sycl::|\bparallel_for\s*\(|\bsycl_ext::oneapi::`

Library marker required:

- `\boneapi::mkl\b|\b(?:onemkl|one_mkl)\b|\b(?:onednn|dnnl)\b`

At runtime, forward may use only the following ATen operations and `aten::sym_*`; `aten::to` and `aten::_to_copy` are allowed only without a device or dtype change. Any other dispatched ATen operation is rejected. This permits allocations, metadata operations, clone and copy but does not permit eager ATen compute delegation. The candidate must return the right device, structure, shape, dtype, alias behavior and numeric values, match the reference's input mutation behavior exactly, and be repeatable on a nondefault current stream. Tests use independent reference instances and private cases; feedback never reveals reference values or seeds. FP32-preserving calculation is required; listed lower-precision patterns are rejected. Runtime device errors, compilation errors, policy rejection, and numeric mismatches have distinct categories.

Allowed ATen names: `aten::empty`, `aten::empty_strided`, `aten::empty_like`, `aten::new_empty`, `aten::new_empty_strided`, `aten::view`, `aten::view_as`, `aten::reshape`, `aten::_reshape_alias`, `aten::as_strided`, `aten::transpose`, `aten::permute`, `aten::t`, `aten::mT`, `aten::mH`, `aten::squeeze`, `aten::unsqueeze`, `aten::select`, `aten::slice`, `aten::narrow`, `aten::expand`, `aten::expand_as`, `aten::unbind`, `aten::split`, `aten::split_with_sizes`, `aten::chunk`, `aten::detach`, `aten::alias`, `aten::clone`, `aten::copy_`, `aten::contiguous`, `aten::resize_`, `aten::resize_as_`, `aten::set_`, `aten::set_storage`, `aten::lift_fresh`, `aten::size`, `aten::sym_size`, `aten::numel`, `aten::sym_numel`, `aten::dim`, `aten::stride`, `aten::storage_offset`, `aten::is_contiguous`, `aten::is_same_size`, `aten::is_set_to`, `aten::is_pinned`, `aten::storage`, `aten::untyped_storage`.
