/*
 * jasna.exe for the overlay Windows build.
 *
 * The official release compiles Jasna with Nuitka. This launcher gives the
 * overlay build the same observable layout without a compiler-heavy step: it
 * embeds the python313.dll that sits next to it, points the interpreter at the
 * dist root (site-packages laid out flat, like Nuitka does) and at Lib\ (the
 * standard library), marks the process as frozen so `jasna._frozen.is_frozen()`
 * resolves tools\, model_weights\ and the fatbins relative to the executable,
 * and runs `jasna.__main__` with the original argv.
 *
 * Console subsystem on purpose: the GUI path drops its own console
 * (`drop_console_window`), and the CLI keeps it.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <windows.h>
#include <wchar.h>

#define PATH_CAP 32768

static const char BOOTSTRAP[] =
    "import sys\n"
    "sys.frozen = True\n"
    "import runpy\n"
    "runpy.run_module('jasna', run_name='__main__', alter_sys=False)\n";

static void fail(const wchar_t *what) {
    fwprintf(stderr, L"jasna launcher: %ls\n", what);
    ExitProcess(2);
}

static void append_path(PyConfig *config, const wchar_t *dir, const wchar_t *leaf) {
    wchar_t buf[PATH_CAP];
    if (leaf == NULL) {
        wcscpy_s(buf, PATH_CAP, dir);
    } else {
        swprintf(buf, PATH_CAP, L"%ls\\%ls", dir, leaf);
    }
    PyStatus status = PyWideStringList_Append(&config->module_search_paths, buf);
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }
}

int wmain(int argc, wchar_t **argv) {
    wchar_t exe[PATH_CAP];
    DWORD n = GetModuleFileNameW(NULL, exe, PATH_CAP);
    if (n == 0 || n >= PATH_CAP) {
        fail(L"cannot resolve the executable path");
    }
    wchar_t dir[PATH_CAP];
    wcscpy_s(dir, PATH_CAP, exe);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash == NULL) {
        fail(L"unexpected executable path");
    }
    *slash = L'\0';

    /* python313.dll and the extension modules live next to the launcher, and
     * packages add their own directories later via os.add_dll_directory. */
    SetDllDirectoryW(dir);

    PyStatus status;
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    config.parse_argv = 0;
    config.site_import = 0;
    config.write_bytecode = 1;
    config.buffered_stdio = 1;
    config.module_search_paths_set = 1;

    status = PyConfig_SetString(&config, &config.program_name, exe);
    if (PyStatus_Exception(status)) Py_ExitStatusException(status);
    status = PyConfig_SetString(&config, &config.executable, exe);
    if (PyStatus_Exception(status)) Py_ExitStatusException(status);
    status = PyConfig_SetString(&config, &config.home, dir);
    if (PyStatus_Exception(status)) Py_ExitStatusException(status);

    append_path(&config, dir, NULL);
    append_path(&config, dir, L"Lib");

    status = PyConfig_SetArgv(&config, argc, argv);
    if (PyStatus_Exception(status)) Py_ExitStatusException(status);

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) Py_ExitStatusException(status);

    /* SystemExit inside the bootstrap ends the process with its code, like a
     * normal `python -m jasna`; any other uncaught error prints and exits 1. */
    int rc = PyRun_SimpleString(BOOTSTRAP);
    if (rc != 0) {
        Py_Finalize();
        return 1;
    }
    return Py_FinalizeEx() < 0 ? 120 : 0;
}
