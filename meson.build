find_program('cargo', required: true)

if get_option('buildtype') == 'release'
    profile = 'release'
    cargo_dir = 'release'
else
    profile = 'dev'
    cargo_dir = 'debug'
endif

cmd = run_command('cargo', 'build', '--profile', profile, check: true)
output = cmd.stdout().strip()
errortxt = cmd.stderr().strip()
message(output)
message(errortxt)  # ninja only prints to stdout, but we still want cargo warnings

if host_machine.system() == 'windows'
    fs.copyfile(
        'target/' + cargo_dir + '/' + 'libpandas_rust_algos.dll',
        'pandas_rust_algos.pyd',
        install: true,
        install_dir: py.get_install_dir(pure: false) / 'pandas/_libs/pandas_rust_algos'
    )
elif host_machine.system() == 'darwin'
    fs.copyfile(
        'target/' + cargo_dir + '/' + 'libpandas_rust_algos.dylib',
        'pandas_rust_algos.so',
        install: true,
        install_dir: py.get_install_dir(pure: false) / 'pandas/_libs/pandas_rust_algos'
    )
else
    fs.copyfile(
        'target/' + cargo_dir + '/' + 'libpandas_rust_algos.so',
        'pandas_rust_algos.so',
        install: true,
        install_dir: py.get_install_dir(pure: false) / 'pandas/_libs/pandas_rust_algos'
    )
endif
