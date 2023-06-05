find_program('cargo', required: true)
build_script = find_program('build.sh')

if get_option('buildtype') == 'release'
    profile = 'release'
    cargo_dir = 'release'
else
    profile = 'dev'
    cargo_dir = 'debug'
endif


cargo_build = custom_target('cargo_build',
                            output: 'pandas_rust_algos.so',
                            command: [
                                build_script,
                                meson.current_source_dir(),
                                profile,
                                cargo_dir,
                                meson.current_build_dir()
                            ],
                            install: true,
                            install_dir: py.get_install_dir(pure: false) / 'pandas/_libs/pandas_rust_algos'
)
