# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import subprocess
import os

import addonmanager_utilities
import freecad


def pip_list(python_exe: str, vendor_path: str):
    res = subprocess.run(
        [
            python_exe,
            '-m',
            'pip',
            'list',
            '--path',
            vendor_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=True,
    )

    if res.returncode != 0:
        print(f'Exit-code: {res.returncode}')
        print(f'Error:     {res.stderr}')
    else:
        for package in res.stdout.decode('utf-8').split('\n'):
            print(package)


def pip_install(python_exe: str, vendor_path: str, package_args: list[str]):
    res = subprocess.run(
        [
            python_exe,
            '-m',
            'pip',
            'install',
            '--disable-pip-version-check',
            '--target',
            vendor_path,
            *package_args,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=True,
    )

    if res.returncode != 0:
        print(f'Exit-code: {res.returncode}')
        print(f'Error:     {res.stderr}')
    else:
        print(res.stdout.decode('utf-8'))


def main():
    python_exe = freecad.utils.get_python_exe()
    vendor_path = addonmanager_utilities.get_pip_target_directory()
    if not os.path.exists(vendor_path):
        os.makedirs(vendor_path)

    print(f'python: {python_exe}')
    print(f'vendor: {vendor_path}')

    pip_list(python_exe, vendor_path)
    # pip_install(python_exe, vendor_path, ['-e', '/home/tobante/Developer/tobanteAudio/pffdtd'])


if __name__ == '__main__':
    main()
