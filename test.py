import subprocess
import os
import sys
from typing import List
import colorama
from colorama import Fore, Style
import time

colorama.init()

# python -m timeit -s "import subprocess" "subprocess.call([\"python\", \"test.py\"])"
# runs the test script multiple times and reports the best run time
# Take it with a grain of salt, it has some overhead (~10% from what I've seen)


def RUN_CMD(filename):
    return ["python", "haupt.py", f"input={filename}.hpt", "optimize=all", "silenced=True"]


def call_cmd(cmd: List):
    print(Fore.CYAN + "[CMD] " + " ".join(cmd) + Style.RESET_ALL)
    return subprocess.call(cmd)


def time_output(filename: str):
    print(f"Running {filename}.hpt")
    start_time = time.perf_counter_ns()
    exitcode = call_cmd(RUN_CMD(filename))
    print(f"Compiling {filename}.hpt exited with exit code " + str(exitcode))
    if exitcode != 0:
        print(Fore.RED + "COMPILATION FAILED: " + filename + Style.RESET_ALL)
        exit(1)
    compile_time = (time.perf_counter_ns() - start_time) / 1e6
    print(f"Compilation took {compile_time}ms.")
    start_time = time.perf_counter_ns()
    exitcode = call_cmd([f"{filename}.exe"])
    print(f"Executing {filename}.exe exited with exit code " + str(exitcode))
    if exitcode != 0:
        print(Fore.RED + "EXECUTION FAILED: " + filename + Style.RESET_ALL)
        exit(1)
    execution_time = (time.perf_counter_ns() - start_time) / 1e6
    print(f"Execution took {execution_time}ms.")
    print("*" * 100)
    os.remove(f"{filename}.exe")
    return compile_time, execution_time


def time_all_output():
    print("TIMING ALL PROJECT-EULER EXAMPLES.")
    print("If everything is fine, this code exits with exit code 0.")
    print("If not, you'll see shiny red text that shows what went wrong.")
    for filename in os.listdir("./examples"):
        if filename.endswith(".hpt"):
            filename = "./examples/" + filename.split(".hpt", 1)[0]
            time_output(filename)
    total_compile_time, total_execution_time = 0, 0
    for filename in os.listdir("./project-euler"):
        if filename.endswith(".hpt"):
            filename = "./project-euler/" + filename.split(".hpt", 1)[0]
            c_t, e_t = time_output(filename)
            total_compile_time += c_t
            total_execution_time += e_t

    print("Total Compilation time: {:.2f}ms.".format(total_compile_time))
    print("Total Execution time: {:.2f}ms.".format(total_execution_time))
    print("Total time: {:.2f}ms.".format(total_execution_time + total_compile_time))


def write_all_output():
    for filename in os.listdir("./examples"):
        if filename.endswith(".hpt"):
            filename = "./examples/" + filename.split(".hpt", 1)[0]
            write_output(filename)
    for filename in os.listdir("./project-euler"):
        if filename.endswith(".hpt"):
            filename = "./project-euler/" + filename.split(".hpt", 1)[0]
            write_output(filename)


def write_output(filename):
    lines = get_output(filename)
    with open(f"{filename}-output.txt", "w") as output:
        for line in lines:
            output.writelines(line.replace("\r", "") + "\n")
        print(f"{output.name} successfully written")
    os.remove(f"{filename}.exe")


def test_all_output():
    for filename in os.listdir("./examples"):
        if filename.endswith(".hpt"):
            filename = "./examples/" + filename.split(".hpt", 1)[0]
            test_output(filename)
    for filename in os.listdir("./project-euler"):
        if filename.endswith(".hpt"):
            filename = "./project-euler/" + filename.split(".hpt", 1)[0]
            test_output(filename)
    print(Fore.GREEN + "ALL TESTS WERE SUCCESSFUL!" + Style.RESET_ALL)


def test_output(filename):
    lines = get_output(filename)
    try:
        expected_output = open(f"{filename}-output.txt", "r")
        expected_lines = expected_output.readlines()
        if len(expected_lines) != len(lines):
            print(f"TEST FAILED FOR {filename}")
            exit(1)

        for i in range(len(lines)):
            formatted_line = lines[i].replace("\r", "")
            formatted_output = expected_lines[i].replace("\r", "").replace("\n", "")
            if formatted_line != formatted_output:
                print(f"TEST FAILED FOR {filename}")
                exit(1)
    except FileNotFoundError:
        print("FILE NOT FOUND! Please make sure to build the output first.")
        exit(1)
    os.remove(f"{filename}.exe")
    print(f"{filename}.hpt successfully tested")


def get_output(filename):
    text = ""
    try:
        exit_code = call_cmd(RUN_CMD(filename))
        if exit_code != 0:
            print(f"NON ZERO EXIT CODE {exit_code} in {filename}")
            exit(1)
        text = subprocess.check_output([f"{filename}.exe"])
    except subprocess.CalledProcessError as e:
        print(f"NON ZERO EXIT CODE {e.returncode} in {filename}")
        print(f"{e.output}")
        exit(1)
    except FileNotFoundError:
        print("FILE NOT FOUND! Please make sure to build the output first.")
        exit(1)
    formatted_text = text.decode("utf-8")
    lines = formatted_text.split("\n")
    return lines


def split(argv: List[str]):
    if len(argv) < 1:
        print("ERROR: Not enough arguments!")
        exit(1)
    return argv[0], argv[1:]


def print_usage(program_name):
    print(f"Usage: {program_name} mode=[build/test/time] file=[all/path] (path=optional)\n"
          f"  mode:\n"
          f"    build: generates expected output for file\n"
          f"    test:  compares output with expected output\n"
          f"    time:  times execution and compilation speed\n"
          f"  file:\n"
          f"    all:   Runs the specified mode for all programs in `./examples/` and `./project-euler/`\n"
          f"    path:  Runs the specified mode for the specified program")


def print_error(error: str, program_name: str = "test.py"):
    print(f"ERROR: {error}")
    print_usage(program_name)
    exit(1)


def main() -> None:
    print(sys.argv)
    argv = sys.argv
    if len(argv) < 3:
        print_error(f"`{argv[0]}` expects at least 2 arguments, found {len(argv) - 1}")
    program_name, argv = split(argv)
    mode: str = ""
    file: str = ""
    path: str = ""
    for arg in argv:
        if arg.startswith("mode="):
            mode = arg.split("mode=", 1)[1]
        elif arg.startswith("file="):
            file = arg.split("file=", 1)[1]
        elif arg.startswith("path="):
            path = arg.split("path=", 1)[1]
            path = path.split(".hpt", 1)[0]

    if mode == "":
        print_error("No mode selected")
    if file == "":
        print_error("No file selected")
    if file == "path" and path == "":
        print_error("No path specified")
    if mode not in ["build", "test", "time"]:
        print_error(f"Unknown mode: {mode}")
    if file not in ["all", "path"]:
        print_error(f"Unknown mode: {file}")

    if mode == "build":
        if file == "all":
            write_all_output()
        elif file == "path":
            assert path is not None
            write_output(path)
    elif mode == "test":
        if file == "all":
            test_all_output()
        elif file == "path":
            assert path is not None
            test_output(path)
    elif mode == "time":
        if file == "all":
            time_all_output()
        elif file == "path":
            assert path is not None
            total_compile_time, total_execution_time = time_output(path)
            print("Total Compilation time: {:.2f}ms.".format(total_compile_time))
            print("Total Execution time: {:.2f}ms.".format(total_execution_time))


if __name__ == "__main__":
    main()
