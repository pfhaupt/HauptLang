import subprocess
import os
from typing import List
import colorama
from colorama import Fore, Style
import time

colorama.init()

# python -m timeit -s "import subprocess" "subprocess.call([\"python\", \"test.py\"])"
# runs the test script multiple times and reports the average time
# Take it with a grain of salt, it has some overhead (~10% from what I've seen)


def call_cmd(cmd: List):
    print(Fore.CYAN + "[CMD] " + " ".join(cmd) + Style.RESET_ALL)
    return subprocess.call(cmd)


def run_file(filename: str):
    print("Running " + filename)
    start_time = time.perf_counter_ns()
    exitcode = call_cmd(["python", "haupt.py", filename, "-c", "-o", "-m"])
    print("Compiling " + filename + " exited with exit code " + str(exitcode))
    if exitcode != 0:
        print(Fore.RED + "COMPILATION FAILED: " + filename + Style.RESET_ALL)
        exit(1)
    compile_time = (time.perf_counter_ns() - start_time) / 1e6
    print(f"Compilation took {compile_time}ms.")
    start_time = time.perf_counter_ns()
    exitcode = call_cmd([f"{filename.replace('.hpt', '.exe')}"])
    print("Executing " + filename + " exited with exit code " + str(exitcode))
    if exitcode != 0:
        print(Fore.RED + "EXECUTION FAILED: " + filename + Style.RESET_ALL)
        exit(1)
    execution_time = (time.perf_counter_ns() - start_time) / 1e6
    print(f"Execution took {execution_time}ms.")
    print("*" * 100)
    os.remove(f"{filename.replace('.hpt', '.exe')}")
    return compile_time, execution_time


def main():
    print("TESTING ALL PROJECT-EULER EXAMPLES.")
    print("If everything is fine, this code exits with exit code 0.")
    print("If not, you'll see shiny red text that shows what went wrong.")

    for filename in os.listdir("./examples"):
        if filename.endswith(".hpt"):
            filename = "./examples/" + filename
            run_file(filename)
    # exit(1)
    total_compile_time, total_execution_time = 0, 0
    for filename in os.listdir("./project-euler"):
        if filename.endswith(".hpt"):
            filename = "./project-euler/" + filename
            c_t, e_t = run_file(filename)
            total_compile_time += c_t
            total_execution_time += e_t

    print("Total Compilation time: {:.2f}ms.".format(total_compile_time))
    print("Total Execution time: {:.2f}ms.".format(total_execution_time))
    print("Total time: {:.2f}ms.".format(total_execution_time + total_compile_time))
    print(Fore.GREEN + "All Tests passed!" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
