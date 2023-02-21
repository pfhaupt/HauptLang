# HauptLang
## Overview
Haupt is a [Concatenative](https://en.wikipedia.org/wiki/Concatenative_programming_language) [Stack-Oriented](https://en.wikipedia.org/wiki/Stack-oriented_programming) programming language created to explore the process of creating a custom programming language.  
It is not intended to be fast or easy to use (this may be subject to change).

This project is inspired by [Porth](https://www.youtube.com/watch?v=8QP2fDBIxjM&list=PLpM-Dvs8t0VbMZA7wW9aR3EtBqe2kinu4).

**All features are highly experimental and might change at any time.**

## Milestones/Roadmap
- [x] Compiled into a [Windows Executable](https://en.wikipedia.org/wiki/Portable_Executable)
- [x] Typed (Strings, Integers, Pointers)
- [x] Functions
- [ ] Self-hosting
- [ ] Optimizations

## Content
### haupt.py
This is the heart of the language.
Usage:
> python haupt.py input=<input.hpt> [flags]

or

> python haupt.py help=all

to show more options.

### ./project-euler/...
This folder contains implementations of a few [Project Euler](https://projecteuler.net/) problems in this language.

**The solutions are only presented to show the capabilities of this language! Don't spoil yourself by looking at them before you solved the problem yourself!**

## Quick Start
### Compilation
Compilation generates assembly code and compiles it with [nasm](https://www.nasm.us/), then links everything with [GoLink](https://www.godevtool.com/).  
So make sure you have both of those tools available in your %PATH%.

```console
> readme.hpt
proc main do
  10 5 + puti "\n" puts drop
  "Hello World!\n" puts drop
end
```
```console
> python haupt.py input=readme.hpt
[compilation logs]
[INFO] Created ./readme.exe
```
```console
> readme.exe
15
Hello World!
```

### Testing
To use the test script, run the following command:
```console
> python test.py mode=[build|test|time] file=[all|path] path=[optional]
```
#### Modes

The `mode` argument specifies the desired operation mode:

- `build`: generates expected output files for each specified file.
- `test`: runs specified files and compares output with the expected files.
- `time`: measures compilation and execution time for each specified file.
#### Files

The `file` argument specifies which files to test:

- `all`: tests all `.hpt` files in the `./examples/` and `./project-euler/` directories.
- `path`: allows you to test a specific `.hpt` file by specifying the file path.

#### Path

The `path` argument is optional and should be used to specify the path to a single `.hpt` file that you wish to test. Do not use this argument to specify directories, as it will only work for individual files. It will also be ignored unless you specify `file=path`.

## Language Support
**TBD**
