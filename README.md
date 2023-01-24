# HauptLang
## Overview
Haupt is a [Concatenative](https://en.wikipedia.org/wiki/Concatenative_programming_language) [Stack-Oriented](https://en.wikipedia.org/wiki/Stack-oriented_programming) programming language created to explore the process of creating a custom programming language.  
It is not intended to be fast or easy to use (this may be subject to change).

This project is inspired by [Porth](https://www.youtube.com/watch?v=8QP2fDBIxjM&list=PLpM-Dvs8t0VbMZA7wW9aR3EtBqe2kinu4).

**All features are highly experimental and might change at any time.**

## Milestones/Roadmap
- [x] Compiled into a [Windows Executable](https://en.wikipedia.org/wiki/Portable_Executable)
- [ ] Functions, Arrays, String support
- [ ] Type system (Only Integers are supported right now)
- [ ] Self-hosting
- [ ] Optimizations
- [ ] Cross-Platform support

## Content
### haupt.py
This is the heart of the language.
Usage:
> haupt.py <input.hpt> [-s | -c | -d] [optional flags]

or

> haupt.py -h

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
42 print
10 5 + print
```
```console
> haupt.py readme.hpt -c
[compilation logs]
[generated output.exe]
```
```console
> output.exe
42
15
```

### Testing
```console
> test.py
```
Runs all programs in the [./project-euler/](#project-euler) folder
and reports the exit code for both compilation and execution.  
If any program fails, the test stops and displays the error.  
Useful to see if everything is working as intended.

## Language Support
**TBD**
