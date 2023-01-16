# HauptLang
## Overview
This is a project I have been working on for a while.

## Milestones/Roadmap
- [x] Compiled to a native instruction set (x86-64, Windows only for now)
- [ ] Functions, Arrays, String support
- [ ] Type system (Only Integers are supported right now)
- [ ] Optimizations
- [ ] Self hosting

## Content
### haupt.py
This is the heart of the language.
Usage:
> haupt.py <input.hpt> [-s | -c | -d]
#### Flags
-s: Interpret the code in Python

-c: Compile the code and link it into a single executable

-d: For debugging purposes - Only prints the parsed instruction list to the screen

-h: Shows help


### ./project-euler/...
This folder contains implementations of a few [Project Euler](https://projecteuler.net/) problems in this language.

**The solutions are only presented to show the capabilities of this language! Don't spoil yourself by looking at them before you solved the problem yourself!**

## Quick Start
### Compilation
Compilation generates assembly code and compiles it with [nasm](https://www.nasm.us/), then links everything with [GoLink](https://www.godevtool.com/).
So make sure you have both of those tools available in your %PATH%.

```
> test.hpt
42 $var
&var print

> haupt.py test.hpt -c
[compilation logs]
[generated output.exe]

> output.exe
42
```

## Language Support

**TBD**
