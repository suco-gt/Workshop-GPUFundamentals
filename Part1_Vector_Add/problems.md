# Problem Background

For each of the following problems, you'll be writing the core logic of the GPU Kernel function and decide how data streams & threads will be organized. This sounds complicated, but is not too hard (as far as these examples go!). 

You'll be writing Python code that closely parallels the algorithmic approach that you would use with low-level ``C`` code written for CUDA. Some of the later examples contain material that we'll discuss later on in the problem session, so don't worry too much if you don't already know the relevant methods.

**Warning** This code looks like Python but it is really CUDA! You cannot use standard python tools like list comprehensions or ask for Numpy properties like shape or size (if you need the size, it is given as an argument). The puzzles only require doing simple operations, basically +, *, simple array indexing, for loops, and if statements. You are allowed to use local variables. If you get an error it is probably because you did something fancy.

*Tip: Think of the function call as being run 1 time for each thread. The only difference is that ``cuda.threadIdx.x`` changes each time.*

Although this is not relevant until later, recall that a **global read** occurs when you read an array from memory, and similarly a **global write** occurs when you write an array to memory. Ideally, you should reduce the extent to which you perform ``global`` operations, so that your code will have minimal overhead. Additionally, our grader (may) throw an error if you do something not nice!

Credits to [Sasha Rush](https://github.com/srush/GPU-Puzzles/tree/main) for the problems. Similar problems are available at the link.

# Grading

For each problem, you can complete the code directly in ``nano`` or a code editor of your choice. Afterwards, once you've logged onto a node that contains a NVIDIA GPU that is CUDA-capable, you may run your code for the **n**th question as follows.
```
python q[n]_file.py
```
Make sure to not remove the ```lib.py``` file, as this contains some of the driver code for our grader!

# Problems! 

1. Implement a kernel that adds 10 to each position of vector ``a`` and stores it in vector out. You have 1 thread per position.

2. Implement a kernel that adds together each position of ``a`` and ``b`` and stores it in out. You have 1 thread per position.