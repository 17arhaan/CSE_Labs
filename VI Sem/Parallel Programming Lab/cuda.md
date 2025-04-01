
### **1. Install CUDA Toolkit (if not installed)**
If you haven’t installed CUDA, install it using:
```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```
Check installation with:
```bash
nvcc --version
```
If installed, you will see something like:
```
nvcc: NVIDIA (R) Cuda compiler
release 11.x, V11.x.x
```

---

### **2. Compile a CUDA Program**
Use the **NVIDIA CUDA Compiler (`nvcc`)** to compile a CUDA program. 

**Command:**
```bash
nvcc -o my_program my_program.cu
```
- `-o my_program`: Specifies the output file name.
- `my_program.cu`: Your CUDA source file.

Example:
```bash
nvcc -o convolution convolution.cu
```
This will create an **executable file** named `convolution`.

---

### **3. Run the Compiled Program**
After compiling, run the executable:
```bash
./my_program
```
Example:
```bash
./convolution
```

---

### **4. Compile with Debugging Enabled (Optional)**
If you want to debug, use:
```bash
nvcc -G -g -o my_program my_program.cu
```
- `-G`: Enables debugging support.
- `-g`: Includes debugging information.

Run the program with `cuda-gdb`:
```bash
cuda-gdb ./my_program
```

---

### **5. Compile with Optimization (Optional)**
For optimized performance, use:
```bash
nvcc -O3 -o my_program my_program.cu
```
- `-O3`: Maximum optimization.

---

### **6. Check GPU Availability**
Before running, ensure CUDA detects your GPU:
```bash
nvidia-smi
```
It should display information about your NVIDIA GPU.

---

### **Common Errors and Fixes**
- **"nvcc: command not found"** → Ensure CUDA Toolkit is installed.
- **"cuda.h: No such file or directory"** → Install `nvidia-cuda-toolkit` or check CUDA installation.

---
