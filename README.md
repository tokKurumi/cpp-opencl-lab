# Requirements

## Arch Linux (NVIDIA GPU)

main opencl packages:

```bash
pacman -Q | grep -E "opencl|ocl|pocl"
```

```txt
ocl-icd 2.3.4-1
opencl-nvidia 580.105.08-5
pocl 7.1-3
```

# NVIDIA GeForce RTX 3060 test results

```bash
nvidia-smi
```

```txt
Wed Dec 17 01:13:06 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        On  |   00000000:07:00.0  On |                  N/A |
| 30%   38C    P5             19W /  170W |    1789MiB /  12288MiB |     74%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1036      G   Hyprland                                 84MiB |
|    0   N/A  N/A            1071      G   /usr/bin/quickshell                     123MiB |
|    0   N/A  N/A            1098      G   Xwayland                                  2MiB |
|    0   N/A  N/A            1178      G   /usr/lib/xdg-desktop-portal-kde           2MiB |
|    0   N/A  N/A           20565      G   ...rack-uuid=3190708988185955192         42MiB |
|    0   N/A  N/A           20796      G   ...asma-browser-integration-host          2MiB |
|    0   N/A  N/A           24505      G   /opt/discord/Discord                    648MiB |
|    0   N/A  N/A           24608      G   /proc/self/exe                            2MiB |
|    0   N/A  N/A           26012      G   /usr/lib/kactivitymanagerd                2MiB |
|    0   N/A  N/A          224521      G   /opt/visual-studio-code/code            418MiB |
|    0   N/A  N/A          224920      G   kitty                                   122MiB |
+-----------------------------------------------------------------------------------------+
```

## Tests

### 512x512 grid, 100 iterations

```bash
./src/build/Release/opencl-lab 512 100
```

```txt
[2025-12-17 01:32:13.653] [info] ========================================
[2025-12-17 01:32:13.653] [info] Jacobi Benchmark: Grid 512x512, 100 iterations
[2025-12-17 01:32:13.653] [info] ========================================
[2025-12-17 01:32:13.653] [info]
[2025-12-17 01:32:13.778] [info] JacobiGpuGlobal: Initialized with NVIDIA GeForce RTX 3060
[2025-12-17 01:32:13.778] [info] --- Running Jacobi with Global Memory ---
[2025-12-17 01:32:13.780] [info] JacobiGpuGlobal: 100 iterations in 1 ms
[2025-12-17 01:32:13.780] [info] Global Memory: Completed 100 iterations in 2 ms
[2025-12-17 01:32:13.845] [info]
[2025-12-17 01:32:13.935] [info] JacobiGpuLocal: OpenCL initialized with device: NVIDIA GeForce RTX 3060
[2025-12-17 01:32:13.935] [info] --- Running Jacobi with Local (Shared) Memory ---
[2025-12-17 01:32:13.937] [info] JacobiGpuLocal: Completed 100 iterations in 2 ms
[2025-12-17 01:32:13.937] [info] Local (Shared) Memory: Completed 100 iterations in 2 ms
[2025-12-17 01:32:14.000] [info]
[2025-12-17 01:32:14.092] [info] JacobiGpuTexture: Initialized with NVIDIA GeForce RTX 3060
[2025-12-17 01:32:14.092] [info] --- Running Jacobi with Texture Memory ---
[2025-12-17 01:32:14.093] [info] JacobiGpuTexture: 100 iterations in 0 ms
[2025-12-17 01:32:14.093] [info] Texture Memory: Completed 100 iterations in 0 ms
[2025-12-17 01:32:14.151] [info]
[2025-12-17 01:32:14.151] [info] ========================================
[2025-12-17 01:32:14.151] [info] Benchmark Complete
[2025-12-17 01:32:14.151] [info] ========================================
```

### 1024x1024 grid, 1000 iterations

```bash
./src/build/Release/opencl-lab 1024 1000
```

```txt
[2025-12-17 01:33:00.010] [info] ========================================
[2025-12-17 01:33:00.010] [info] Jacobi Benchmark: Grid 1024x1024, 1000 iterations
[2025-12-17 01:33:00.010] [info] ========================================
[2025-12-17 01:33:00.010] [info]
[2025-12-17 01:33:00.168] [info] JacobiGpuGlobal: Initialized with NVIDIA GeForce RTX 3060
[2025-12-17 01:33:00.168] [info] --- Running Jacobi with Global Memory ---
[2025-12-17 01:33:00.221] [info] JacobiGpuGlobal: 1000 iterations in 53 ms
[2025-12-17 01:33:00.221] [info] Global Memory: Completed 1000 iterations in 53 ms
[2025-12-17 01:33:00.296] [info]
[2025-12-17 01:33:00.399] [info] JacobiGpuLocal: OpenCL initialized with device: NVIDIA GeForce RTX 3060
[2025-12-17 01:33:00.399] [info] --- Running Jacobi with Local (Shared) Memory ---
[2025-12-17 01:33:00.451] [info] JacobiGpuLocal: Completed 1000 iterations in 51 ms
[2025-12-17 01:33:00.451] [info] Local (Shared) Memory: Completed 1000 iterations in 52 ms
[2025-12-17 01:33:00.521] [info]
[2025-12-17 01:33:00.620] [info] JacobiGpuTexture: Initialized with NVIDIA GeForce RTX 3060
[2025-12-17 01:33:00.620] [info] --- Running Jacobi with Texture Memory ---
[2025-12-17 01:33:00.624] [info] JacobiGpuTexture: 1000 iterations in 3 ms
[2025-12-17 01:33:00.624] [info] Texture Memory: Completed 1000 iterations in 3 ms
[2025-12-17 01:33:00.698] [info]
[2025-12-17 01:33:00.698] [info] ========================================
[2025-12-17 01:33:00.698] [info] Benchmark Complete
[2025-12-17 01:33:00.698] [info] ========================================
```

### 2560x2560 grid, 10000 iterations

```bash
./src/build/Release/opencl-lab 2560 10000
```

```txt
[2025-12-17 01:35:01.315] [info] ========================================
[2025-12-17 01:35:01.315] [info] Jacobi Benchmark: Grid 2560x2560, 10000 iterations
[2025-12-17 01:35:01.315] [info] ========================================
[2025-12-17 01:35:01.315] [info]
[2025-12-17 01:35:01.432] [info] JacobiGpuGlobal: Initialized with NVIDIA GeForce RTX 3060
[2025-12-17 01:35:01.432] [info] --- Running Jacobi with Global Memory ---
[2025-12-17 01:35:03.158] [info] JacobiGpuGlobal: 10000 iterations in 1726 ms
[2025-12-17 01:35:03.159] [info] Global Memory: Completed 10000 iterations in 1727 ms
[2025-12-17 01:35:03.223] [info]
[2025-12-17 01:35:03.307] [info] JacobiGpuLocal: OpenCL initialized with device: NVIDIA GeForce RTX 3060
[2025-12-17 01:35:03.307] [info] --- Running Jacobi with Local (Shared) Memory ---
[2025-12-17 01:35:05.597] [info] JacobiGpuLocal: Completed 10000 iterations in 2289 ms
[2025-12-17 01:35:05.598] [info] Local (Shared) Memory: Completed 10000 iterations in 2290 ms
[2025-12-17 01:35:05.670] [info]
[2025-12-17 01:35:05.769] [info] JacobiGpuTexture: Initialized with NVIDIA GeForce RTX 3060
[2025-12-17 01:35:05.769] [info] --- Running Jacobi with Texture Memory ---
[2025-12-17 01:35:05.801] [info] JacobiGpuTexture: 10000 iterations in 32 ms
[2025-12-17 01:35:05.801] [info] Texture Memory: Completed 10000 iterations in 32 ms
[2025-12-17 01:35:05.884] [info]
[2025-12-17 01:35:05.884] [info] ========================================
[2025-12-17 01:35:05.884] [info] Benchmark Complete
[2025-12-17 01:35:05.884] [info] ========================================
```
