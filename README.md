<h2 align="center">⚒️ CUDA by Example ⚒️</h2>
<h4 align="center">An Introduction to General-Purpose GPU Programming</h4>

<table align="center">
  <tr>
    <td align="center" style="padding: 5px"><strong>Build System</strong></td>
    <td align="center" style="padding: 5px">
      <a href="https://cmake.org/">
        <img
          src="https://img.shields.io/badge/CMake-064F8C?style=flat&logo=cmake&logoColor=white"
      /></a>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding: 5px"><strong>Languages</strong></td>
    <td align="center" style="padding: 5px">
      <a href="https://isocpp.org/">
        <img
          src="https://img.shields.io/badge/C++-00599C?style=flat&logo=cplusplus&logoColor=white"
      /></a>
      <a href="https://developer.nvidia.com/cuda-toolkit" target="_blank">
        <img
          src="https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white"
        />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding: 5px"><strong>Compilers</strong></td>
    <td align="center" style="padding: 5px">
      <a href="https://clang.llvm.org/" target="_blank">
        <img
          src="https://img.shields.io/badge/Clang-262D3A?style=flat&logo=llvm&logoColor=white"
        />
      </a>
      <a
        href="https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/"
        target="_blank"
      >
        <img
          src="https://img.shields.io/badge/NVCC-76B900?style=flat&logo=nvidia&logoColor=white"
        />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" style="padding: 5px">
      <strong>Infrastructure</strong>
    </td>
    <td align="center" style="padding: 5px">
      <a href="https://llvm.org/" target="_blank">
        <img
          src="https://img.shields.io/badge/LLVM-262D3A?style=flat&logo=llvm&logoColor=white"
        />
      </a>
    </td>
  </tr>
</table>

<h3 align="center">👷 Setup 👷</h3>

```bash
$ make docker
docker run \
-d --name gpu \
-v ./:/workspace \
-w /workspace \
zerohertzkr/gpu
566c42b4cc083fff284fd35b518465552699ebda67162987d755fb812da51c9a
```

```bash
$ make exec
docker exec -it gpu zsh
```

```bash
$ make init
...
```

```bash
$ make rm
docker rm -f gpu
gpu
```

<h3 align="center">⚙️ Build & Run ⚙️</h3>

<table align="center">
  <tr>
    <th>Clang++</th>
    <th>NVCC</th>
  </tr>
  <tr>
    <td>
      <pre lang="bash"><code>$ make clang
rm -rf build && \
cmake -S . -B build \
-DUSE_NVCC=OFF \

-- Using Clang++ compiler
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is Clang 18.1.3
-- The CUDA compiler identification is Clang 18.1.3
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/clang++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/bin/clang++ - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found CUDAToolkit: /usr/local/cuda/include (found version "12.6.85")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Found OpenGL: /usr/lib/x86_64-linux-gnu/libOpenGL.so
-- Configuring done (8.2s)
-- Generating done (0.0s)
-- Build files have been written to: /workspace/build
</code></pre>

</td>
<td>
<pre lang="bash"><code>$ make nvcc
rm -rf build && \
cmake -S . -B build \
-DUSE_NVCC=ON \

-- Using NVCC compiler
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is Clang 18.1.3
-- The CUDA compiler identification is NVIDIA 12.6.85
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/clang++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found CUDAToolkit: /usr/local/cuda/targets/x86_64-linux/include (found version "12.6.85")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Found OpenGL: /usr/lib/x86_64-linux-gnu/libOpenGL.so
-- Configuring done (8.1s)
-- Generating done (0.0s)
-- Build files have been written to: /workspace/build
</code></pre>

</td>

  </tr>
  <tr>
    <td>
      <pre lang="bash"><code>$ cd build
$ make 01_hello_world && ./01_hello_world
[ 50%] Building CUDA object CMakeFiles/01_hello_world.dir/src/chapter03/01_hello_world.cu.o
clang++: warning: CUDA version is newer than the latest supported version 12.3 [-Wunknown-cuda-version]
[100%] Linking CUDA executable 01_hello_world
[100%] Built target 01_hello_world
[2025-09-03 23:28:58] [src/chapter03/01_hello_world.cu:5] ℹ️ Hello, World!
[2025-09-03 23:28:58] [src/chapter03/01_hello_world.cu:6] ⚠️ Hello, World!
[2025-09-03 23:28:58] [src/chapter03/01_hello_world.cu:7] ❌ Hello, World!
</code></pre>
    </td>
    <td>
      <pre lang="bash"><code>$ cd build
$ make 01_hello_world && ./01_hello_world
[ 50%] Building CUDA object CMakeFiles/01_hello_world.dir/src/chapter03/01_hello_world.cu.o
[100%] Linking CUDA executable 01_hello_world
[100%] Built target 01_hello_world
[2025-09-03 23:20:13] [src/chapter03/01_hello_world.cu:5] ℹ️ Hello, World!
[2025-09-03 23:20:13] [src/chapter03/01_hello_world.cu:6] ⚠️ Hello, World!
[2025-09-03 23:20:13] [src/chapter03/01_hello_world.cu:7] ❌ Hello, World!
</code></pre>
    </td>
  </tr>
  <tr>
    <td>
      <pre lang="bash"><code>$ make clang DEBUG=1
rm -rf build && \
cmake -S . -B build \
-DUSE_NVCC=OFF \
-DENABLE_CUDA_DEBUG=ON
...
$ cd build
$ make 11_dot_product && cuda-gdb ./11_dot_product
[ 50%] Building CUDA object CMakeFiles/11_dot_product.dir/src/chapter05/11_dot_product.cu.o
clang++: warning: CUDA version is newer than the latest supported version 12.3 [-Wunknown-cuda-version]
[100%] Linking CUDA executable 11_dot_product
[100%] Built target 11_dot_product
NVIDIA (R) cuda-gdb 12.6
...
Reading symbols from ./11_dot_product...
(cuda-gdb) b main
Breakpoint 1 at 0xbf86: file /workspace/src/chapter05/11_dot_product.cu, line 41.
(cuda-gdb) r
Starting program: /workspace/build/11_dot_product
warning: Error disabling address space randomization: Operation not permitted
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, main () at /workspace/src/chapter05/11*dot_product.cu:41
41 std::unique_ptr<float[]> a, b, partial_c;
(cuda-gdb) list
36 c[blockIdx.x] = cache[0];
37 }
38
39 int main(void)
40 {
41 std::unique_ptr<float[]> a, b, partial_c;
42 float c;
43 float*dev_a,*dev_b, \_dev_partial_c;
44
...
49 CUDA_CHECK(cudaMalloc((void \*\*)&dev_a, N* sizeof(float)));
(cuda-gdb) n
[New Thread 0x7fbfb4ee3000 (LWP 85577)]
[New Thread 0x7fbfaf522000 (LWP 85578)]
[Detaching after fork from child process 85579]
[New Thread 0x7fbfaca85000 (LWP 85586)]
[2025-09-03 23:32:58] [src/chapter05/11_dot_product.cu:49] ✅ cudaMalloc((void **)&dev_a, N \* sizeof(float))
50 CUDA_CHECK(cudaMalloc((void**)&dev_b, N \* sizeof(float)));
$ nsys profile --stats=true ./11_dot_product
Collecting data...
...
Time (%) Total Time (ns) Num Calls Avg (ns) Med (ns) Min (ns) Max (ns) StdDev (ns) Name

---

     56.3    1,892,847,558      1,658   1,141,645.1      18,479.5     1,055  220,450,458    7,900,221.3  ioctl
     41.2    1,383,336,538         16  86,458,533.6  10,093,906.5     3,558  964,746,540  236,532,591.2  poll
      1.3       43,752,207        121     361,588.5       6,372.0     2,732    8,126,082    1,496,006.8  open64

...
Generated:
/workspace/build/report1.nsys-rep
/workspace/build/report1.sqlite
</code></pre>

</td>
<td>
<pre lang="bash"><code>$ make nvcc DEBUG=1
rm -rf build && \
cmake -S . -B build \
-DUSE_NVCC=ON \
-DENABLE_CUDA_DEBUG=ON
...
$ cd build
$ make 11_dot_product && cuda-gdb ./11_dot_product
[ 50%] Building CUDA object CMakeFiles/11_dot_product.dir/src/chapter05/11_dot_product.cu.o
[100%] Linking CUDA executable 11_dot_product
[100%] Built target 11_dot_product
NVIDIA (R) cuda-gdb 12.6
...
Reading symbols from ./11_dot_product...
(cuda-gdb) b main
Breakpoint 1 at 0xbf43: file /workspace/src/chapter05/11_dot_product.cu, line 40.
(cuda-gdb) r
Starting program: /workspace/build/11_dot_product
warning: Error disabling address space randomization: Operation not permitted
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, main () at /workspace/src/chapter05/11*dot_product.cu:40
40 {
(cuda-gdb) list
35 if (cacheIndex == 0)
36 c[blockIdx.x] = cache[0];
37 }
38
39 int main(void)
40 {
41 std::unique_ptr<float[]> a, b, partial_c;
42 float c;
43 float*dev_a,*dev_b, \_dev_partial_c;
44
...
49 CUDA_CHECK(cudaMalloc((void \*\*)&dev_a, N* sizeof(float)));
(cuda-gdb) n
[New Thread 0x7f6214443000 (LWP 84317)]
[New Thread 0x7f6213160000 (LWP 84318)]
[Detaching after fork from child process 84319]
[New Thread 0x7f6208f9c000 (LWP 84326)]
[2025-09-03 23:24:15] [src/chapter05/11_dot_product.cu:49] ✅ cudaMalloc((void \**)&dev_a, N* sizeof(float))
$ nsys profile --stats=true ./11_dot_product
Collecting data...
...
Time (%) Total Time (ns) Num Calls Avg (ns) Med (ns) Min (ns) Max (ns) StdDev (ns) Name

---

     50.1    1,761,121,669         22  80,050,985.0  35,298,208.0     3,314  758,247,341  157,132,669.5  poll
     48.5    1,705,147,300      1,652   1,032,171.5      12,548.0     1,019  210,198,255    7,771,683.0  ioctl
      0.7       23,606,803        121     195,097.5       4,625.0     1,629   10,492,489    1,167,185.7  open64

...
Generated:
/workspace/build/report1.nsys-rep
/workspace/build/report1.sqlite
</code></pre>

</td>

  </tr>
</table>

---

<h3 align="center">📖 Study Info 📖</h3>

- [Book](https://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685)
- [Udemy](https://www.udemy.com/course/cuda-course/)

<details><summary>
    <a href="https://www.cyberseowon.com/forum/teugbyeolmoim/2025nyeon-9weol-membeosib-cuda-gpu-peurogeuraeming-ibmunban">
        Study Info
    </a>
</summary>
<p>

[📢 GPU/CUDA 프로그래밍을 배워보세요! 북클럽 나란에서 두 가지 레벨의 스터디를 모집합니다. 📢](https://www.linkedin.com/posts/sungjuc_gpucuda-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%EC%8A%A4%ED%84%B0%EB%94%94-%EC%B0%B8%EA%B0%80-%EC%8B%A0%EC%B2%AD%ED%8F%BC-activity-7361337346170003457-rZhS?utm_source=share&utm_medium=member_desktop&rcm=ACoAADxDAfgBHmsGgos6Xqn5TZXS7NqO4fcxRGc)

최근 AI 열풍으로 AI Engineering 분야도 상당히 뜨겁습니다.
AI Engineering에서 GPU/CUDA 아키텍쳐는 반드시 알아야할 핵심 지식입니다.
이에 NVIDIA GPU와 Google Colab을 활용해 실습 중심으로 GPU 병렬 프로그래밍을 통해 GPU/CUDA 아키텍쳐도 배우고 GPU/CUDA 프로그래밍에 필요한 지식들을 배우는 스터디 그룹을 만들었습니다.

스터디 그룹은 입문반과 중급반으로 나누었습니다.
필요에 따라 원하는 레벨을 선택하여 참여할 수 있습니다.

- 입문반:
  - 2025.09.06 마감 (9/6 시작, 8주간)
  - CUDA 기본 개념, 스레드/블록 모델, 간단한 커널 작성, 메모리 관리, 이미지 처리 프로젝트
  - 상세정보: <https://docs.google.com/document/d/1dPeaSxsZEgKrfE9RS4gxf54b801p35Hw84DY0fT7c-k/edit?pli=1&tab=t.0#bookmark=id.6omg6jr8xiln>
- 중급반:
  - 2025.11.10 마감 (11/15 시작, 12주간)
  - 메모리 최적화, 워프 다이버전스 최소화, 멀티-GPU 활용, Nsight 성능 분석, 행렬 곱셈 프로젝트
  - 상세정보: <https://docs.google.com/document/d/1dPeaSxsZEgKrfE9RS4gxf54b801p35Hw84DY0fT7c-k/edit?pli=1&tab=t.qjg678960psu#heading=h.110u61emtje2>
- 참가비:
  - 각 모임당 참가비는 동기부여를 위해 $50을 받고, 모임을 끝까지 마무리하시면 환불해드리는 기본적으로 무료입니다.
  - 책 값과 Udemy 강의비용은 개별 부담입니다.
- GPU/CUDA 프로그래밍 스터디는 북클럽 나란에서 지원합니다.
  - 북클럽 나란에 대한 자세한 정보는 웹사이트를 참고하세요.
  - 북클럽 나란 웹사이트: <https://www.cyberseowon.com>
- 스터디 그룹에 대한 정보가 필요하신 분들은 다음 카톡 방에 문의해주시거나 메일로 문의해주세요.
  - 상담 카톡방: <https://open.kakao.com/o/gitHslMh>
  - 상담 이메일: <admin@cyberseowon.com>

참가를 원하시는 분들은 [참가 신청폼](https://docs.google.com/forms/d/e/1FAIpQLSf6ADaK-RAz_YlZLIamdQx-_UiqUXSP8TSKbIt3ZbBuglyCjQ/viewform)을 작성해주세요.
참가 신청폼을 작성해주신 분들에게는 자세한 참가 방법을 보내드립니다.
(정원이 초과하는 경우 신청폼의 내용을 기반으로 참가지를 선정할 수 있습니다.)

🚀 GPU 프로그래밍 세계에 도전하고 싶은 분들의 많은 참여를 기다립니다!

</p>
</details>
