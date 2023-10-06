/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Simple example to show how to use bitcomp's native lossy API to compress
// floating point data.
//
// Bitcomp's lossy compression performs an on-the-fly integer quantization
// and compresses the resulting integral values with the lossless encoder.
// A smaller delta used for the quantization will typically lower the
// compression ratio, but will increase precision.

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <native/bitcomp.h>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw;                                                                   \
    }                                                                          \
  } while (0);

#define BITCOMP_CHECK(call)                                                    \
  {                                                                            \
    bitcompResult_t err = call;                                                \
    if (BITCOMP_SUCCESS != err) {                                              \
      fprintf(                                                                 \
          stderr,                                                              \
          "Bitcomp error %d in file '%s' in line %i.\n",                       \
          err,                                                                 \
          __FILE__,                                                            \
          __LINE__);                                                           \
      fflush(stderr);                                                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

size_t getFileSize(const std::string filename)
{
  std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
  return static_cast<size_t>(in.tellg());
}

template <typename T>
T* read_binary_to_new_array(const std::string& fname, size_t dtype_len)
{
  std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
  if (not ifs.is_open()) {
    std::cerr << "fail to open " << fname << std::endl;
    exit(1);
  }
  auto _a = new T[dtype_len]();
  ifs.read(reinterpret_cast<char*>(_a), std::streamsize(dtype_len * sizeof(T)));
  ifs.close();
  return _a;
}

int main(int argc, char* argv[])
{
  std::string fileName = argv[1];
  float errorBound = std::stof(argv[2]);
  size_t fileSize = getFileSize(fileName);
  size_t datatypeLen = fileSize / sizeof(float);

  float* input = read_binary_to_new_array<float>(fileName, datatypeLen);
  float* output = (float*)malloc(fileSize);

  // Delta used for the integer quantization. we use range based error bound
  float *minimum, *maximum;
  minimum = std::min_element(input, input + datatypeLen);
  maximum = std::max_element(input, input + datatypeLen);
  float range = *maximum - *minimum;
  const float delta = 2 * range * errorBound;
  printf("Input range: %f, Using delta = %f\n", range, delta);

  cudaEvent_t compressStart, compressStop, decompressStart, decompressStop;
  cudaEventCreate(&compressStart);
  cudaEventCreate(&compressStop);
  cudaEventCreate(&decompressStart);
  cudaEventCreate(&decompressStop);

  // Let's execute all the GPU code in a non-default stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Allocate and initialize the floating point input
  float *inputGpu, *outputGpu;
  CUDA_CHECK(cudaMalloc((void**)&inputGpu, fileSize));
  CUDA_CHECK(cudaMalloc((void**)&outputGpu, fileSize));

  // copy input from host to device
  cudaMemcpy(inputGpu, input, fileSize, cudaMemcpyHostToDevice);

  // Create a bitcomp plan to compress FP32 data using a signed integer
  // quantization, since the input data contains positive and negative values.
  bitcompHandle_t plan;
  BITCOMP_CHECK(bitcompCreatePlan(
      &plan,                      // Bitcomp handle
      fileSize,                   // Size in bytes of the uncompressed data
      BITCOMP_FP32_DATA,          // Data type
      BITCOMP_LOSSY_FP_TO_SIGNED, // Compression type
      BITCOMP_DEFAULT_ALGO));     // Bitcomp algo, default or sparse

  // Query the maximum size of the compressed data (worst case scenario)
  // and allocate the compressed buffer
  size_t maxlen = bitcompMaxBuflen(fileSize);
  void* compbuf;
  CUDA_CHECK(cudaMalloc(&compbuf, maxlen));

  // Associate the bitcomp plan to the stream, otherwise the compression
  // or decompression would happen in the default stream
  BITCOMP_CHECK(bitcompSetStream(plan, stream));

  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  cudaEventRecord(compressStart, stream);
  // Compress the input data with the chosen quantization delta
  BITCOMP_CHECK(bitcompCompressLossy_fp32(plan, inputGpu, compbuf, delta));
  cudaEventRecord(compressStop, stream);
  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  // Wait for the compression kernel to finish
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Query the compressed size
  size_t compressedSize;
  BITCOMP_CHECK(bitcompGetCompressedSize(compbuf, &compressedSize));
  float compressionRatio
      = static_cast<float>(fileSize) / static_cast<float>(compressedSize);
  printf("Compression ratio = %f\n", compressionRatio);

  cudaEventRecord(decompressStart, stream);
  // Decompress the data
  BITCOMP_CHECK(bitcompUncompress(plan, compbuf, outputGpu));
  cudaEventRecord(decompressStop, stream);

  // Wait for the decompression to finish
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEventSynchronize(decompressStop);
  float compressTime = 0;
  float decompressTime = 0;
  cudaEventElapsedTime(&compressTime, compressStart, compressStop);
  cudaEventElapsedTime(&decompressTime, decompressStart, decompressStop);
  float compressionThroughput
      = static_cast<float>(fileSize) / 1024 / 1024 / compressTime;
  float decompressionThroughput
      = static_cast<float>(compressedSize) / 1024 / 1024 / decompressTime;
  printf(
      "file size: %ld\ncompressed size: %ld\nCompression time: %f "
      "s\nDecompression time: %f s\nCompression "
      "throughput: %f GB/s\nDecompression throughput: %f GB/s\nCompression "
      "ratio = %f\n",
      fileSize,
      compressedSize,
      compressTime,
      decompressTime,
      compressionThroughput,
      decompressionThroughput,
      compressionRatio);

  // write back from gpu to cpu
  cudaMemcpy(output, outputGpu, fileSize, cudaMemcpyDeviceToHost);

  // Compare the results
  float maxdiff = 0.0f;
  for (size_t i = 0; i < datatypeLen; i++)
    maxdiff = std::max(maxdiff, fabsf(output[i] - input[i]));
  printf("Max absolute difference  = %f\n", maxdiff);

  // Clean up
  BITCOMP_CHECK(bitcompDestroyPlan(plan));
  CUDA_CHECK(cudaFree(inputGpu));
  CUDA_CHECK(cudaFree(compbuf));
  CUDA_CHECK(cudaFree(outputGpu));

  free(input);
  free(output);

  return 0;
}