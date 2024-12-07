#include <cpuid.h>
#include <immintrin.h>
#include <stdio.h>
#include <xmmintrin.h>

// Check if the CPU supports AVX
int check_avx() {
  unsigned int eax, ebx, ecx, edx;
  __cpuid(1, eax, ebx, ecx, edx);
  return ecx & bit_AVX;
}

// Check if the CPU supports SSE
int check_sse() {
  unsigned int eax, ebx, ecx, edx;
  __cpuid(1, eax, ebx, ecx, edx);
  return edx & bit_SSE;
}

// Check if the CPU supports SSE4.1 which is required for SSE intrinsics
int check_sse41() {
  unsigned int eax, ebx, ecx, edx;
  __cpuid(1, eax, ebx, ecx, edx);
  return ecx & bit_SSE4_1;
}

// Check for simd support at compile time with macros
#if defined(__AVX2__)
#define AVX2_SUPPORTED 1
#else
#define AVX2_SUPPORTED 0
#endif

#if defined(__SSE__)
#define SSE_SUPPORTED 1
#else
#define SSE_SUPPORTED 0
#endif

#if defined(__SSE4_1__)
#define SSE41_SUPPORTED 1
#else
#define SSE41_SUPPORTED 0
#endif

// SSE vector addition.
void sse_vector_add(float* a, float* b, float* result, int size) {
  // Ensure size if a multiple of 4 for the 128-bit SSE registers
  for (int i = 0; i < size; i += 4) {
    // Load 4 floats from each input array into sse registers

    __m128 vec_a = _mm_loadu_ps(&a[i]);
    __m128 vec_b = _mm_loadu_ps(&b[i]);

    // Add the vectors
    __m128 vec_result = _mm_add_ps(vec_a, vec_b);

    // Store the result back to memory
    _mm_storeu_ps(&result[i], vec_result);
  }
}

// AVX vector multiplication
void avx_vector_multiply(float* a, float* b, float* result, int size) {
  // Ensure size if a multiple of 8, for 256-bit AVX registers
  for (int i = 0; i < size; i += 8) {
    // Load 8 floats from input array.
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);

    // Multiply the vectors
    __m256 vec_result = _mm256_mul_ps(vec_a, vec_b);

    _mm256_store_ps(&result[i], vec_result);
  }
}

// Dot product using AVX registers
float avx_dot_product(float* a, float* b, int size) {
  __m256 sum_vec = _mm256_setzero_ps();

  for (int i = 0; i < size; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);

    // multiply and add to sum
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec_a, vec_b));
  }

  // Horizontal sum of the vector
  __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
  __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);

  sum_low = _mm_add_ps(sum_low, sum_high);
  sum_low = _mm_hadd_ps(sum_low, sum_low);
  sum_low = _mm_hadd_ps(sum_low, sum_low);

  float result;
  _mm_store_ss(&result, sum_low);
  return result;
}

#define SIZE 32

int main(void) {
  if (!check_sse()) {
    printf("SSE is not supported\n");
    return 1;
  }

  if (!check_sse41()) {
    printf("SSE4.1 is not supported\n");
    return 1;
  }

  if (!check_avx()) {
    printf("AVX is not supported\n");
    return 1;
  }

  // compile with gcc -mavx2 -msse4.1 -O3 simd.c -o simd
  float a[SIZE] = {
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
  };

  float b[SIZE] = {
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
  };

  float result[SIZE];

  // SSE vector addition
  sse_vector_add(a, b, result, SIZE);

  printf("SSE Vector Addition\n");
  for (int i = 0; i < SIZE; i++) {
    printf("%f ", result[i]);
  }
  printf("\n");

  // AVX vector multiplication
  avx_vector_multiply(a, b, result, SIZE);

  printf("AVX Vector Multiplication\n");
  for (int i = 0; i < SIZE; i++) {
    printf("%f ", result[i]);
  }
  printf("\n");

  // AVX dot product
  float dot_product = avx_dot_product(a, b, SIZE);
  printf("AVX Dot Product: %f\n", dot_product);

  return 0;
}