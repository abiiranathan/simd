#ifndef SIMD_DETECTION_H
#define SIMD_DETECTION_H

/*
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#define __MMX_WITH_SSE__ 1
#define __SSE2_MATH__ 1
#define __AVX__ 1
#define __SSE_MATH__ 1
#define __AVX2__ 1
#define __SSSE3__ 1
#define __SSE__ 1
#define __SSE2__ 1
#define __SSE3__ 1
*/

// Compilation flags to enable SIMD
// For GCC/Clang:
// -mavx2 -mfma       (enables AVX2)
// -msse4.1           (enables SSE4.1)
// For MSVC:
// /arch:AVX2         (enables AVX2)
// /arch:SSE4.1       (enables SSE4.1)

#endif  // SIMD_DETECTION_H