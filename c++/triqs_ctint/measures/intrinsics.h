#pragma once

#if defined(_MSC_VER)
// For Microsoft Visual Studio
#define RESTRICT __restrict
#elif defined(__GNUC__)
// For GCC and Clang
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif
