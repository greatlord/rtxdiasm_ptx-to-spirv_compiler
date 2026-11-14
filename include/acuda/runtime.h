// ACuda public status codes and error strings. C ABI.
#ifndef ACUDA_RUNTIME_H
#define ACUDA_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum acuda_status {
    ACUDA_SUCCESS = 0,
    ACUDA_ERROR_UNINITIALIZED = 1,
    ACUDA_ERROR_INVALID_VALUE = 2,
    ACUDA_ERROR_OUT_OF_MEMORY = 3,
    ACUDA_ERROR_NO_DEVICE = 4,
    ACUDA_ERROR_NOT_SUPPORTED = 5,
    ACUDA_ERROR_INTERNAL = 6,
    ACUDA_ERROR_INVALID_HANDLE = 7,
    ACUDA_ERROR_NOT_READY = 8,
    ACUDA_ERROR_TIMEOUT = 9
} acuda_status_t;

const char *acuda_strerror(acuda_status_t status);

#ifdef __cplusplus
}
#endif

#endif
