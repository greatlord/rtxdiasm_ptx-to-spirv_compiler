// Host-call callbacks for unresolved PTX `.extern .func` calls.
//
// A host-call kernel is compiled as a chain of dispatches. The runtime
// stops between segments, invokes the callback registered here, and
// resumes. Callbacks run on the host thread that synchronizes the
// launch; they must not re-enter the same context/stream.
#ifndef ACUDA_HOSTCALL_H
#define ACUDA_HOSTCALL_H

#include <stddef.h>
#include <stdint.h>
#include "acuda/runtime.h"

#define ACUDA_HOST_NAME_MAX 64
// Internal request-buffer layout shared by the compiler and runtime.
#define ACUDA_HOST_REQ_STRIDE      528u
#define ACUDA_HOST_REQ_ARGS_OFF     16u
#define ACUDA_HOST_REQ_RET_OFF     272u
#define ACUDA_HOST_REQ_PTR_BIND_OFF 400u
#define ACUDA_HOST_REQ_MAX_ARGS     32u
#define ACUDA_HOST_STATE_HEADER     16u
#define ACUDA_HOST_STATE_REG_STRIDE 16u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct acuda_context_impl *acuda_context_t;

// Return 0 for success; a nonzero status aborts the remaining relaunches.
// `args` is the request argument area: one 8-byte little-endian slot per
// PTX `.extern .func` parameter, with .b8/.b16 values zero-extended. A
// pointer argument is replaced with a real host virtual address for the
// corresponding user buffer. `ret` receives the declared return bytes.
typedef int (*acuda_host_call_fn)(void *user,
                                  const uint8_t *args, size_t args_bytes,
                                  uint8_t *ret, size_t ret_bytes);

acuda_status_t acuda_host_call_register(acuda_context_t ctx,
                                        const char *name,
                                        acuda_host_call_fn fn,
                                        void *user);
acuda_status_t acuda_host_call_unregister(acuda_context_t ctx,
                                          const char *name);

#ifdef __cplusplus
}
#endif

#endif
