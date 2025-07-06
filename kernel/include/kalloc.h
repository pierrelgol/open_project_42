#ifndef __KALLOC_H__
#define __KALLOC_H__

#include "types.h"

/**
 * @file kalloc.h
 * @brief ESP32 kernel slab allocator
 *
 * Simple but efficient slab allocator for the ESP32 kernel.
 * Uses power-of-2 sized slabs to minimize fragmentation and
 * provides fast allocation/deallocation for common object sizes.
 */

/* Slab allocator configuration */
#define KALLOC_MAX_SLAB_SIZE    2048    /* Maximum slab size in bytes */
#define KALLOC_MIN_SLAB_SIZE    8       /* Minimum slab size in bytes */
#define KALLOC_SLAB_COUNT       8       /* Number of different slab sizes */
#define KALLOC_PAGES_PER_SLAB   4       /* Number of pages per slab */

/* Calculate slab sizes: 8, 16, 32, 64, 128, 256, 512, 1024, 2048 bytes */
#define KALLOC_SLAB_SIZE(slab_idx) (KALLOC_MIN_SLAB_SIZE << (slab_idx))

/* Memory alignment */
#define KALLOC_ALIGNMENT        8
#define KALLOC_ALIGN_UP(addr)   (((addr) + KALLOC_ALIGNMENT - 1) & ~(KALLOC_ALIGNMENT - 1))

/* Slab header structure */
typedef struct slab_header {
    struct slab_header* next;    /* Next slab in the list */
    u32 free_count;             /* Number of free objects in this slab */
    u32 bitmap[8];              /* Bitmap for tracking free objects (256 bits) */
    u8 data[];                  /* Start of object data area */
} slab_header_t;

/* Slab cache structure */
typedef struct slab_cache {
    slab_header_t* free_slabs;   /* List of slabs with free objects */
    slab_header_t* full_slabs;   /* List of slabs that are full */
    u32 object_size;            /* Size of objects in this cache */
    u32 objects_per_slab;       /* Number of objects per slab */
    u32 total_allocated;        /* Total number of allocated objects */
} slab_cache_t;

/* Main allocator structure */
typedef struct kalloc_state {
    slab_cache_t caches[KALLOC_SLAB_COUNT];  /* Array of slab caches */
    u8* heap_start;                          /* Start of heap memory */
    u8* heap_end;                            /* End of heap memory */
    u8* heap_current;                        /* Current heap pointer */
    u32 total_allocated;                     /* Total allocated memory */
    u32 total_freed;                         /* Total freed memory */
} kalloc_state_t;

/**
 * @brief Initialize the slab allocator
 * 
 * Sets up the slab allocator with the given heap memory region.
 * 
 * @param heap_start Start address of heap memory
 * @param heap_size Size of heap memory in bytes
 * @return 0 on success, -1 on failure
 */
int kalloc_init(u8* heap_start, usize heap_size);

/**
 * @brief Allocate memory from the slab allocator
 * 
 * Allocates memory of the requested size. For small objects (<= KALLOC_MAX_SLAB_SIZE),
 * uses the appropriate slab cache. For larger objects, allocates from the heap.
 * 
 * @param size Size of memory to allocate in bytes
 * @return Pointer to allocated memory, or NULL if allocation failed
 */
void* kalloc(usize size);

/**
 * @brief Free memory allocated by kalloc
 * 
 * Returns memory to the appropriate slab cache or heap.
 * 
 * @param ptr Pointer to memory to free
 */
void kfree(void* ptr);

/**
 * @brief Get allocation statistics
 * 
 * Returns information about the current state of the allocator.
 * 
 * @param total_allocated Pointer to store total allocated bytes
 * @param total_freed Pointer to store total freed bytes
 * @param heap_used Pointer to store current heap usage
 */
void kalloc_stats(u32* total_allocated, u32* total_freed, u32* heap_used);

/**
 * @brief Get detailed allocator statistics
 * 
 * Returns detailed information about allocation patterns.
 * 
 * @param alloc_count Pointer to store number of allocations
 * @param free_count Pointer to store number of frees
 * @param slab_allocated Pointer to store slab-allocated bytes
 * @param heap_allocated Pointer to store heap-allocated bytes
 */
void kalloc_detailed_stats(u32* alloc_count, u32* free_count, u32* slab_allocated, u32* heap_allocated);

/**
 * @brief Get the slab index for a given size
 * 
 * Returns the appropriate slab cache index for the given allocation size.
 * 
 * @param size Size in bytes
 * @return Slab cache index, or -1 if size is too large
 */
int kalloc_get_slab_index(usize size);



#endif /* __KALLOC_H__ */ 