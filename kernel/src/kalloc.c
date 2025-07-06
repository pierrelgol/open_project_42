/**
 * @file kalloc.c
 * @brief ESP32 kernel slab allocator implementation
 *
 * This file implements a simple but efficient slab allocator for the ESP32 kernel.
 * The allocator uses power-of-2 sized slabs to minimize fragmentation and
 * provides fast allocation/deallocation for common object sizes.
 */

#include "kalloc.h"
#include "kernel.h"
#include "std.h"

/* Global allocator state */
static kalloc_state_t g_kalloc_state;

/* Debugging and statistics */
static u32 g_alloc_count     = 0;
static u32 g_free_count      = 0;
static u32 g_total_allocated = 0;
static u32 g_total_freed     = 0;

/* Forward declarations */
static slab_header_t* kalloc_new_slab(slab_cache_t* cache);
static void           kalloc_init_slab(slab_header_t* slab, u32 object_size, u32 objects_per_slab);
static void*          kalloc_from_slab(slab_cache_t* cache);
static void           kfree_to_slab(slab_cache_t* cache, void* ptr);
static void           kalloc_validate_pointer(void* ptr);

/**
 * @brief Initialize the slab allocator
 */
int kalloc_init(u8* heap_start, usize heap_size) {
        if (!heap_start || heap_size == 0) {
                return -1;
        }

        /* Ensure heap is aligned */
        u8* aligned_start = (u8*)KALLOC_ALIGN_UP((usize)heap_start);
        heap_size         = heap_size - ((usize)aligned_start - (usize)heap_start);
        heap_start        = aligned_start;

        /* Initialize global state */
        memory_zero(&g_kalloc_state, sizeof(kalloc_state_t));
        g_kalloc_state.heap_start   = heap_start;
        g_kalloc_state.heap_end     = heap_start + heap_size;
        g_kalloc_state.heap_current = heap_start;

        /* Initialize slab caches */
        for (int i = 0; i < KALLOC_SLAB_COUNT; i++) {
                slab_cache_t* cache     = &g_kalloc_state.caches[i];
                cache->object_size      = KALLOC_SLAB_SIZE(i);
                cache->objects_per_slab = (KALLOC_PAGES_PER_SLAB * 4096 - sizeof(slab_header_t)) / cache->object_size;
                cache->free_slabs       = NULL;
                cache->full_slabs       = NULL;
                cache->total_allocated  = 0;
        }

        /* Reset statistics */
        g_alloc_count     = 0;
        g_free_count      = 0;
        g_total_allocated = 0;
        g_total_freed     = 0;

        return 0;
}

/**
 * @brief Allocate memory from the slab allocator
 */
void* kalloc(usize size) {
        if (size == 0) {
                return NULL;
        }

        /* Align size */
        size = KALLOC_ALIGN_UP(size);

        /* Check if size fits in a slab cache */
        int slab_idx = kalloc_get_slab_index(size);
        if (slab_idx >= 0) {
                /* Use slab allocator */
                slab_cache_t* cache = &g_kalloc_state.caches[slab_idx];
                void*         ptr   = kalloc_from_slab(cache);
                if (ptr) {
                        g_alloc_count++;
                        g_total_allocated += cache->object_size;
                        g_kalloc_state.total_allocated += cache->object_size;
                }
                return ptr;
        } else {
                /* Allocate directly from heap */
                u8* ptr         = g_kalloc_state.heap_current;
                u8* new_current = ptr + size;

                if (new_current > g_kalloc_state.heap_end) {
                        return NULL; /* Out of memory */
                }

                g_kalloc_state.heap_current = new_current;
                g_alloc_count++;
                g_total_allocated += size;
                g_kalloc_state.total_allocated += size;
                return ptr;
        }
}

/**
 * @brief Free memory allocated by kalloc
 */
void kfree(void* ptr) {
        if (!ptr) {
                return;
        }

        /* Validate pointer */
        kalloc_validate_pointer(ptr);

        /* Try to find which slab cache this pointer belongs to */
        for (int i = 0; i < KALLOC_SLAB_COUNT; i++) {
                slab_cache_t* cache = &g_kalloc_state.caches[i];

                /* Check if pointer is in any slab of this cache */
                slab_header_t* slab = cache->free_slabs;
                while (slab) {
                        if (ptr >= (void*)slab->data && ptr < (void*)(slab->data + cache->objects_per_slab * cache->object_size)) {
                                kfree_to_slab(cache, ptr);
                                g_free_count++;
                                g_total_freed += cache->object_size;
                                g_kalloc_state.total_freed += cache->object_size;
                                return;
                        }
                        slab = slab->next;
                }

                slab = cache->full_slabs;
                while (slab) {
                        if (ptr >= (void*)slab->data && ptr < (void*)(slab->data + cache->objects_per_slab * cache->object_size)) {
                                kfree_to_slab(cache, ptr);
                                g_free_count++;
                                g_total_freed += cache->object_size;
                                g_kalloc_state.total_freed += cache->object_size;
                                return;
                        }
                        slab = slab->next;
                }
        }

        /* If not found in any slab cache, it might be a direct heap allocation */
        /* For simplicity, we'll just mark it as freed without actually reclaiming the memory */
        /* @TODO : finish working on this implementation */
        g_free_count++;
        g_total_freed += 8;
        g_kalloc_state.total_freed += 8;
}

/**
 * @brief Get allocation statistics
 */
void kalloc_stats(u32* total_allocated, u32* total_freed, u32* heap_used) {
        if (total_allocated) {
                *total_allocated = g_total_allocated;
        }
        if (total_freed) {
                *total_freed = g_total_freed;
        }
        if (heap_used) {
                *heap_used = (u32)(g_kalloc_state.heap_current - g_kalloc_state.heap_start);
        }
}

/**
 * @brief Get detailed allocator statistics
 */
void kalloc_detailed_stats(u32* alloc_count, u32* free_count, u32* slab_allocated, u32* heap_allocated) {
        if (alloc_count) {
                *alloc_count = g_alloc_count;
        }
        if (free_count) {
                *free_count = g_free_count;
        }
        if (slab_allocated) {
                *slab_allocated = g_kalloc_state.total_allocated;
        }
        if (heap_allocated) {
                *heap_allocated = (u32)(g_kalloc_state.heap_current - g_kalloc_state.heap_start);
        }
}

/**
 * @brief Get the slab index for a given size
 */
int kalloc_get_slab_index(usize size) {
        if (size > KALLOC_MAX_SLAB_SIZE) {
                return -1;
        }

        /* Find the smallest slab that can fit the size */
        for (int i = 0; i < KALLOC_SLAB_COUNT; i++) {
                if ((usize)KALLOC_SLAB_SIZE(i) >= size) {
                        return i;
                }
        }

        return -1;
}

/**
 * @brief Allocate a new slab for a cache
 */
slab_header_t* kalloc_new_slab(slab_cache_t* cache) {
        usize slab_size = KALLOC_PAGES_PER_SLAB * 4096;
        u8*   slab_mem  = g_kalloc_state.heap_current;

        /* Ensure alignment */
        slab_mem = (u8*)KALLOC_ALIGN_UP((usize)slab_mem);

        if (slab_mem + slab_size > g_kalloc_state.heap_end) {
                return NULL; /* Out of memory */
        }

        g_kalloc_state.heap_current = slab_mem + slab_size;

        slab_header_t* slab         = (slab_header_t*)slab_mem;
        kalloc_init_slab(slab, cache->object_size, cache->objects_per_slab);

        return slab;
}

/**
 * @brief Initialize a new slab
 */
static void kalloc_init_slab(slab_header_t* slab, u32 object_size, u32 objects_per_slab) {
        (void)object_size;
        slab->next       = NULL;
        slab->free_count = objects_per_slab;

        /* Initialize bitmap - all objects are free initially */
        for (int i = 0; i < 8; i++) {
                slab->bitmap[i] = 0xFFFFFFFF;
        }

        /* Clear any objects beyond the actual count */
        u32 total_bits   = objects_per_slab;
        u32 bitmap_words = (total_bits + 31) / 32;

        for (u32 i = bitmap_words; i < 8; i++) {
                slab->bitmap[i] = 0;
        }

        /* Clear the last word if needed */
        if (total_bits % 32 != 0) {
                u32 last_word           = bitmap_words - 1;
                u32 bits_in_last_word   = total_bits % 32;
                slab->bitmap[last_word] = (1U << bits_in_last_word) - 1;
        }
}

/**
 * @brief Allocate an object from a slab cache
 */
static void* kalloc_from_slab(slab_cache_t* cache) {
        slab_header_t* slab = cache->free_slabs;

        /* If no free slabs, allocate a new one */
        if (!slab) {
                slab = kalloc_new_slab(cache);
                if (!slab) {
                        return NULL;
                }
                cache->free_slabs = slab;
        }

        /* Find a free object in the slab */
        u32 object_index = 0;
        b8  found        = false;

        for (int i = 0; i < 8 && !found; i++) {
                if (slab->bitmap[i] != 0) {
                        /* Find the first set bit */
                        for (int j = 0; j < 32; j++) {
                                if (slab->bitmap[i] & (1U << j)) {
                                        object_index = i * 32 + j;
                                        slab->bitmap[i] &= ~(1U << j);
                                        found = true;
                                        break;
                                }
                        }
                }
        }

        if (!found) {
                return NULL; /* Should not happen if free_count > 0 */
        }

        /* Calculate object address */
        void* obj = slab->data + (object_index * cache->object_size);

        /* Update slab state */
        slab->free_count--;
        cache->total_allocated++;

        /* If slab is now full, move it to full_slabs list */
        if (slab->free_count == 0) {
                cache->free_slabs = slab->next;
                slab->next        = cache->full_slabs;
                cache->full_slabs = slab;
        }

        return obj;
}

/**
 * @brief Free an object back to a slab cache
 */
static void kfree_to_slab(slab_cache_t* cache, void* ptr) {
        /* Find which slab this object belongs to */
        slab_header_t* slab = cache->free_slabs;
        slab_header_t* prev = NULL;

        /* Search in free slabs */
        while (slab) {
                if (ptr >= (void*)slab->data && ptr < (void*)(slab->data + cache->objects_per_slab * cache->object_size)) {
                        break;
                }
                prev = slab;
                slab = slab->next;
        }

        /* If not found in free slabs, search in full slabs */
        if (!slab) {
                slab = cache->full_slabs;
                prev = NULL;

                while (slab) {
                        if (ptr >= (void*)slab->data && ptr < (void*)(slab->data + cache->objects_per_slab * cache->object_size)) {
                                break;
                        }
                        prev = slab;
                        slab = slab->next;
                }
        }

        if (!slab) {
                return; /* Object not found in any slab */
        }

        /* Calculate object index */
        u32 object_index = ((u8*)ptr - slab->data) / cache->object_size;

        /* Update bitmap */
        u32 bitmap_word = object_index / 32;
        u32 bitmap_bit  = object_index % 32;
        slab->bitmap[bitmap_word] |= (1U << bitmap_bit);

        /* Update slab state */
        slab->free_count++;
        cache->total_allocated--;

        /* If slab was full and now has free space, move it to free_slabs */
        if (slab->free_count == 1) {
                /* Remove from full_slabs */
                if (prev) {
                        prev->next = slab->next;
                } else {
                        cache->full_slabs = slab->next;
                }

                /* Add to free_slabs */
                slab->next        = cache->free_slabs;
                cache->free_slabs = slab;
        }
}

/**
 * @brief Validate pointer before freeing
 */
static void kalloc_validate_pointer(void* ptr) {
        /* Check if pointer is in heap range */
        if (ptr < (void*)g_kalloc_state.heap_start || ptr >= (void*)g_kalloc_state.heap_end) {
                kernel_panic("segmentation fault\n");
                return;
        }

        /* Check if pointer is aligned */
        if (((usize)ptr & (KALLOC_ALIGNMENT - 1)) != 0) {
                kernel_panic("segmentation fault\n");
                return;
        }
}
