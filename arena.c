#include "arena.h"


Arena *arenaInit(const size_t capacityBytes){
	
	Arena *a = (Arena *)malloc(sizeof(Arena));
    if (a == NULL) {
        LOG_EXT(LOG_THREAD | LOG_FILE, LOG_FATAL, "Failed to allocate Arena struct");
        exit(1);
    }	

	a->capacity = capacityBytes;
	a->offset = 0;
	
	if(posix_memalign((void **)&(a->buffer), ALIGNMENT, capacityBytes) != 0){
		LOG_EXT(LOG_THREAD | LOG_FILE, LOG_FATAL, "Failed to allocate Arena");
		exit(1);
	}
	
	return a;
}

void *arenaAlloc(Arena * restrict a, const size_t sizeBytes){
	
	size_t alignedOffset = (a->offset + (ALIGNMENT - 1) & ~((size_t)(ALIGNMENT - 1)));

	if(alignedOffset + sizeBytes > a->capacity){
		LOG_EXT(LOG_THREAD | LOG_FILE, LOG_FATAL, "Arena out of memory | Required : %llu | Capacity : %llu", 
		(unsigned long long)(alignedOffset + sizeBytes),
		(unsigned long long)a->capacity);
		exit(1);
	}

	void *ptr = a->buffer + alignedOffset;
	a->offset = alignedOffset + sizeBytes;

	return ptr;
}

void *arenaCalloc(Arena * restrict a, const size_t sizeBytes){
	
	void *ptr = arenaAlloc(a, sizeBytes);

	memset(ptr, 0, sizeBytes);

	return ptr;
}

void arenaReset(Arena * restrict a){
	a->offset = 0;
}

void arenaDestroy(Arena *a){
	
	if (a != NULL) {
        if (a->buffer != NULL) {
            free(a->buffer);
            a->buffer = NULL;
        }
        free(a);
    }
}
