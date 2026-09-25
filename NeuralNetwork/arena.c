#include "arena.h"


Arena *arenaInit(const size_t capacityBytes){
	
	Arena *a = (Arena *)malloc(sizeof(Arena));
    if (a == NULL) {
        printf("FATAL : Failed to allocate Arena struct\n");
        exit(1);
    }	

	*a = (Arena){
		.capacity = capacityBytes,
		.offset = 0
	};
	
	if(posix_memalign((void **)&(a->buffer), ALIGNMENT, capacityBytes) != 0){
		printf("FATAL : Failed to allocate Arena");
		exit(1);
	}
	
	return a;
}

void *arenaAlloc(Arena * restrict a, const size_t sizeBytes){
	
	size_t alignedOffset = (a->offset + (ALIGNMENT - 1) & ~(ALIGNMENT - 1));

	if(alignedOffset + sizeBytes > a->capacity){
		printf("FATAL : Arena out of memory\nRequired : %zu\nCapacity : %zu\n", 
		alignedOffset + sizeBytes, a->capacity);
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
