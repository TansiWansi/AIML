
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


#ifndef ARENA_H
#define ARENA_H

typedef uint8_t u8;

#define ALIGNMENT 64

typedef struct Arena{
	
	u8 *buffer;
	size_t capacity;
	size_t offset;
}Arena;

Arena *arenaInit(const size_t capacityBytes);

void *arenaAlloc(Arena * restrict a, const size_t sizeBytes);

void *arenaCalloc(Arena * restrict a, const size_t sizeBytes);

void arenaReset(Arena * restrict a);

void arenaDestroy(Arena *a);

#endif
