#ifndef TREE_H
#define TREE_H

#include <stdint.h>	
#include <stdlib.h>	
#include <stdio.h>	
#include <math.h>	
#include "../arena.h"
#include "../logging.h"

#define u32 uint32_t
#define i32 int32_t 
#define f32 float

#define TREE_LEAF_NODE -1


typedef struct{
	
	i32 featureIdx; // -1 for leaf 
	f32 threshold;
	i32 leftChild;
	i32 rightChild;
	i32 value;
}TreeNode;

typedef struct{
	
	TreeNode *nodes;
	u32 maxNodes;
	u32 nodeCount;
	u32 maxDepth;
	u32 minSamplesSplit;
}DecisionTree;

typedef struct{

	u32 count;
	u32 numFeatures;
	u32 numClasses;
	u32 depth;
	Arena *arena;
}BuildContext;

static inline u32 calculateMaxNodes(const u32 maxDepth){
	
	// 2^(depth + 1) - 1
	return (1 << (maxDepth + 1)) - 1;
}

DecisionTree *initTree(const u32 maxDepth, const u32 minSamplesSplit);
void freeTree(DecisionTree * restrict tree);

i32 buildTreeRecursive(DecisionTree * restrict tree, const f32 * restrict X, const i32 * restrict y, const u32 * restrict indices, BuildContext * restrict ctx);

#endif
