#include "tree.h"
#include <string.h>

#define FLOAT_MAX_32 3.40282347e+38

static f32 calculateGini(const i32 * restrict y, const u32 * restrict indices, const u32 count, const u32 numClasses){
	
	if(count == 0) return 0.0f;

	u32 classCounts[numClasses];
	memset(classCounts, 0, numClasses * sizeof(u32));

	for(u32 i = 0; i < count; i++){	classCounts[y[indices[i]]]++; }

	f32 gini = 1.0f;
	for(u32 i = 0; i < numClasses; i++){
		f32 prob = (f32)classCounts[i] / (f32)count;
		gini -= (prob * prob);
	}

	return gini;
}


static i32 getMajorityClass(const i32 * restrict y, const u32 * restrict indices, const u32 count, const u32 numClasses){
	
	u32 classCounts[numClasses];
	memset(classCounts, 0, numClasses * sizeof(u32));

	i32 majorityClass = -1;
	u32 maxCount = 0;

	for(u32 i = 0; i < count; i++){
		u32 c = y[indices[i]];
		classCounts[c]++;
		
		if(classCounts[c] > maxCount){
			maxCount = classCounts[c];
			majorityClass = c;
		}
	}

	return majorityClass;
}


i32 buildTreeRecursive(DecisionTree * restrict tree, const f32 * restrict X, const i32 * restrict y, const u32 * restrict indices, BuildContext * restrict ctx){
	
	if(ctx->count == 0) return -1;
	

	i32 nodeIdx = tree->nodeCount++;
	TreeNode *node = &tree->nodes[nodeIdx];

	*node = (TreeNode){
		
		.featureIdx = -1,
		.leftChild = -1,
		.rightChild = -1,
		.value = getMajorityClass(y, indices, ctx->count, ctx->numClasses)
	};

	if(ctx->depth >= tree->maxDepth || ctx->count < tree->minSamplesSplit || calculateGini(y, indices, ctx->count, ctx->numClasses) == 0.0f){
		return nodeIdx;
	}

	f32 bestGini 	= FLOAT_MAX_32;
	i32 bestFeat 	= -1;
	f32 bestThresh 	= 0.0f;

	size_t stateStart = ctx->arena->offset;

	u32 *leftIdx = (u32 *)arenaAlloc(ctx->arena, ctx->count * sizeof(u32));
    u32 *rightIdx = (u32 *)arenaAlloc(ctx->arena, ctx->count * sizeof(u32));
	
	for(u32 feat = 0; feat < ctx->numFeatures; feat++){
		for(u32 i = 0; i < ctx->count; i++){
			
			f32 thresh = X[indices[i] * ctx->numFeatures + feat];

			u32 leftCount = 0;
			u32 rightCount = 0;

			for(u32 j = 0; j < ctx->count; j++){
				if(X[indices[i] * ctx->numFeatures + feat] <= thresh){
					
					leftIdx[leftCount++] = indices[j];
				}
				else{
					rightIdx[rightCount++] = indices[j];
				}
			}
		
			if(leftCount > 0 && rightCount > 0){
				
				f32 giniLeft = calculateGini(y, leftIdx, leftCount, ctx->numClasses);
				f32 giniRight = calculateGini(y, rightIdx, rightCount, ctx->numClasses);

				f32 giniSplit = ((f32)leftCount / ctx->count) * giniLeft + ((f32)rightCount / ctx->count) * giniRight;


				if(giniSplit < bestGini){
					bestGini 	= giniSplit;
					bestFeat	= feat;
					bestThresh 	= thresh;
				}

			}
		}
	}

	ctx->arena->offset = stateStart;

	if(bestFeat == -1) return nodeIdx;

	u32 *finalLeftIdx 	= (u32 *)arenaAlloc(ctx->arena, ctx->count * sizeof(u32));
	u32 *finalRightIdx 	= (u32 *)arenaAlloc(ctx->arena, ctx->count * sizeof(u32)); 
	u32 finalLeftCount 	= 0;
	u32 finalRightCount = 0;

	for(u32 i = 0; i < ctx->count; i++){
		if(X[indices[i] * ctx->numFeatures + bestFeat] <= bestThresh){
				
			finalLeftIdx[finalLeftCount++] = indices[i];
			continue;
		}
		finalRightIdx[finalRightCount++] = indices[i];
		
	}
	
	BuildContext leftCtx = *ctx;
	leftCtx.count = finalLeftCount;
	leftCtx.depth = ctx->depth + 1;

	BuildContext rightCtx = *ctx;
	rightCtx.count = finalRightCount;
	rightCtx.depth = ctx->depth + 1;

	*node = (TreeNode){
		
		.featureIdx = bestFeat,
		.threshold 	= bestThresh,
		.value		= -1,
		.leftChild 	= buildTreeRecursive(tree, X, y, finalLeftIdx, &leftCtx),
		.rightChild = buildTreeRecursive(tree, X, y, finalRightIdx, &rightCtx)
	};

	ctx->arena->offset = stateStart;

	return nodeIdx;
}
