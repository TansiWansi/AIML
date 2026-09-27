#include "tree.h"
#include "../arena.h"
#include "../logging.h"

DecisionTree *fit(const f32 * restrict X, const i32 * restrict y, const u32 numSamples, const u32 numFeatures, const u32 numClasses, const u32 maxDepth, const u32 minSamplesSplit){

    // Hard guard against register shifts
    if (maxDepth > 30) {
        LOG_EXT(LOG_FILE, LOG_FATAL, "Max depth is astronomically high (%u). ABI is corrupted!", maxDepth);
        exit(1);
    }
	
	DecisionTree * restrict tree = initTree(maxDepth, minSamplesSplit);
	u32 * restrict indices = (u32 *)malloc(numSamples * sizeof(u32));
	
	for(u32 i = 0; i < numSamples; i++){
		indices[i] = i;
	}
	
	size_t requiredBytes = numSamples * (maxDepth + 1) * 2 * sizeof(u32);
    Arena *scratch = arenaInit(requiredBytes);

	BuildContext ctx = {
		
		.count 		 = numSamples,
		.numFeatures = numFeatures,
		.depth 		 = 0,
		.arena 		 = scratch
	};

	buildTreeRecursive(tree, X, y, indices, &ctx);
	
	arenaDestroy(scratch);
	free(indices);
	return tree;
}


static i32 traverseTree(const DecisionTree * restrict tree, const f32 * restrict x, const i32 nodeIdx){

	if(nodeIdx < 0 || (u32)nodeIdx >= tree->nodeCount) return 0;

	TreeNode *node = &tree->nodes[nodeIdx];

	if(node->featureIdx == TREE_LEAF_NODE){
		return node->value;
	}

	if (x[node->featureIdx] <= node->threshold) {
        if (node->leftChild == -1) return node->value;
        return traverseTree(tree, x, node->leftChild);
    }
	else if (node->rightChild == -1) return node->value;
	return traverseTree(tree, x, node->rightChild);

}


void predict(const DecisionTree * restrict tree, f32 * X, i32 * restrict yPred, const u32 numSamples, const u32 numFeatures){
	
	if(!tree) return;

	for(u32 i = 0; i < numSamples; i++){
		
		f32 *x = &X[i * numFeatures];

		yPred[i] = traverseTree(tree, x, 0);
	}
}

void freeDecisionTree(DecisionTree * restrict tree){
	
	freeTree(tree);
}
