#include "tree.h"


DecisionTree *initTree(const u32 maxDepth, const u32 minSamplesSplit){


	DecisionTree *tree = (DecisionTree *)malloc(sizeof(DecisionTree));
	if(!tree){
		printf("FATAL : Failed to allocate tree\n");
		exit(1);
	}

	tree->maxDepth = maxDepth;
	tree->minSamplesSplit = minSamplesSplit;
	tree->maxNodes = calculateMaxNodes(maxDepth);
	tree->nodeCount = 0;

	tree->nodes = (TreeNode *)malloc(tree->maxNodes * sizeof(TreeNode));

	if(!tree->nodes){
		printf("FATAL : Failed to allocate tree\n");
		free(tree);
		exit(1);
	}

	for (u32 i = 0; i < tree->maxNodes; i++) {
        tree->nodes[i].featureIdx = -1;
        tree->nodes[i].leftChild = -1;
        tree->nodes[i].rightChild = -1;
        tree->nodes[i].value = 0;
    }

	return tree;	
}


void freeTree(DecisionTree * restrict tree){
	
	if(tree){
		if(tree->nodes){
			free(tree->nodes);
		}
		free(tree);
	}
}
