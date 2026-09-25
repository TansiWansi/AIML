#ifndef ENGINE_H
#define ENGINE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <string.h>
#include "mathlib.h"


typedef struct DenseLayer{
		
	const Matrix *inputs;
	// ADAM optimizer
	Matrix 	*w;
	Matrix 	*b;
	Matrix 	*wM;
	Matrix 	*wV;
	Matrix 	*bM;
	Matrix 	*bV;

	// L2 Regularization
	f32 l2lambda;
	u32 timeStep;
	
	// "Methods"
	Matrix 	*(*forward) (struct DenseLayer * restrict, const Matrix * restrict, Arena * restrict);
	Matrix 	*(*backward)(struct DenseLayer * restrict, const Matrix * restrict, const f32, Arena * restrict);
	void 	(*delete)(struct DenseLayer **);
}DenseLayer;

DenseLayer *newDenseLayer(Arena * restrict arena, const u32 inSize, const u32 outSize, const f32 l2lambda);
Matrix *forwardDenseLayer(DenseLayer * restrict l, const Matrix * restrict inputs, Arena * restrict arena);
Matrix *backwardDenseLayer(DenseLayer * restrict l, const Matrix * restrict outGrad, const f32 lr, Arena * restrict arena);
void deleteDenseLayer(DenseLayer **layer);


typedef struct LeakyReluLayer{

	const Matrix *inputs;
	u32 inSize;

	// "methods"
	Matrix 	*(*forward)(struct LeakyReluLayer * restrict, const Matrix * restrict, const f32, Arena * restrict);
	Matrix 	*(*backward)(struct LeakyReluLayer * resrtict, const Matrix * restrict, const f32, Arena * restrict);
	void 	(*delete)(struct LeakyReluLayer **);
}LeakyReluLayer;

LeakyReluLayer *newLeakyReluLayer(Arena * restrict arena, const u32 inSize);
Matrix *forwardLeakyReluLayer(LeakyReluLayer * restrict l, const Matrix * restrict inputs, const f32 leak, Arena * restrict arena);
Matrix *backwardLeakyReluLayer(LeakyReluLayer * restrict l, const Matrix * restrict outGrad, const f32 leak, Arena * restrict arena);
void deleteLeakyReluLayer(LeakyReluLayer **layer);


typedef enum LayerType{

	DENSE_LAYER,
	LEAKY_RELU_LAYER
}LayerType;

typedef struct Layer{

	LayerType type;
	union{

		DenseLayer *dense;
		LeakyReluLayer *leakyRelu;
	}layer;

}Layer;

typedef struct Architecture{

	u32 *layerSizes;
	u32  numLayers;
	f32  l2lambda;
}Architecture;

Architecture *newArchitecture(Arena * restrict arena, const u32 * restrict layerSizes, const u32 numLayers, const f32 l2lambda);



typedef struct NeuralNetwork{

	Layer  *layers;
	u32 	numLayers;

	Matrix *(*predict)(struct NeuralNetwork * restrict, const Matrix * restrict, Arena * restrict);
	void    (*train)(struct NeuralNetwork * restrict, const Matrix * restrict, const Matrix * restrict, const u32, const f32, const u32, Arena * restrict);

}NeuralNetwork;

NeuralNetwork *newNeuralNetwork(Arena * restrict arena, const Architecture * restrict arch);
Matrix *predictNeuralNetwork(NeuralNetwork * restrict nn, const Matrix * restrict inputs, Arena * restrict arena);
void trainNeuralNetwork(NeuralNetwork * restrict nn, const Matrix * restrict xTrain, const Matrix * restrict yTrain, const u32 epochs, const f32 lr, const u32 batchSize, Arena * restrict arena);

#endif
