#include "engine.h"


//// ------ DenseLayer START ------- ////

// "constructor"
DenseLayer *newDenseLayer(Arena * restrict arena, const u32 inSize, const u32 outSize, const f32 l2lambda){
	
	DenseLayer *layer = (DenseLayer *)arenaAlloc(arena, sizeof(DenseLayer));
	if(layer == NULL) return NULL;

	*layer = (DenseLayer){
	
		.inputs		= NULL, 
		.l2lambda 	= l2lambda,
		.timeStep 	= 0,

		.w 			= createMatrix(inSize, outSize),
		.wM 		= createMatrix(inSize, outSize),
		.wV 		= createMatrix(inSize, outSize), 
		.b 			= createMatrix(1, outSize), 
		.bM 		= createMatrix(1, outSize), 
		.bV 		= createMatrix(1, outSize), 

		.forward 	= forwardDenseLayer,
		.backward 	= backwardDenseLayer,
		.delete 	= deleteDenseLayer
	};
	
	
	f32 scale = sqrtf(2.0f / (f32)inSize);
	u32 totalWeights = inSize * outSize;

	for(u32 i = 0; i < totalWeights; i++){
	
		f32 randVal = ((f32)rand() / (f32)RAND_MAX) * 2.0f - 1.0f;
		layer->w->data[i] = randVal * scale;
	}

	return layer;
}

// .forward() "method"
Matrix *forwardDenseLayer(DenseLayer * restrict l, const Matrix * restrict inputs, Arena * restrict arena){
	
	l->inputs = inputs;

	
	Matrix *outputs = createArenaMatrix(arena, inputs->rows, l->w->cols);

	matDot(inputs , l->w, outputs);
	matAdd(outputs, l->b, outputs);

	return outputs;
}


Matrix *backwardDenseLayer(DenseLayer * restrict l, const Matrix * restrict outGrad, const f32 lr, Arena * restrict arena){
	
	Matrix *wGrad 	= createArenaMatrix(arena, l->w->rows, l->w->cols); 
	Matrix *bGrad 	= createArenaMatrix(arena, 1, l->b->cols);
	Matrix *inGrad 	= createArenaMatrix(arena, outGrad->rows, l->inputs->cols);
	
	// wGrad = dot( inputs.T, outGrad )
	// Casted to Matrix * to bypass const definition
	((Matrix *)l->inputs)->flags |= MAT_TRANSPOSED;	// inputs.T
	matDot(l->inputs, outGrad, wGrad);	// wGrad = dot(inputs.T, outGrad)
	((Matrix *)l->inputs)->flags &= ~MAT_TRANSPOSED;// inputs
	
	// bGrad = sum(outGrad)
	matColSum(outGrad, bGrad);

	// inGrad = dot(outGrad, w.T)
	l->w->flags |= MAT_TRANSPOSED;	// w.T
	matDot(outGrad, l->w, inGrad);	// inGrad = dot(outGrad, w.T)
	l->w->flags &= ~MAT_TRANSPOSED;	// w

	// ADAM
	l->timeStep += 1;
	matAdamUpdate(l->w, wGrad, l->wM, l->wV, lr, BETA_M, BETA_V, EPSILON, l->l2lambda, l->timeStep);
	matAdamUpdate(l->b, bGrad, l->bM, l->bV, lr, BETA_M, BETA_V, EPSILON, 0.0f, l->timeStep); // No l2lambda.
	
	return inGrad;
}


// "destructor"
void deleteDenseLayer(DenseLayer **layer){

	if(layer == NULL || *layer == NULL) return;
	
	DenseLayer *l = *layer;

    freeMatrix(l->w); freeMatrix(l->wM); freeMatrix(l->wV);
    freeMatrix(l->b); freeMatrix(l->bM); freeMatrix(l->bV);

    *layer= NULL;
}

// ----- DenseLayer end -----



// ----- LeakyReluLayer START ----



// "constructor"
LeakyReluLayer *newLeakyReluLayer(Arena * restrict arena, const u32 inSize){

	LeakyReluLayer *layer = (LeakyReluLayer *)arenaAlloc(arena, sizeof(LeakyReluLayer));

	*layer = (LeakyReluLayer){
		
		.inputs 	= NULL,
		.inSize 	= inSize,
		.forward 	= forwardLeakyReluLayer,
		.backward 	= backwardLeakyReluLayer,
		.delete 	= deleteLeakyReluLayer
	};

	return layer;
}

Matrix *forwardLeakyReluLayer(LeakyReluLayer * restrict l, const Matrix * restrict inputs, const f32 leak, Arena * restrict arena){
	
	l->inputs = inputs;

	Matrix *outputs = createArenaMatrix(arena, inputs->rows, inputs->cols);
	
	for(u32 i = 0; i < l->inSize; i++){
		outputs->data[i] = (inputs->data[i] < 0.0f)? inputs->data[i] * leak : inputs->data[i];
	}

	return outputs;
}


Matrix *backwardLeakyReluLayer(LeakyReluLayer * restrict l, const Matrix * restrict outGrad, const f32 leak, Arena * restrict arena){
	
	Matrix *inGrad = createArenaMatrix(arena, outGrad->rows, outGrad->cols);
	
	u32 total = outGrad->rows * outGrad->cols;

	for(u32 i = 0; i < total; i++){
		inGrad->data[i] = outGrad->data[i] * ((l->inputs->data[i] < 0)? leak : 1);
	}

	return inGrad;
}

void deleteLeakyReluLayer(LeakyReluLayer **layer){
	
	if(layer == NULL || *layer == NULL) return;

	LeakyReluLayer *l = *layer;
	free(l);
	layer = NULL;
}
// ----- LeakyReluLayer END -----



// ----- Layers START -----


Architecture *newArchitecture(Arena * restrict arena, const u32 * restrict layerSizes, const u32 numLayers, const f32 l2lambda){
	
	Architecture *a = (Architecture *)arenaAlloc(arena, sizeof(Architecture));
	*a = (Architecture){
	
		.layerSizes = (u32 *)arenaAlloc(arena, sizeof(u32) * numLayers),
		.numLayers = numLayers,
		.l2lambda = l2lambda
	};

	for(u32 i = 0; i < numLayers; i++){
		a->layerSizes[i] = layerSizes[i];
	}

	return a;
}



// ----- Layers END -----



// ----- NeuralNetwork START -----



// "constructor"

NeuralNetwork *newNeuralNetwork(Arena * restrict arena, const Architecture * restrict arch){
	
	NeuralNetwork *nn = (NeuralNetwork *)arenaAlloc(arena, sizeof(NeuralNetwork));

	*nn = (NeuralNetwork){
		
		.numLayers 	= (arch->numLayers * 2) - 3,
		.layers 	= (Layer *)arenaAlloc(arena, sizeof(Layer) * ((arch->numLayers * 2) - 3)),
		.predict	= predictNeuralNetwork,
		.train		= trainNeuralNetwork
	};
	
	u32 layerIdx = 0;

    for(u32 i = 0; i < arch->numLayers - 1; i++){

        u32 inSize = arch->layerSizes[i];
        u32 outSize = arch->layerSizes[i + 1];

        nn->layers[layerIdx].type = DENSE_LAYER;
        nn->layers[layerIdx++].layer.dense = newDenseLayer(arena, inSize, outSize, arch->l2lambda);
        if(i < arch->numLayers - 2){

            nn->layers[layerIdx].type = LEAKY_RELU_LAYER;
            nn->layers[layerIdx++].layer.leakyRelu = newLeakyReluLayer(arena, outSize);
        }
    }	

	return nn;
}

Matrix *predictNeuralNetwork(NeuralNetwork * restrict nn, const Matrix * restrict inputs, Arena * restrict arena){

	// Casted to discard const
	Matrix *outputs = (Matrix *)inputs;

	for(u32 i = 0; i < nn->numLayers; i++){
		
		Layer l = nn->layers[i];

		switch(l.type){

		case DENSE_LAYER:
			outputs = l.layer.dense->forward(l.layer.dense, outputs, arena);	
			break;

		case LEAKY_RELU_LAYER:
			outputs = l.layer.leakyRelu->forward(l.layer.leakyRelu, outputs, 0.01f, arena);	
			break;
			
		}
	}

	return outputs;
}

Matrix *_getBatch(const Matrix * restrict data, const u32 startIdx, const u32 batchSize, Arena * restrict arena){
	
	u32 actualSize 	= MIN(batchSize, data->rows - startIdx);
	Matrix *batch 	= createArenaMatrix(arena, actualSize, data->cols);

	u32 offset 	= startIdx * data->cols;
	u32 bytes 	= actualSize * data->cols * sizeof(f32);

	memcpy(batch->data, data->data + offset, bytes);

	return batch;
}

void trainNeuralNetwork(NeuralNetwork * restrict nn, const Matrix * restrict xTrain, const Matrix * restrict yTrain, const u32 epochs, const f32 lr, const u32 batchSize, Arena * restrict arena){
	

	u32 numSamples =  xTrain->rows;

	for(u32 epoch = 0; epoch < epochs; epoch++){
		for(u32 start = 0; start < numSamples; start += batchSize){
			
			// Just for testing, print the first batch execution
            
           
			Matrix *xBatch = _getBatch(xTrain, start, batchSize, arena);
			Matrix *yBatch = _getBatch(yTrain, start, batchSize, arena);
			
			// Forward pass
			Matrix *predictions = nn->predict(nn, xBatch, arena);

			// Loss
			Matrix *lossGrad = createArenaMatrix(arena, predictions->rows, predictions->cols);

			// MSE
			for(int32_t i = 0; i < lossGrad->rows * lossGrad->cols; i++){
				lossGrad->data[i] = (predictions->data[i] - yBatch->data[i]) * (2.0f / predictions->rows);
			}

			// Backward Pass
			Matrix *inGrad = lossGrad;

			for(int i = nn->numLayers - 1; i >= 0; i--){

				Layer l = nn->layers[i];
				
				switch(l.type){
		
				case DENSE_LAYER:
					inGrad = l.layer.dense->backward(l.layer.dense, inGrad, lr, arena);	
					break;
				
				case LEAKY_RELU_LAYER:
					inGrad = l.layer.leakyRelu->backward(l.layer.leakyRelu, inGrad, 0.01f, arena);	
					break;
			
				}

			}
			
			arenaReset(arena);
		}

		if((epoch + 1) % 10 == 0){ 
			printf("Epoch %u\n", epoch + 1); 
			fflush(stdout);
	}
}

	printf("\nTraining Complete!\n");
}
