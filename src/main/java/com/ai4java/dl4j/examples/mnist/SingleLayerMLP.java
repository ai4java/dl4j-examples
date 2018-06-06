package com.ai4java.dl4j.examples.mnist;


import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/** A Multi Layered Perceptron (MLP) with a single hidden layer,
 * applied to the digit classification task of the MNIST Dataset
 * 
 * Based on the corresponding example from DL4J:
 * github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistSingleLayerExample.java
 * 
 * For more information see: ai4java.com
 */
public class SingleLayerMLP {

    public static void main(String[] args) throws Exception {

    	final int rngSeed = 123; // random number seed for reproducibility

    	final int numOfInputs = 28 * 28;  // numRows * numColumns in the input images
        final int numOfOutputs = 10;      // number of output classes ('0'..'9')
        final int hiddenLayerSize = 1000; // number of nodes in hidden layer

        final int batchSize = 125;//128; // batch size for each epoch
        final int numEpochs = 15; // number of epochs to perform

        // prepare the data sets:
        DataSetIterator trainSet = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator testSet = new MnistDataSetIterator(batchSize, false, rngSeed);

        // crate, train and evaluate the MLP:
        SingleLayerMLP mlp = new SingleLayerMLP(rngSeed, numOfInputs, numOfOutputs, hiddenLayerSize);
        mlp.create();
        mlp.train(trainSet, numEpochs);
        mlp.evaluate(testSet);
    }

    private static Logger log = LoggerFactory.getLogger(SingleLayerMLP.class);

    private final int rngSeed;
    private final int numOfInputs;
    private final int numOfOutputs;
    private final int hiddenLayerSize;
    
    private MultiLayerNetwork mlp;
   
    public SingleLayerMLP(int rngSeed, int numOfInputs, int numOfOutputs, int hiddenLayerSize) {
		this.rngSeed = rngSeed;
		this.numOfInputs = numOfInputs;
		this.numOfOutputs = numOfOutputs;
		this.hiddenLayerSize = hiddenLayerSize;
	}

	public void create() {
		
    	log.info("Creating model....");
        
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                
    			.seed(rngSeed) //include a random seed for reproducibility
        
                // use stochastic gradient descent as an optimization algorithm
                .updater(new Nesterovs(0.006, 0.9))
                .l2(1e-4)
                .list()
                
                // input layer:
                .layer(0, new DenseLayer.Builder() 
                        .nIn(numOfInputs)
                        .nOut(hiddenLayerSize)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                        
                // hidden layer:        
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) 
                        .nIn(hiddenLayerSize)
                        .nOut(numOfOutputs)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                        
                .pretrain(false)
                .backprop(true) //use backpropagation to adjust weights
                
                .build();

        mlp = new MultiLayerNetwork(conf);
        mlp.init();
    }
    
	public void train(DataSetIterator trainSet, int numEpochs) {
		log.info("Training model....");
		
        // print the score every 100 iterations:
        mlp.setListeners(new ScoreIterationListener(100));
		
        for( int i=0; i<numEpochs; i++ ){
        	log.info("epoch {}....", i);
            mlp.fit(trainSet);
        }
	}
	
	public void evaluate(DataSetIterator testSet) {
		log.info("Evaluating model....");
		
        Evaluation eval = new Evaluation(numOfOutputs); // create an evaluation object with 10 possible classes
       
        int batchCounter = 0;
        while(testSet.hasNext()){
            DataSet next = testSet.next();
            INDArray output = mlp.output(next.getFeatureMatrix()); //get the networks prediction
            log.info("Evaluating next batch ({})...", batchCounter++);
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
	}
}