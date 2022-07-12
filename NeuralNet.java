import java.util.ArrayList;
import java.util.Random;

public class NeuralNet extends SupervisedLearner {

	int HIDDEN_LAYERS = 1;
	boolean  training = true;
	String rigged = "Test";
	Random random;
	ArrayList<Double> outputs;
	ArrayList<Double> target;
	HiddenLayer hiddenLayer;
	OutputLayer outputLayer;
	
	public NeuralNet(Random rand) {
		random = rand;
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		// TODO Auto-generated method stub
		int HIDDEN_NODES_COUNT = 16, epochLimit = 10, epochTimer = 0;
		int highestAccuracy = 0;
		int OUTPUT_NODES_COUNT = labels.valueCount(0);
		boolean multitarget = false;
		if (OUTPUT_NODES_COUNT == 0)  {
			multitarget = false;
			OUTPUT_NODES_COUNT = 1;
		} else {
			multitarget = true;
		}
		
		if (rigged == "Problem") {
			HIDDEN_NODES_COUNT = 2;
			OUTPUT_NODES_COUNT = 1;
			//EPOCH_MAX = 2;
		}
		features.shuffle(this.random, labels);
		double TRAIN_PERCENT = .6;
		
		int trainSize = (int)(TRAIN_PERCENT * features.rows());
		
		int validationSize = features.rows() - trainSize;
		validationSize /= 2;
		int testSize = validationSize;
		
		features.shuffle(this.random, labels);
		
		Matrix trainFeatures = new Matrix(features, 0, 0, trainSize, features.cols());
		Matrix trainLabels = new Matrix(labels, 0, 0, trainSize, labels.cols());
		Matrix validationFeatures = new Matrix();
		Matrix 	validationLabels = new Matrix();
		Matrix testFeatures = new Matrix();
		Matrix 	testLabels = new Matrix();
		if (TRAIN_PERCENT != 1.0) {
			validationFeatures = new Matrix(features, trainSize, 0, validationSize, features.cols());
			validationLabels = new Matrix(labels, trainSize, 0, validationSize, labels.cols());
			testFeatures = new Matrix(features, trainSize + validationSize, 0, testSize,features.cols());
			testLabels = new Matrix(labels, trainSize + validationSize, 0, testSize, labels.cols());	
		}

		//First establish the nodes for hidden layer
		hiddenLayer = new HiddenLayer(HIDDEN_NODES_COUNT, trainFeatures.cols(), rigged);
		outputLayer = new OutputLayer(OUTPUT_NODES_COUNT, HIDDEN_NODES_COUNT, rigged);
		ArrayList<Double> targets = new ArrayList<Double>();

		int epoch = 0;
		
		System.out.println("Labels has " + trainLabels.rows() + " rows\n");
		
		/*for (Node n : outputLayer.nodes) {
			for (int j = 0; j < n.weights.size(); j++){
				output += n.weights.get(j).toString() + ",";
			}
			output  += n.weightBias + "\n\n";
		}	
		output += "Hidden weights are: ";
		for (Node n : hiddenLayer.nodes) {
			for (int j = 0; j < n.weights.size(); j++){
				output += n.weights.get(j).toString() + ",";
			}
			output  += n.weightBias + "\n\n";
		}
		
		System.out.println(output);
		*/
		//trainFeatures.shuffle(this.random, trainLabels);
		//printWeights();
		
		//Epoch
		/*
		double validationMSE = MSE(validationFeatures, validationLabels);
		double trainingMSE = MSE(trainFeatures,trainLabels);
		String MSE = epoch + "," + trainingMSE + "," + validationMSE;
		System.out.println(MSE);
		*/
		/*
		trainingMSE = MSE(trainFeatures, trainLabels);
		String tMSE = epoch + "," + trainingMSE;
		
		System.out.println(tMSE);
		*/
		while (training) {
			for (int r = 0; r < trainFeatures.rows(); r++) {
				
				if (r == 0) {
					 targets = new ArrayList<Double>();
					 outputs = new ArrayList<Double>(); 
				} 
				
				ArrayList<Double> inputs = new ArrayList<Double>();
				for (int c = 0; c < trainFeatures.cols(); c++) {
					inputs.add(trainFeatures.get(r, c));
				}
				
				targets.add(trainLabels.get(r, 0));
				
				hiddenLayer.calculateOutput(inputs);
				outputLayer.calculateOutput(hiddenLayer);
				Node n = outputLayer.getLargestNode();
				
				//hiddenLayer.printOutput();
				//outputLayer.printOutput();
				
				double targetToHit = targets.get(targets.size() - 1);
				
				outputLayer.backpropogate(targetToHit, hiddenLayer, multitarget);
				hiddenLayer.backpropogate(inputs, outputLayer);	
				//outputLayer.printBackprop();
				//hiddenLayer.printBackprop();
				outputLayer.updateWeights();
				hiddenLayer.updateWeights();
				
				
				//String vMSE = epoch + "," + validationMSE;
				//System.out.println(trainingMSE + "\n");
			    //this.printWeights();
			   // System.out.println("--------------------------------------------------");		    
			}
		
			//ArrayList<Double> MSE = new ArrayList<Double>():
			
		
			//double validationMSE = MSE(validationFeatures, validationLabels);
			//System.out.println(output);
			//Calculate MSE Here
			//Check MSE accuracy. Change if better. begin count. Stop training when it reaches 10	
			//Random
			if (validationFeatures.rows() > 0) {
				int correctCount = measureAccuracy2(validationFeatures, validationLabels);
				int total = trainFeatures.rows();
				validationFeatures.shuffle(this.random, validationLabels);
				double accuracy = this.measureAccuracy2(validationFeatures, validationLabels);
				//System.out.println("Test set accuracy: " + correctCount + "/" + validationFeatures.rows());
				//this.OutputResults(validationFeatures, validationLabels);
				
				if (correctCount > highestAccuracy) {
					highestAccuracy = correctCount;
					hiddenLayer.setBest();
					outputLayer.setBest();
					epochTimer = 0;
				} else {
					epochTimer++;
					if (epochTimer == epochLimit) {
						training = false;
					}
				}
			} else {
			
				
				int correctCount = measureAccuracy2(trainFeatures, trainLabels);
				int total = trainFeatures.rows();
				
				
				if (correctCount > highestAccuracy) {
					highestAccuracy = correctCount;
					hiddenLayer.setBest();
					outputLayer.setBest();
					epochTimer = 0;
				} else {
					epochTimer++;
					if (epochTimer == epochLimit) {
						training = false;
					}
				}
			}
			//double accuracy = this.measureAccuracy2(validationFeatures, validationLabels);
			
			
			/*
			double trainingMSE = MSE(trainFeatures, trainLabels);
			
			double validationMSE = MSE(validationFeatures, validationLabels);
			String MSE = epoch + "," + trainingMSE + "," + validationMSE;
			System.out.println(MSE);
			 */
			
			
			
			
			trainFeatures.shuffle(this.random, trainLabels);
		    epoch++;
		   //System.out.println("Epoch " + epoch + " accuracy: " + correctCount + "/" + total);
		    trainFeatures.shuffle(this.random, trainLabels);
		}	
		/*
		double trainingMSE = MSE(trainFeatures, trainLabels);
		double validationMSE = MSE(validationFeatures, validationLabels);
		double testMSE = MSE(testFeatures, testLabels);
		String MSE =  (int)hiddenLayer.nodes.size() + "," + trainingMSE + "," + validationMSE + "," + testMSE;
		System.out.println(MSE);
		*/
		
		
		System.out.println(outputLayer.nodes.get(0).momentum + "," + epoch);
		
		System.out.println(epoch + " Epochs");
		//OutputResults(trainFeatures, trainLabels);
	}
	
	public void printWeights() {
		String output = "Output weights are:\n";
		for (Node n : outputLayer.nodes) {
			for (int j = 0; j < n.weights.size(); j++){
				output += n.weights.get(j).toString() + ",";
			}
			output  += n.weightBias + "\n\n";
		}	
		output += "Hidden weights are: \n";
		for (Node n : hiddenLayer.nodes) {
			for (int j = 0; j < n.weights.size(); j++){
				output += n.weights.get(j).toString() + ",";
			}
			output  += n.weightBias + "\n\n";
		}	
		System.out.println(output);
	}
	
	//This method is a copied from supervised learner to check accuracy so far of the training to determine when to stop
		///Confusion matrix moved as a conveniance
		private int measureAccuracy2(Matrix features, Matrix labels) throws Exception
		{
			int correctCount = 0;
			double[] prediction = new double[1];
			for (int i = 0; i < features.rows(); i++) {
				double[] feat = features.row(i);
				int targ = (int)labels.get(i, 0);
				this.predict2(feat, prediction);
				int pred = (int)prediction[0];	
				if(pred == targ)
					correctCount++;
			}
			return correctCount;
		}
	
	public void OutputResults(Matrix input, Matrix target) {
		
		ArrayList<Double> targets = new ArrayList<Double>();
		for (int r = 0; r < target.rows(); r++) {
			targets.add(target.get(r, 0));
		}
		
		for (int r = 0; r < input.rows(); r++) {
			ArrayList<Double> inputs = new ArrayList<Double>();
			for (int c = 0; c < input.cols(); c++) {
				inputs.add(input.get(r, c));	
			}
			hiddenLayer.calculateOutput(inputs);
			outputLayer.calculateOutput(hiddenLayer);
			
			String output = "";
			for (Node n : outputLayer.nodes) {
				output += n.output + ",";  
			}
			
			Node n = outputLayer.getLargestNode();
			output += "\nBest out of the output is probability: " + n.identity;
			output += "\nReal target was: " + target.get(r, 0);
			
			System.out.println(output);
		}
	}
	
	public double Output(Matrix input, Matrix target) {
		
		try {
			return this.measureAccuracy(input, target, null);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			return 0.0;
		}
	}
	
	public double MSE(Matrix input, Matrix target) {

		ArrayList<Double> overallMSE = new ArrayList<Double>();	
			for (int r = 0; r < input.rows(); r++) {
			
				ArrayList<Double> inputs = new ArrayList<Double>();
				for (int c = 0; c < input.cols(); c++) {
					inputs.add(input.get(r, c));	
				}
				ArrayList<Double> singleMSE = new ArrayList<Double>();
				
				hiddenLayer.calculateOutput(inputs);
				outputLayer.calculateOutput(hiddenLayer);
				
				double currentMSE = 0.0;
				for (Node n : outputLayer.nodes) {					
					double targettouse = 0;
					double id = target.get(r, 0);
					if (n.identity == (int) id) {
						targettouse = 1;					
					} 					
					currentMSE += Math.pow(n.output - targettouse , 2);
				}
				currentMSE /= target.valueCount(0);
				overallMSE.add(currentMSE);
			}
			double finalMSE = 0;
			for (Double d : overallMSE) {
				finalMSE += d;
			}
			
			return finalMSE / target.rows();
		}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		ArrayList<Double> inputs = new ArrayList<Double>();
		
		for (int c = 0; c < features.length; c++) {
			inputs.add(features[c]);
		}
		
		hiddenLayer.calculateBestOutput(inputs);
		outputLayer.calculateBestOutput(hiddenLayer);
		Node n = outputLayer.getLargestNode();
		labels[0] = n.identity;

 		//Determine output from here
	}
	
	public void predict2(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		ArrayList<Double> inputs = new ArrayList<Double>();	
		for (int c = 0; c < features.length; c++) {
			inputs.add(features[c]);
		}
		
		hiddenLayer.calculateOutput(inputs);
		outputLayer.calculateOutput(hiddenLayer);
		Node n = outputLayer.getLargestNode();
		labels[0] = n.identity;

 		//Determine output from here
	}
}



class Node {
	double BIAS = 1;
	double momentum = 0, LEARNING_RATE = .3;
	ArrayList<Double> weights;
	ArrayList<Double> bestWeights;
	double weightBias = 0.0;
	double bestBiasWeight = 0.0;
	double deltawBias = 0.0;
	double prevBias = 0.0;
	double net = 0.0;
	double output = 0.0;
	double delta; 
	ArrayList<Double>  deltaw, previousDeltaw;
	double error; 
	int inputs;
	String type;
	int identity;
	
	Node(int inputCount, int iden) {
		inputs = inputCount;
		weights = new ArrayList<Double>();
		deltaw = new ArrayList<Double>();
		previousDeltaw = new ArrayList<Double>();
		identity = iden;
	}
	
	public void bestWeightSet() {
		bestWeights = new ArrayList<Double>();
		for (int i = 0; i < weights.size(); i++) {
			bestWeights.add(weights.get(i));
		}
		bestBiasWeight = weightBias;
	}
	
	public void calculateNet(ArrayList<Double> inputs) {
		net = 0;
		for (int i = 0; i < inputs.size(); i++) {
			net += inputs.get(i) * weights.get(i);
		}
		net += weightBias * BIAS;
		
	}
	
	public void sigmoid(double x) {
		output = (1/(1 + Math.pow(Math.E, -1 * x)));
	}
	
	public void delta (double target, String nodeType, double deltasum) {
		if (nodeType == "hidden") {
			delta = this.output * (1 - this.output) * deltasum;
		} else {
			delta = (target - output)*output*(1 - output);
		}
	}
	
	public void weightChanges(double outputi, double previousDeltaW) {	
			double deltawchange = LEARNING_RATE * delta * outputi + momentum * previousDeltaW;
			this.deltaw.add(deltawchange);		
	}
	
	public void updateWeights() {
		for (int i = 0; i < weights.size(); i++) {
			double weight = weights.get(i);
			weight += deltaw.get(i);
			weights.set(i, weight);
		}		
		weightBias += deltawBias;	
	}
	
	public void calculateBestNet(ArrayList<Double> inputs) {
		net = 0;
		for (int i = 0; i < inputs.size(); i++) {
			net += inputs.get(i) * bestWeights.get(i);
		}
		net += bestBiasWeight * BIAS;
		
	}
}

class HiddenLayer {
	ArrayList<Node> nodes;
	ArrayList<Double> inputVector;
	double momentum = 0;
	
	String nodeType = "hidden";
	HiddenLayer(int NodeCount, int inputCount, String rigged) {
		nodes = new ArrayList<Node>();
		if (rigged == "Problem") {
			setWeights(NodeCount, inputCount);
			momentum = 0;
		} else if (rigged == "Test3") {
			Node n1 = new Node(2, 0);
			Node n2 = new Node(2, 1);
			Node n3 = new Node(2, 2);
			n1.weightBias = -0.01;
			n1.weights.add(-0.03);
			n1.weights.add(0.03);		
			
			n2.weightBias = 0.01;
			n2.weights.add(0.04);
			n2.weights.add(-0.02);
			
			n3.weightBias = -0.02;
			n3.weights.add(0.03);
			n3.weights.add(0.02);
			
			nodes.add(n1);
			nodes.add(n2);
			nodes.add(n3);
			
			for (Node n : nodes) {
				n.LEARNING_RATE = .175;
				n.momentum = .9;
				int i = 0;
				while (i < 3) {
					n.previousDeltaw.add(0.0);
					i++;
				}
			}	
			this.momentum = .9;
		} else {
			initWeights(NodeCount, inputCount);
		}	
	}
	
	private void setWeights(int NodeCount, int inputCount) {
		
		/*
		for (int i = 0; i < NodeCount; i++) {
			Node node = new Node(inputCount, i);
			nodes.add(node);
			for (int j = 0; j < inputCount; j++) {
				node.weights.add(1.0);
				node.previousDeltaw.add(0.0);
			}
			node.weightBias = 1.0;
			node.LEARNING_RATE = 1;
		}*/
		
		
		Node n10 = new Node(2, 0);
		Node n11 = new Node(2, 1);
		n10.weights.add(.3);
		n10.weights.add(-.1);
		n10.weightBias = .1;

		n11.weights.add(.3);
		n11.weights.add(-.2);
		n11.weightBias = -.2;
		
		n10.previousDeltaw.add(0.0);
		n10.previousDeltaw.add(0.0);
		n11.previousDeltaw.add(0.0);
		n11.previousDeltaw.add(0.0);
		
		
		nodes.add(n10);
		nodes.add(n11);
	}
	
	
	private void initWeights(int NodeCount, int inputCount) {
		for (int i = 0; i < NodeCount; i++) {
			Node node = new Node(inputCount, i);
			nodes.add(node);
			
			for (int j = 0; j < inputCount; j++) {
				node.weights.add(-0.5 + Math.random()*(.5 + .5));
				node.previousDeltaw.add(0.0);
			}
			node.weightBias = -0.5 + Math.random()*(.5 + .5);
		}	
	}
	
	public void calculateOutput(ArrayList<Double> inputs) {
		inputVector = new ArrayList<Double>();
		
		for (Double d : inputs) {
			inputVector.add(d);
		}
		
		for (Node n : nodes) {
			n.calculateNet(inputVector);
			n.sigmoid(n.net);
		}
	}
	
	public void calculateBestOutput(ArrayList<Double> inputs) {
		inputVector = new ArrayList<Double>();
		
		for (Double d : inputs) {
			inputVector.add(d);
		}
		
		for (Node n : nodes) {
			n.calculateBestNet(inputs);
			n.sigmoid(n.net);
		}
	}
	
	public void backpropogate(ArrayList<Double> input, OutputLayer outputLayer) {
		for (int i = 0; i < nodes.size(); i++) {
			Node n = nodes.get(i);
			double deltasum = 0, weightij = 0, nodedelta = 0;
			
			for (Node o : outputLayer.nodes) {
				nodedelta = o.delta;
				weightij = o.weights.get(i);
				deltasum += nodedelta * weightij;
			}
			n.delta(0, nodeType, deltasum);
		}
		
		for (Node n : nodes) {
		
			for (int i = 0; i < input.size(); i++) {
				n.weightChanges(input.get(i), n.previousDeltaw.get(i));
			}				
			n.deltawBias = n.LEARNING_RATE* n.delta * n.BIAS + momentum * n.deltawBias;
		}
	}
	
	public void updateWeights() {
		for (Node n: nodes) { 
			n.updateWeights();
		}
		
		for (Node n: nodes) {
			n.previousDeltaw.clear();
			
			for (int i = 0; i < n.deltaw.size(); i++) {
				n.previousDeltaw.add(n.deltaw.get(i));
				

			}
			
			n.deltaw.clear();
		}
	}
	
	public void setBest() {
		for (Node n : nodes) {
			n.bestWeightSet();
		}
	}
	
	
	public void printOutput() {
		
		String output = "Nets for hidden layer are: \n";
		for (Node n : nodes) {
			output += n.net + ",";	
		}
		output += "\n Outputs for hidden layer are: \n";
		for (Node n : nodes) {
			output += n.output + ",";	
		}
		System.out.println(output);
	}
	
	public void printBackprop() {
		String output = "::::::Hidden node backprop:::::::\n";
		for (Node p : nodes) {
			output += "Weights of node identity " + p.identity + "\n";
			for (int j = 0; j < p.weights.size(); j++){
				output += p.weights.get(j) + ",";
			}
			output  += p.weightBias + "\n";
			output += "Delta: " + p.delta + "\n";
			output += "New delta w values are: \n";
			for (int j = 0; j < p.weights.size(); j++){
				output += p.deltaw.get(j).toString() + ",";
			}	
			output  += p.deltawBias + "\n\n";
		}
		
		output += "::::::::::::::::::::::::::::";
		System.out.println(output + "\n");
	}
}


class OutputLayer{
	ArrayList<Node> nodes;
	ArrayList<Node> deltaSums;
	int nodecount = 0;
	double momentum = .9;
	String nodeType = "output";
	
	OutputLayer(int NodeCount, int inputCount, String rigged) {
		nodes = new ArrayList<Node>();
		deltaSums = new ArrayList<Node>();
		if (rigged == "Problem") {
			setWeights(NodeCount, inputCount);
			this.momentum = 0;
		} else if (rigged == "Test3") {
			
			Node n0 = new Node(3,0);
			n0.weights.add(-0.01);
			n0.weights.add(0.03);
			n0.weights.add(0.02);
			n0.LEARNING_RATE = .175;
			n0.momentum = .9;
			n0.weightBias = 0.02;
			n0.previousDeltaw.add(0.0);
			n0.previousDeltaw.add(0.0);
			n0.previousDeltaw.add(0.0);
			n0.prevBias = 0;
			nodes.add(n0);
			this.momentum = .9;
			
		} else {
			initWeights(NodeCount, inputCount);
			nodecount = NodeCount;
		}
		
	}
	
	private void initWeights(int NodeCount, int inputCount) {
		for (int i = 0; i < NodeCount; i++) {
			nodes.add(new Node(inputCount, i));
			Node node = nodes.get(i);
			for (int j = 0; j < inputCount; j++) {
				node.weights.add(-0.5 + Math.random()*(.5 + .5));
				node.previousDeltaw.add(0.0);
			}
			node.weightBias = -0.5 + Math.random()*(.5 + .5);
		}
	}
	
	private void setWeights(int NodeCount, int inputCount) {
			/*for (int i = 0; i < NodeCount; i++) {
				nodes.add(new Node(inputCount, i));
				Node node = nodes.get(i);
				for (int j = 0; j < inputCount; j++) {
					node.weights.add(1.0);
					node.previousDeltaw.add(0.0);
				}
				node.weightBias = 1;
				node.LEARNING_RATE = 1;
			}*/
		
		
		Node n20 = new Node(2,0);
		n20.weights.add(0.0);
		n20.weights.add(-.1);
		n20.weightBias = .3;
		n20.previousDeltaw.add(0.0);
		n20.previousDeltaw.add(0.0);
		nodes.add(n20);
		
	}
	
	public void calculateOutput(HiddenLayer hiddenLayer) {
		ArrayList<Double> inputs = new ArrayList<Double>();
		for (int i = 0; i < hiddenLayer.nodes.size(); i++) {
			inputs.add(hiddenLayer.nodes.get(i).output);
		}
		
		for (Node n : nodes) {
			n.calculateNet(inputs);
			n.sigmoid(n.net);
		}
	}
	
	public void calculateBestOutput(HiddenLayer hiddenLayer) {
		ArrayList<Double> inputs = new ArrayList<Double>();
		for (int i = 0; i < hiddenLayer.nodes.size(); i++) {
			inputs.add(hiddenLayer.nodes.get(i).output);
		}
		
		for (Node n : nodes) {
			n.calculateBestNet(inputs);
			n.sigmoid(n.net);
		}
	}
	
	
	public void backpropogate(double target, HiddenLayer hiddenLayer, boolean multitarget) {
		
		if (multitarget)  {
			for (Node n: nodes) {
				double targettouse = 0;
				if (target == n.identity) {
					targettouse = 1;
				}
				
				
				n.delta(targettouse, nodeType, 0);	
				
				ArrayList<Double> inputVector = new ArrayList<Double>();
				
				for (Node o : hiddenLayer.nodes) {
					inputVector.add(o.output);
				}
				
				for (int i = 0; i < inputVector.size(); i++) {
					n.weightChanges(inputVector.get(i), n.previousDeltaw.get(i));	
				}
				
				n.deltawBias = n.LEARNING_RATE* n.delta * n.BIAS + momentum * n.deltawBias;
			}
		} else {
			for (Node n: nodes) {
				n.delta(target, nodeType, 0);	
				
				ArrayList<Double> inputVector = new ArrayList<Double>();
				
				
				for (Node o : hiddenLayer.nodes) {
					inputVector.add(o.output);
				}
				
				for (int i = 0; i < inputVector.size(); i++) {
					n.weightChanges(inputVector.get(i), n.previousDeltaw.get(i));	
				}
				
				n.deltawBias = n.LEARNING_RATE* n.delta * n.BIAS + momentum * n.deltawBias;
			}
		}
	}
	
	public void updateWeights() {
		for (Node n: nodes) { 
			n.updateWeights();
		}
		
		for (Node n: nodes) {
			n.previousDeltaw.clear();
			
			for (int i = 0; i < n.deltaw.size(); i++) {
				n.previousDeltaw.add(n.deltaw.get(i));
			}
			n.deltaw.clear();
		}
	}


	public double getOutput(){
		double largest = 0.0;
		for (Node n : nodes) {
			if (n.output > largest) {
				largest = n.output;
			}	
		}
		return largest;
	}
	
	public Node getLargestNode(){
		double largest = -Double.MAX_VALUE;
		Node returnN = null;
		for (Node n : nodes) {
			if (n.output > largest) {
				largest = n.output;
				returnN = n;
			}	
		}
		return returnN;
	}

	public void setBest() {
		for (Node n : nodes) {
			n.bestWeightSet();
		}
	}
	
	public void printOutput() {
		
		String output = "Nets for output layer are: \n";
		for (Node n : nodes) {
			output += n.net + ",";	
		}
		output += "\nOutputs for output layer are: \n";
		for (Node n : nodes) {
			output += n.output + ",";	
		}
		System.out.println(output + "\n");
	}
	
	public void printBackprop() {
		String output = ":::::Output node backprop::::\n";
		for (Node p : nodes) {
			for (int j = 0; j < p.weights.size(); j++){
				output += p.weights.get(j).toString() + ",";
			}
			output  += p.weightBias + "\n\n";
			output += "Delta: " + p.delta + "\n";
			output += "New delta w values are: \n";
			for (int j = 0; j < p.weights.size(); j++){
				output += p.deltaw.get(j).toString() + ",";
			}	
			output  += p.deltawBias + "\n";
		}
		System.out.println(output + "\n");
	}
	
	/*public int getOutputID(){
		int id = -1;
		double largest = 0;
		for (Node n : nodes) {
			if (n.output > largest) {
				id = n.identity;
				largest = n.identity;
			} 
		}
		return id;
	}*/
}

