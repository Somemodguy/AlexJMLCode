//Perceptron.java
//Alex Jensen
//CS 4478
//Dr. Bodily

import java.util.ArrayList;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Random;

public class Perceptron extends SupervisedLearner {
	//Need a cvalue
	//need weights stored here
	ArrayList<Double> weights;
	double biasInput = 1.0;
	double c = .1;
	double outputs[];
	Random RAND;
	ArrayList<Double> bestWeights;
	
	public Perceptron(Random rand) {
		RAND = rand;
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {

		boolean training = true;		
		weights = new ArrayList<Double>();
		int runcount = 0, maxOverfitRunCount = 10, epoch = 0;
		double[] ziOutputs = new double[features.rows()];
		int highestAccuracy = 0, currentAccuracy = 0;

		//code help for java number usage was found
				//https://stackoverflow.com/questions/28786856/java-get-random-double-between-two-doubles
				for (int i = 0; i < features.cols() + 1; i++) {
					//weights.add(0.0);
					double weightToAdd = -0.5 + Math.random()*(.5 + .5);
					weights.add(weightToAdd);
				}
		
		int startAccuracy = measureAccuracy2(features,labels);
		double startAcc = (double) startAccuracy / features.rows();
		System.out.println("0, " + Double.toString(startAcc));
		
		
		
		//System.out.println("weights has " + weights.size() + " weights in inside");
		while (training) {
			
			//This for loop is one epoch through the training set
			
			//One epoch code
			for (int i =0; i < features.rows(); i++) {
				/*
				String outputStr = new String(" ");
				for(int j = 0; j < weights.size();j++) {
					//System.out.println(weights.get(j).toString());
					outputStr += String.format("%.2f", weights.get(j)) + ", ";
				}			
				outputStr = outputStr.substring(0, outputStr.length() - 2);
				System.out.println(outputStr);
				*/
				double rowsum = 0, output;
				for(int j = 0; j < features.cols();j++) {
					//System.out.println("Value at (" + i + ", " + j + ") is " + features.get(i,j));
					rowsum += features.get(i, j) * weights.get(j);
				}	
				rowsum += biasInput * weights.get(weights.size() - 1);
				//System.out.println("Net is: " + String.format("%.2f",rowsum));
				
				if (rowsum > 0) {
					output = 1;
				} else {
					output = 0;
				}
				ziOutputs[i] = output;	
				
				if (output != labels.get(i, 0)) {
					//System.out.println("Row " + i + " was not predicated correctly. Changing weights");				
					//For each column of weights, change them, then change bias weight at the end
					for (int j = 0; j < features.cols(); j++) {
						double t = labels.get(i,0);
						double z = output;
						double oldWeight = weights.get(j);
						double newWeight = oldWeight + c * (t-z) * features.get(i, j);		
						weights.set(j, newWeight);
					}				
					weights.set(weights.size() - 1, weights.get(weights.size() - 1) + c * (labels.get(i, 0) - output) * biasInput);
				}
			}
			
			currentAccuracy = measureAccuracy2(features, labels);
			//System.out.println("Total correct out of " + features.rows() + " is " + currentAccuracy);
			if (currentAccuracy > highestAccuracy) {
				highestAccuracy = currentAccuracy; 
				bestWeights = new ArrayList<Double>();
				for (double n : weights) {
					bestWeights.add(n);
				}
				runcount = 0;
			} else {
				runcount++;
				if (runcount >= maxOverfitRunCount) {
					training = false;
				}
			}		
			epoch++;	
			features.shuffle(RAND, labels);
			double accuracy = this.measureAccuracy(features, labels, null);
			 System.out.println(Integer.toString(epoch) + "," + (1 - accuracy));
		}	
		
		

		/*
		String outputStr = new String(" ");
		for(int j = 0; j < weights.size();j++) {
			//System.out.println(weights.get(j).toString());
			outputStr += String.format("%.6f", weights.get(j)) + ", ";
		}		*/	
		//
		
		//outputStr = outputStr.substring(0, outputStr.length() - 2);
		//System.out.println(outputStr);*/
		
		double accuracy = this.measureAccuracy(features, labels, null);
		 System.out.println(Integer.toString(epoch));
	}
	
	
	@Override
	//Populate array of labels. Could have multiple outputs
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		
		double output;
		double sum = 0;
		//Binary classificiaton
		for (int i = 0; i < features.length; i++) {
			double weight = bestWeights.get(i);
			double feature = features[i];
			sum += features[i] * weight;
		}
		
		double biasInputI = biasInput;
		double biasWieght = bestWeights.get(bestWeights.size() - 1);
		
		sum += biasInputI * biasWieght;
		if (sum > 0)  { output = 1; } else { output = 0; }
		
		labels[0] = output;	
	}
	
	///This method is a copied from supervised learner to check accuracy so far of the training to determine when to stop
	///Confusion matrix moved as a conveniance
	private int measureAccuracy2(Matrix features, Matrix labels) throws Exception
	{
		int correctCount = 0;
		double[] prediction = new double[1];
		for (int i = 0; i < features.rows(); i++) {
			double[] feat = features.row(i);
			int targ = (int)labels.get(i, 0);
			predictlocal(feat, prediction);
			int pred = (int)prediction[0];
			if(pred == targ)
				correctCount++;
		}
		return correctCount;
	}
	
	//Populate array of labels. Could have multiple outputs
	private void predictlocal(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		
		double output;
		double sum = 0;
		//Binary classificiaton
		for (int i = 0; i < features.length; i++) {
			sum += features[i] * weights.get(i);
		}
		
		sum += biasInput * weights.get(weights.size() - 1);
		if (sum > 0)  { output = 1; } else { output = 0; }
		
		labels[0] = output;	
	}
	
	
}
