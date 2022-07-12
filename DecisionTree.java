import java.util.HashMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.TreeMap;

public class DecisionTree extends SupervisedLearner {
	Leaf startLeaf;
	int count = 0;
	Random rng = new Random();
	boolean Prune = true;
	DecisionTree() {
		
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		// TODO Auto-generated method stub		
		//Replace missing values here
		
		ArrayList<Integer> masterSubset = new ArrayList<Integer>();
		for (int i = 0; i < features.rows(); i++) {
			masterSubset.add(i);
		}
		
		
		double TRAIN_PERCENT = .7;
		
		int trainSize = (int)(TRAIN_PERCENT * features.rows());
		
		int validationSize = features.rows() - trainSize;
		validationSize /= 2;
		int testSize = validationSize;
		
		features.shuffle(this.rng, labels);
		
		ArrayList<Integer> trainSubset = new ArrayList<Integer>();
		ArrayList<Integer> validationSubset = new ArrayList<Integer>();
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
			
			
			for (int i = 0; i < trainSize; i++) {
				trainSubset.add(i);
			}
			for (int i = 0; i < validationSize; i++) {
				validationSubset.add(i);
			}
		}

		
		startLeaf = new Leaf(trainFeatures, trainLabels, trainSubset, new ArrayList<Integer>(), null);
		//System.out.println("Mode of the matrix is " + labels.mostCommonValue(0));
		
		
		int correctCount = measureAccuracy2(validationFeatures, validationLabels);
		int rows = validationFeatures.rows();
		double accuracy = (double) correctCount/(double) rows;
		
		//System.out.println("Features has row count of:   " + rows);
		System.out.println("Unpruned:		" + accuracy);
		//System.out.println("Training Set Accuracy:   " + accuracy);
		
		int nodecount = startLeaf.countNodes();
		System.out.println("Tree has " + nodecount + " nodes");
		
		int depth = startLeaf.maxDepth();
		System.out.println("Tree has depth of " + depth);
		
		if (Prune) {
			Prune(startLeaf, validationFeatures,  validationLabels);
		}
		
		nodecount = startLeaf.countNodes();
		System.out.println("Tree has " + nodecount + " nodes");
		
		depth = startLeaf.maxDepth();
		System.out.println("Tree has depth of " + depth);
		
		correctCount = measureAccuracy2(features, labels);
		rows = features.rows();
		accuracy = (double) correctCount/(double) rows;
		System.out.println("Pruned: 	" + accuracy);
		
		
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		

		labels[0] = startLeaf.Traverse(features);
		//System.out.println("Output of prediction is " + labels[0]);
		//labels[0] = startLeaf.Traverse(features);
		//System.out.println("Output of prediction is " + labels[0]);
	}
	
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
	public void predict2(double[] features, double[] labels) throws Exception {
		
		labels[0] = startLeaf.Traverse(features);
 		
	}
	
	//Wrapper function
	public void Prune(Leaf start, Matrix features, Matrix labels) {
		isPruning(start, features, labels);
	}
	
	
	//I received help from Rosetta regarding the understanding of pruning
	public void isPruning(Leaf current, Matrix features, Matrix labels) {
		if (current.end != true) {
			for(int i = 0; i < current.branches.size(); i++) {
				isPruning(current.branches.get(i), features, labels);
			}
			try {			
				//Resetta helped me with understanding the accuracy of every node
				//here on this section.
				int accuracy1 = measureAccuracy2(features, labels);
				current.end = true;
				current.output = current.majority;
				int accuracy2 = measureAccuracy2(features, labels);
				if (accuracy2 >= accuracy1) {
					current.end = true;
					current.output = current.majority;
				} else {
					current.end = false;
					current.output = -1;
				}
			} catch (Exception e) {
			}
		} else {
			//do nothing
		}
	}
}

//Even though I treat all leaf classes like nodes due to another file already containing the node class,
//I use "end" and "isPure" to define the end of a node as a leaf
class Leaf {
	Leaf Parent;
	String Attribute;
	Matrix Features, Labels;
	boolean isPure = false;
	boolean end = false;
	double output = -1;
	double majority;
	int splittingColumn = -1;
	ArrayList<Integer> nodeSubset;
	ArrayList<Leaf> branches;
	ArrayList<Integer> columnsSplitOn;
	ArrayList<ArrayList<Integer>> subsets = new ArrayList<ArrayList<Integer>>();
	
	Leaf(Matrix features, Matrix labels, ArrayList<Integer> subset, ArrayList<Integer> columnsUsed, Leaf parent) {
		Parent = parent;
		Features = features;
		Labels = labels;
		columnsSplitOn = new ArrayList<Integer>();
		copyIntList(columnsUsed, columnsSplitOn);
		nodeSubset = subset;
		if (Purity(features, labels, subset)) {
		
			isPure = true;
			end = true;
			int indexOfPurity = subset.get(0);
			double featureValue = labels.get(indexOfPurity, 0);
			output = labels.get(subset.get(0), 0);
			
			
		} else if (columnsUsed != null && features.cols() - columnsUsed.size() == 1 || columnsUsed != null && features.cols() - columnsUsed.size() == 0 ){
			 if (columnsUsed != null && features.cols() - columnsUsed.size() == 1) {
				 end = true;
					ArrayList<Double> outputs = new ArrayList<Double>();
					for (int i = 0; i < subset.size(); i++) {
						outputs.add(labels.get(subset.get(i), 0));
					}
					
					output = mode(outputs, labels, 0);
			 } else {
				 System.out.println("Empty subset.");
			 }
			 
		} else {
			//First determine column to split on.
			ArrayList<Double> infoGains = new ArrayList<Double>();
			if (subset == null) {
				for (int i = 0; i < features.cols(); i++) {
					//infoGains.put(features.attrName(i), ID3(features, labels, null, i));
					//System.out.println("------------------------");
					//System.out.println("For Column " + features.attrName(i));				
					Double gains = ID3(features, labels, null, i);
					//System.out.println("Info gain is " + gains);
					//System.out.println("************************\n");
					infoGains.add(gains);
				}
			} else {
				for (int i = 0; i < features.cols(); i++) {
					//infoGains.put(features.attrName(i), ID3(features, labels, null, i));
					if (!columnsUsed.contains(i)) {
						//System.out.println("------------------------");
						//System.out.println("For Column " + features.attrName(i));
						Double gains = ID3(features, labels, subset, i);
						//System.out.println("Info gain is " + gains);
						//System.out.println("************************\n");
						infoGains.add(gains);
					} else {
						infoGains.add(Double.MAX_VALUE);
					}
				}
			}
			
			Double smallest = Collections.min(infoGains);
			splittingColumn = infoGains.indexOf(smallest); 
			columnsSplitOn.add(splittingColumn);
			Attribute = features.attrName(splittingColumn);
			//System.out.println("Splitting on column " + Attribute); 
			
			
			ArrayList<Double> outputs = new ArrayList<Double>();
			for (int i = 0; i < subset.size(); i++) {
				outputs.add(labels.get(subset.get(i), 0));
			}
			majority = mode(outputs, labels, 0);
			
			
			//SPlit dataset based on column
			subsets = new ArrayList<ArrayList<Integer> >(features.valueCount(splittingColumn)); 
			for (double i = 0.0; i < features.valueCount(splittingColumn); i++) {
					ArrayList<Integer> newSubset = this.Split(features, subset, splittingColumn, i);
					subsets.add(newSubset);
					//System.out.println("New subset has " + newSubset.size() + " elements");
			}	
			//Proceed to work through each subset to either create a final node or split again
			branches = new ArrayList<Leaf>();
			for (double i = 0.0; i < features.valueCount(splittingColumn); i++) {
				branches.add(new Leaf(features, labels, subsets.get((int)i), columnsSplitOn, this));
			}
		}
	}
	
	public ArrayList<Integer> Split(Matrix features, ArrayList<Integer> currentSubset, double splittingColumn, double splittingValue) {
		ArrayList<Integer> subset = new ArrayList<Integer>();
		if (currentSubset != null) {
			for (int instance = 0; instance < currentSubset.size(); instance++) {
				if (features.get(currentSubset.get(instance), (int)splittingColumn) == splittingValue) {
					subset.add(currentSubset.get(instance));
					//System.out.println("Index " + instance + " Added to current subset");
				}
				
				if(features.get(instance, (int)splittingColumn) == Double.MAX_VALUE) {
					
				}
			}
		} else {
			for (int instance = 0; instance < features.rows(); instance++) {
				if(features.get(instance, (int)splittingColumn) == Double.MAX_VALUE) {
					
				}
				
				if (features.get(instance, (int)splittingColumn) == splittingValue) {
					subset.add(instance);
					//System.out.println("Index " + instance + " Added to current subset");
				}
			}
		}
		
		return subset;
	}
	
	
	//Decide function using ID3 Here
	public double ID3(Matrix features, Matrix labels, ArrayList<Integer> featureSubset, int column) {
		double S, Sj;
		
		if (featureSubset != null) {
			S = featureSubset.size();
		} else {
			S = features.rows();
		}
		int featureCount = features.valueCount(column);
		
		
		double gain = 0;
		//Get total of 
		for (int i = 0; i < featureCount; i++) {
			Sj = columnValues(features, featureSubset, column, i);
			//System.out.println("There are " + Sj + "/" + S + " of feature type " + features.attrValue(column, i));
			Double log = this.logFunction(features, labels, featureSubset, column, Sj, i);
			gain += (Sj / S) * log;
		}
		return gain;
	}
	
	public boolean Purity(Matrix features, Matrix labels, ArrayList<Integer> featureSubset) {
		if (featureSubset == null) return false;
		if (featureSubset.isEmpty()) return false;
		double subsetCheck = labels.get(featureSubset.get(0), 0);	
		boolean ispure = true;
		for (int i = 0; i < featureSubset.size(); i++) {	
			if (labels.get(featureSubset.get(i), 0) != subsetCheck) {
				ispure = false;
				break;
			}
		}
		return ispure;
	}
	
	
	public Double logFunction(Matrix features, Matrix labels, ArrayList<Integer> featureSubset, int column, double Sj, double featureValue) {
		double log = 0;	
		for (int i = 0; i < labels.valueCount(0); i++) {
			double numerator = matchValues(features, labels, featureSubset, column, featureValue, i);
			if (numerator != 0) {
				log += -1 * (numerator / Sj) * (Math.log(numerator / Sj) / Math.log(2));
			}
			//System.out.println("Of feature " + features.attrName(column) + ", " + numerator + "/" + Sj + " are " + labels.attrValue(0, i));
		}
		return log;
	}
	
	
	//Detect purity function
	
	//count instances of a column with a certain value
	public int columnValues(Matrix matrix, ArrayList<Integer> subsetIndex, int column, double value) {
		int count = 0;
		if (subsetIndex == null) {	
			for (int r = 0; r < matrix.rows(); r++) {
				if (matrix.get(r, column) == value) {
					count++;
				}
			}
		} else {
			for (Integer row : subsetIndex) {
				if (matrix.get(row, column) == value) {
					count++;
				}
			}
		}
		return count;
	}
	
	public int matchValues(Matrix matrix, Matrix output, ArrayList<Integer> subsetIndex, int column, double featureV, double labelV) {
		int count = 0;
		if (subsetIndex == null) {	
			for (int r = 0; r < matrix.rows(); r++) {
				if (matrix.get(r, column) == featureV && output.get(r, 0) == labelV) {
					count++;
				}
			}
		} else {
			for (Integer row : subsetIndex) {
				if (matrix.get(row, column) == featureV && output.get(row, 0) == labelV) {
					count++;
				}
			}
		}
		return count;
	}
	//Determine split and subsets here
	//public boolean determinePureColumn 

	public void copyIntList(ArrayList<Integer> list1, ArrayList<Integer> list2) {
		if (list1 == null ) {
		} else {
			for (Integer i : list1) {
				list2.add(i);
			}
		}
		
	}
	
	//Returns the most occuring element in the column
	public double mode(ArrayList<Double> inputs, Matrix labels, int column) {
		Integer index = -1;
		Integer[] counts = new Integer[labels.valueCount(0)];
		
		for (int i = 0; i < counts.length; i++) {
			counts[i] = 0;
			for (int j = 0; j < inputs.size(); j++) {
				if (inputs.get(j) == i) {
					counts[i]++;
				}
			}
		}
		
		double highestCount = -1;
		for (int i = 0; i < counts.length; i++) {
			if (counts[i] > highestCount) {
				highestCount = counts[i];
				index = i;
			}
		}
		return index;
	}
	
	public double Traverse(double[] features) {
		if (this.end) {
			return this.output;
		} else {
			double feature = features[this.splittingColumn];
			if (feature == Double.MAX_VALUE) {
				return this.branches.get((int)majority).Traverse(features);
			}
			return this.branches.get((int)feature).Traverse(features);
		}
	}
	
	public int countNodes() {
		int count = 1;
		
		if (this.end != true) {
			for (Leaf l : branches) {
				count += l.countNodes();
			}
		}
		return count;
	}
	
	public int maxDepth() {
		ArrayList<Integer> depths = new ArrayList<Integer>();
		if (this.end != true) {
			for (Leaf b : branches) {
				depths.add(b.maxDepth());
			}
			return Collections.max(depths) + 1;
		} else {
			return 1;
		}
	}
	
	
}





