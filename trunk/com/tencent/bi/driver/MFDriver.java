package com.tencent.bi.driver;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

import com.tencent.bi.cf.model.AbstractMF;
import com.tencent.bi.cf.model.collectivemf.AbstractCMF;
import com.tencent.bi.cf.model.combine.EnHybridBiasMF;
import com.tencent.bi.cf.model.common.MFPrediction;
import com.tencent.bi.cf.model.hybridmf.AbstractHybridMF;
import com.tencent.bi.cf.model.tensor.AbstractTensor;
import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * Main driver, and user interface
 * @author tigerzhong
 *
 */
public class MFDriver {
	/**
	 * Model name
	 */
	private static String modelName = "";
	/**
	 * Optimization method name
	 */
	private static String opName = "";
	/**
	 * Number of users
	 */
	private static int numUser = -1;
	/**
	 * Number of items
	 */
	private static int numItem = -1;
	/**
	 * Number of contexts
	 */
	private static int numContext = -1;
	/**
	 * Number of latent dimensions
	 */
	private static int numD = -1;
	/**
	 * Data input path
	 */
	private static String inputPath;
	/**
	 * Number of iterations
	 */
	private static int numIt = 100;
	/**
	 * Number of reguralization parameter
	 */
	private static double lambda = 0.005;
	/**
	 * Number of reguralization parameter
	 */
	private static double lambdaS = 0.005;
	/**
	 * Number of reguralization parameter
	 */
	private static double lambdaV = 0.005;
	/**
	 * Learning rate
	 */
	private static double learningRate = 0.01;
	/**
	 * Number of user features
	 */
	private static int numUserFeature = 0;
	/**
	 * Number of item features
	 */
	private static int numItemFeature = 0;
	
	private static double ra = 1.0;
	
	private static double rb = 1.0;
	
	private static double rc = 1.0;
	
	/**
	 * Preform prediction
	 * @throws Exception
	 */
	
	public static void performPredictionAll() throws Exception {
		AbstractMF model = (AbstractMF) Class.forName(modelName).newInstance();
		model.predictAll();
	}
	
	public static void performPredictionPair() throws Exception {
		MFPrediction.predictAll(inputPath, modelName);
	}
	
	public static void performPredictionPairDist()  throws Exception {
		AbstractMF model = (AbstractMF) Class.forName(modelName).newInstance();
		model.predictPair(inputPath, numD);
	}
	
	/**
	 * Perform training
	 * @throws Exception
	 */
	public static void performTraining() throws Exception {
		if(modelName.indexOf("combine")>-1){					//combine
			EnHybridBiasMF model = new EnHybridBiasMF();
			model.initModel(numUser, numItem, numD, numUserFeature, numItemFeature, ra, rb, rc, lambda, learningRate, numIt, inputPath);
			model.buildModel();
		} else if(modelName.indexOf("tensor")>-1){				//tensor
			AbstractTensor model = (AbstractTensor) Class.forName(modelName).newInstance();
			model.initModel(numUser, numItem, numContext, numD, opName, lambda, learningRate, numIt, inputPath);
			model.buildModel();
		} else if(modelName.indexOf("collectivemf")>-1){ 		//cmf
			AbstractCMF model = (AbstractCMF) Class.forName(modelName).newInstance();
			model.initModel(numUser, numItem, numContext, numD, opName, lambdaV, lambdaS, lambda, learningRate, numIt, inputPath);
			model.buildModel();
		} else if(modelName.indexOf("hybridmf")>-1){ 			//hybridmf
			AbstractHybridMF model = (AbstractHybridMF) Class.forName(modelName).newInstance();
			model.initModel(numUser, numItem, numD, numUserFeature, numItemFeature, opName, lambda, learningRate, numIt, inputPath);
			model.buildModel();
		} else{ 												//mf, biasmf, rankmf
			AbstractMF model = (AbstractMF) Class.forName(modelName).newInstance();
			model.initModel(numUser, numItem, numD, opName, lambda, learningRate, numIt, inputPath);
			model.buildModel();
		}
	}
	
	/**
	 * Build options for command line
	 * @return Command options
	 * @throws Exception
	 */
	@SuppressWarnings("static-access")
	public static Options buildOption() throws Exception {
		//Initialize
		Options options = new Options();

		//Generate command
		Option help = new Option("help", "print this message");
		
		Option task = OptionBuilder.withArgName("task").hasArg().withDescription("task, training or prediction").create("T");
		Option model = OptionBuilder.withArgName("com.tencent.mf.model").hasArg().withDescription("com.tencent.mf.model").create("M");
		Option op = OptionBuilder.withArgName("com.tencent.mf.optimization").hasArg().withDescription("op").create("O"); 
				
		Option numIt = OptionBuilder.withArgName("iteration").hasArg().withDescription("number of iteration").create("i");
		Option learningRate = OptionBuilder.withArgName("learning-rate").hasArg().withDescription("learning rate").create("e");
		Option lambda = OptionBuilder.withArgName("lambda").hasArg().withDescription("regularization parameter").create("r");
		
		Option lambdaS = OptionBuilder.withArgName("lambdaS").hasArg().withDescription("regularization parameter").create("rs");
		Option lambdaV = OptionBuilder.withArgName("lambdaV").hasArg().withDescription("regularization parameter").create("rv");
		Option lambdaA = OptionBuilder.withArgName("lambdaA").hasArg().withDescription("regularization parameter").create("ra");
		Option lambdaB = OptionBuilder.withArgName("lambdaB").hasArg().withDescription("regularization parameter").create("rb");
		Option lambdaC = OptionBuilder.withArgName("lambdaC").hasArg().withDescription("regularization parameter").create("rc");
		
		
		Option numU = OptionBuilder.withArgName("nuser").hasArg().withDescription("Number of Users").create("m");
		Option numV = OptionBuilder.withArgName("nitem").hasArg().withDescription("Number of Items").create("n");
		Option numS = OptionBuilder.withArgName("ncontext").hasArg().withDescription("Number of Contexts").create("s");
		Option numD = OptionBuilder.withArgName("ndimension").hasArg().withDescription("Number of Dimension").create("d");

		Option featureU = OptionBuilder.withArgName("nuserfeature").hasArg().withDescription("Number of Users' Features").create("fu");
		Option featureV = OptionBuilder.withArgName("nitemfeature").hasArg().withDescription("Number of Items' Features").create("fv");
		
		Option trainingData = OptionBuilder.withArgName("training-data").hasArg().withDescription("path of training data").create("t");
		Option predictionData = OptionBuilder.withArgName("prediction-data").hasArg().withDescription("path of prediction data").create("p");
		
		Option confFile = OptionBuilder.withArgName("configuration-file").hasArg().withDescription("path of configuration file").create("R");
		
		//Add command
		options.addOption(help);
		options.addOption(task);
		options.addOption(model);
		options.addOption(op);
		options.addOption(trainingData);
		options.addOption(predictionData);
		options.addOption(numIt);
		options.addOption(learningRate);
		options.addOption(lambda);
		options.addOption(lambdaS);
		options.addOption(lambdaV);
		options.addOption(lambdaA);
		options.addOption(lambdaB);
		options.addOption(lambdaC);
		options.addOption(numU);
		options.addOption(numV);
		options.addOption(numS);
		options.addOption(numD);
		options.addOption(featureU);
		options.addOption(featureV);
		options.addOption(confFile);
		
		return options;
	}
	
	/**
	 * Main function
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		CommandLineParser parser = new PosixParser();
		try {
			Options options = buildOption();
			CommandLine line = parser.parse(options, args);
			if (line.hasOption("help")) {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("MF", options);
				return;
			}
			/*Data parameter*/
			if (line.hasOption("m")) {
				numUser = Integer.parseInt(line.getOptionValue("m"));
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("MF", options);
				return;
			}
			if (line.hasOption("n")) {
				numItem = Integer.parseInt(line.getOptionValue("n"));
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("MF", options);
				return;
			}
			if (line.hasOption("s")) {
				numContext = Integer.parseInt(line.getOptionValue("s"));
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("MF", options);
				return;
			}
			if (line.hasOption("fu")){
				numUserFeature = Integer.parseInt(line.getOptionValue("fu"));
			}
			if (line.hasOption("fv")){
				numItemFeature = Integer.parseInt(line.getOptionValue("fv"));
			}
			/*Model parameter*/
			if (line.hasOption("d")) {
				numD = Integer.parseInt(line.getOptionValue("d"));
			}
			if (line.hasOption("i")) {
				numIt = Integer.parseInt(line.getOptionValue("i"));
			}
			if (line.hasOption("e")) {
				learningRate = Double.parseDouble(line.getOptionValue("e"));
			}
			if (line.hasOption("r")) {
				lambda = Double.parseDouble(line.getOptionValue("r"));
			}
			if (line.hasOption("rs")) {
				lambdaS = Double.parseDouble(line.getOptionValue("rs"));
			}
			if (line.hasOption("rv")) {
				lambdaV = Double.parseDouble(line.getOptionValue("rv"));
			}
			if (line.hasOption("ra")) {
				ra = Double.parseDouble(line.getOptionValue("ra"));
			}
			if (line.hasOption("rb")) {
				rb = Double.parseDouble(line.getOptionValue("rb"));
			}
			if (line.hasOption("rc")) {
				rc = Double.parseDouble(line.getOptionValue("rc"));
			}
			if(line.hasOption("R")){
				FileOperators.confName = line.getOptionValue("R");
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("MF", options);
				return;
			}
			/*Task parameter*/
			if (line.hasOption("T")) {
				String task = line.getOptionValue("T");
				if (!line.hasOption("M") || !line.hasOption("O")) return;
				modelName = line.getOptionValue("M");
				opName = line.getOptionValue("O");
				if (task.equalsIgnoreCase("training")) {
					if (!line.hasOption("t")) return;
					inputPath = line.getOptionValue("t");
					performTraining();
				} else if(task.equalsIgnoreCase("predictionPair")){
					if (!line.hasOption("p")) return;
					inputPath = line.getOptionValue("p");
					performPredictionPair();
				} else if(task.equalsIgnoreCase("predictionAll")){
					performPredictionAll();
				} else if(task.equalsIgnoreCase("predictionPairDist")){
					if (!line.hasOption("p")) return;
					inputPath = line.getOptionValue("p");
					performPredictionPairDist();
				}
			}
			
		} catch (ParseException exp) {
			System.err.println("Parsing failed.  Reason: " + exp.getMessage());
		}
	}
}
