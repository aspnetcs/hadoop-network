package com.tencent.bi.driver;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

import com.tencent.bi.graph.model.randomwalk.AbstractRWModel;
import com.tencent.bi.graph.model.search.BFSPair;
//import com.tencent.bi.graph.model.search.Reachable;
import com.tencent.bi.graph.statistic.FirstOrder;
import com.tencent.bi.graph.utility.ADListGenerator;
import com.tencent.bi.graph.utility.ImportanceGenerator;
import com.tencent.bi.graph.utility.NumberLinksGenerator;
import com.tencent.bi.utils.hadoop.FileOperators;

public class GraphDriver {

	private static String modelName;
	/**
	 * Data input path
	 */
	private static String inputPath;
	/**
	 * Data output path
	 */
	private static String outputPath;
	/**
	 * Number of iterations
	 */
	private static int numIt = 100;
	/**
	 * Number of reguralization parameter
	 */
	private static double alpha = 0.9;
	
	private static int numNodes = 0;
	
	private static int numValues = 1;
	
	@SuppressWarnings("unused")
	private static long startNode = -1;
	
	@SuppressWarnings("unused")
	private static long endNode = -1;
	
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
		
		//All
		Option help = new Option("help", "print this message");
		Option model = OptionBuilder.withArgName("model").hasArg().withDescription("model").create("M");
		
		//Model parameter
		Option numIt = OptionBuilder.withArgName("iteration").hasArg().withDescription("number of iteration").create("i");
		Option numNodes = OptionBuilder.withArgName("number-nodes").hasArg().withDescription("Number of Nodes").create("n");
		
		//For pagerank
		Option alpha = OptionBuilder.withArgName("alpha").hasArg().withDescription("trade-off parameter").create("a");
		Option numValues = OptionBuilder.withArgName("number-values").hasArg().withDescription("Number of Importance Values").create("v");
		Option useADList = new Option("ad", "adjacency list or pair?");
		
		//For BFS
		Option startNode = OptionBuilder.withArgName("start-node").hasArg().withDescription("Start Node").create("na");
		Option endNode = OptionBuilder.withArgName("end-node").hasArg().withDescription("Target Node").create("nb");
		
		//For Statistic
		Option method = OptionBuilder.withArgName("method").hasArg().withDescription("Statistic Method").create("F");
		
		//Path
		Option inputPath = OptionBuilder.withArgName("input-path").hasArg().withDescription("input path").create("pi");
		Option outputPath = OptionBuilder.withArgName("output-path").hasArg().withDescription("output path").create("po");

		Option confFile = OptionBuilder.withArgName("configuration-file").hasArg().withDescription("path of configuration file").create("R");
		
		//Add command
		options.addOption(help);
		options.addOption(method);
		options.addOption(model);
		options.addOption(numNodes);
		options.addOption(alpha);
		options.addOption(numIt);
		options.addOption(numValues);
		options.addOption(startNode);
		options.addOption(endNode);
		options.addOption(inputPath);
		options.addOption(outputPath);
		options.addOption(confFile);
		options.addOption(useADList);
		
		return options;
	}
	
	/**
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
			//Parameter
			if (line.hasOption("n")) {
				numNodes = Integer.parseInt(line.getOptionValue("n"));
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("Graph", options);
				return;
			}
			if (line.hasOption("i")) {
				numIt = Integer.parseInt(line.getOptionValue("i"));
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("Graph", options);
				return;
			}
			if (line.hasOption("v")) {
				numValues = Integer.parseInt(line.getOptionValue("v"));
			}
			if (line.hasOption("a")) {
				alpha = Double.parseDouble(line.getOptionValue("a"));
			}
			if (line.hasOption("na")) {
				startNode = Long.parseLong(line.getOptionValue("na"));
			}
			if (line.hasOption("nb")) {
				endNode = Long.parseLong(line.getOptionValue("nb"));
			}
			//Path
			if (line.hasOption("pi")) {
				inputPath = line.getOptionValue("pi");
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("Graph", options);
				return;
			}
			if (line.hasOption("po")) {
				outputPath = line.getOptionValue("po");
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("Graph", options);
				return;
			}
			if(line.hasOption("R")){
				FileOperators.confName = line.getOptionValue("R");
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("MF", options);
				return;
			}
			//Model
			if(line.hasOption("M")){
				modelName = line.getOptionValue("M");
				if(modelName.indexOf("randomwalk")>-1){	//RW
					AbstractRWModel model = (AbstractRWModel) Class.forName(modelName).newInstance();
					model.initModel(numNodes, numValues, numIt, alpha, inputPath, outputPath);
					model.buildModel();
				} else if (modelName.indexOf("search")>-1){	//Search
					//Reachable model = (Reachable) Class.forName(modelName).newInstance();
					BFSPair model = new BFSPair();
					model.initModel(numIt, inputPath);
					model.performSearch();
				} else if(modelName.indexOf("ADListGenerator")>-1){	//utility
					ADListGenerator.generateADList(numNodes, 1, inputPath, outputPath);
				} else if(modelName.indexOf("ImportanceGenerator")>-1){ //utility
					if(line.hasOption("ad")) ImportanceGenerator.generateValueList(numValues, numNodes, true, inputPath, outputPath);
					else ImportanceGenerator.generateValueList(numValues, numNodes, false, inputPath, outputPath);
				} else if(modelName.indexOf("NumberLinksGenerator")>-1) {
					if(line.hasOption("ad")) NumberLinksGenerator.getNumLinks(inputPath, outputPath, true);
					else NumberLinksGenerator.getNumLinks(inputPath, outputPath, false);
				} else if(modelName.indexOf("FirstOrder")>-1){
					if (!line.hasOption("F")) {
						HelpFormatter formatter = new HelpFormatter();
						formatter.printHelp("Graph", options);
						return;
					} 
					String funcName = line.getOptionValue("F");
					if(funcName.equals("getNumEdges"))
						FirstOrder.getNumEdges(inputPath);
					else if(funcName.equals("getFollowDistribution")){
						FirstOrder.getFollowDistribution(inputPath, 0);
						FirstOrder.getFollowDistribution(inputPath, 1);
					}
				}
			} else {
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("Graph", options);
				return;
			}
						
		} catch (ParseException exp) {
			System.err.println("Parsing failed.  Reason: " + exp.getMessage());
		}
	}

}
