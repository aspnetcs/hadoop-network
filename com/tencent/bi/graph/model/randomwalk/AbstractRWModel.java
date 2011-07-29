package com.tencent.bi.graph.model.randomwalk;

import java.io.IOException;

/**
 * Abstract model for random walk
 * @author tigerzhong
 *
 */
public abstract class AbstractRWModel {
	
	/**
	 * Number of nodes
	 */
	protected static int numNode = 1;
	/**
	 * Number of importance values
	 */
	protected static int numValues = 1;
	/**
	 * Trade-off parameter between update and re-start
	 */
	protected static double alpha = 0.9;
	/**
	 * Number of iterations
	 */
	protected int numIt = 10;
	/**
	 * Input path
	 */
	protected String inputPath = "";
	/**
	 * Output path
	 */
	protected String adListPath = "";
	
	protected boolean isADList = false;
	
	public void initModel(int numNode, int numValues, int numIt, double alpha, String inputPath, String adListPath) throws IOException, InterruptedException, ClassNotFoundException{
		AbstractRWModel.numNode = numNode;
		AbstractRWModel.numValues = numValues;
		AbstractRWModel.alpha = alpha;
		this.inputPath = inputPath;
		this.adListPath = adListPath;
		this.numIt = numIt;
	}
	
	public void buildModel() throws Exception {
		for(int i=0;i<numIt;i++){
			performRandomWalk(i);
		}
	}
	
	protected abstract void performRandomWalk(int numIt) throws Exception;

}
