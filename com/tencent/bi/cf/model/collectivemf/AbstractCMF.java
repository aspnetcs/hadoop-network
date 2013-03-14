package com.tencent.bi.cf.model.collectivemf;

import com.tencent.bi.cf.model.AbstractMF;

/**
 * Abstract class for CMF 
 * @author tigerzhong
 *
 */
public abstract class AbstractCMF extends AbstractMF {
	
	/**
	 * Initialize model
	 * @param m, number of users
	 * @param n, number of items
	 * @param s, number of contexts
	 * @param d, number of latent dimensions
	 * @param solverName, name of optimization method
	 * @param lambdaV, trade-off parameter for R_v
	 * @param lambdaS, trade-off parameter for R_s
	 * @param lambda, reguralization parameter
	 * @param learningRate, learning rate
	 * @param inPath, input path
	 * @throws Exception
	 */
	public abstract void initModel(int m, int n, int s, int d, String solverName, double lambdaV, double lambdaS, double lambda, double learningRate, int numIt, String inPath) throws Exception;

}
