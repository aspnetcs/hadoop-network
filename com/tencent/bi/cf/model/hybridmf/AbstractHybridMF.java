package com.tencent.bi.cf.model.hybridmf;

import com.tencent.bi.cf.model.AbstractMF;

/**
 * Abstract class for hybrid MF 
 * @author tigerzhong
 *
 */
public abstract class AbstractHybridMF extends AbstractMF {
	
	/**
	 * Initialize model
	 * @param m, number of users
	 * @param n, number of items
	 * @param d, number of latent dimensions
	 * @param fm, number of user features
	 * @param fn, number of item features
	 * @param solverName, optimization method
	 * @param lambda, reguralization parameter
	 * @param learningRate, learning rate
	 * @param numIt, number of iterations
	 * @param inputPath, input path for data
	 * @throws Exception
	 */
	public abstract void initModel(int m, int n, int fm, int fn, int d, String solverName, double lambda, double learningRate, int numIt, String inPath) throws Exception;

}
