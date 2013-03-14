package com.tencent.bi.cf.model.knn;

import com.tencent.bi.cf.model.CF;

/**
 * KNN for Collaborative Filtering
 * @author tigerzhong
 *
 */
public class KNN implements CF {
	/**
	 * Input path
	 */
	protected String inputPath;
	/**
	 * Number of neighbors
	 */
	protected int K;
	/**
	 * Name of criterion
	 */
	protected String c;
	/**
	 * Modification term
	 */
	protected double lambda;
	/**
	 * User-base or item-base
	 */
	protected boolean userBase;
	/**
	 * Number of users
	 */
	protected int m;
	/**
	 * Number of items
	 */
	protected int n;
	/**
	 * Initialize models
	 * @param inputPath
	 * @param K
	 * @param criterion
	 * @param lambda
	 * @param userBase
	 * @param m
	 * @param n
	 */
	public void initModel(String inputPath, int K, String criterion, double lambda, boolean userBase, int m, int n){
		this.inputPath = inputPath;
		this.K = K;
		this.c = criterion;
		this.lambda = lambda;
		this.userBase = userBase;
		this.m = m;
		this.n = n;
	}
	
	@Override
	public void buildModel() throws Exception {
		InverseIndexBuilder.build(inputPath, K, c, lambda, userBase, m, n);	//build inverse index
		NgBuilder.build(K, c, lambda, userBase, m, n);						//build neighborhood 
	}

	@Deprecated
	public void predictPair(String inputPath, int numD) throws Exception {
		String[] items = inputPath.split("|");
		KNNPredictionPair.predict(items[0], items[1], numD==1);
	}
	
	@Override
	public void predictAll() throws Exception {
		KNNPredictionAll.predict(inputPath, userBase);
	}

}
