package com.tencent.bi.cf.optimization;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * Interface of MF Solver.
 * <p>
 * A MF-based Collaborative Filtering model will use a MF solver in each step of CF model building.
 * @author tigerzhong
 */
public interface MFSolver {
	/**
	 * Initialize this MF solver with specified parameters.
	 * @param lossFunc loss function
	 * @param lambda regularization parameter
	 * @param learningRate learning rate
	 * @param numD number of latent dimensions
	 * @throws Exception
	 */
	public void initialize(Loss lossFunc, double lambda, double learningRate, int numD) throws Exception;
	/**
	 * Update method. Update the model (U and V) with the specified input data in inputPath. This is a step
	 * in an iterative model building procedure.
	 * @param U latent matrix for user
	 * @param V latent matrix for item
	 * @param inputPath path for input data
	 * @param outputPath path for output training results
	 * @throws Exception
	 */
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, String inputPath, String outputPath) throws Exception;	
}
