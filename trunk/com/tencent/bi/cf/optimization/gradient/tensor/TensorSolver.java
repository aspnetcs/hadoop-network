package com.tencent.bi.cf.optimization.gradient.tensor;

import com.tencent.bi.cf.optimization.MFSolver;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * Interface of optimization for Tensor
 * @author tigerzhong
 *
 */
public interface TensorSolver extends MFSolver {
	/**
	 * Update the latent matrices
	 * @param U, latent matrix for users
	 * @param V, latent matrix for items
	 * @param S, latent matrix for contexts
	 * @param inPath, input path for data
	 * @param outPath, output path for training results
	 * @throws Exception
	 */
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, DenseDoubleMatrix2D S, String inPath, String outPath) throws Exception;	
}
