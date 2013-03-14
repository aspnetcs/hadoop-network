package com.tencent.bi.cf.model.tensor;

import com.tencent.bi.cf.model.AbstractMF;
import com.tencent.bi.cf.optimization.gradient.tensor.TensorSolver;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * Abstract Class for Tensor or CMF
 * @author tigerzhong
 *
 */
public abstract class AbstractTensor extends AbstractMF {
	
	/**
	 * Matrix for context
	 */
	protected DenseDoubleMatrix2D S = null;
	/**
	 * Optimization
	 */
	protected TensorSolver solver = null;
	/**
	 * Initialize model
	 * @param m, number of user
	 * @param n, number of item
	 * @param s, number of context
	 * @param d, number of latent dimensions
	 * @param solverName, name of solver
	 * @param lambda, regularization parameter
	 * @param learningRate, learning rate
	 * @param numIt, number of iterations
	 * @param inPath, input path for data
	 * @param outPath, output path for training results
	 * @throws Exception
	 */
	public void initModel(int m, int n, int s, int d, String solverName, double lambda, double learningRate, int numIt, String inPath) throws Exception{
		super.initModel(m, n, d, "", lambda, learningRate, numIt, inPath);
		S = (DenseDoubleMatrix2D) DoubleFactory2D.dense.random(s,d);
		solver = (TensorSolver) Class.forName(solverName).newInstance();
	}

	public void setS(DenseDoubleMatrix2D s) {
		S = s;
	}
}
