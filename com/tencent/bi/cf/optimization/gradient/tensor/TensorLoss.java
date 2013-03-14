package com.tencent.bi.cf.optimization.gradient.tensor;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleMatrix1D;

/**
 * Interface of Loss Function for Tensor
 * @author tigerzhong
 *
 */
public interface TensorLoss extends Loss {
	/**
	 * Get the loss value
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param s, context latent vector
	 * @param r, target value
	 * @return loss value
	 */
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s, double r);
	/**
	 * Get the gradients
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param s, context latent vector
	 * @param r, target value
	 * @param lambda, reguralization parameter
	 * @return gradients
	 */
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,  DoubleMatrix1D s, double r, double lambda);
}
