package com.tencent.bi.cf.optimization;

import java.io.Serializable;

import cern.colt.matrix.DoubleMatrix1D;

/**
 * Inteface for Loss Function
 * @author tigerzhong
 *
 */
public interface Loss extends Serializable{
	/**
	 * Get the value of loss function
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param r, target value
	 * @return
	 */
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r);
	/**
	 * Get the gradients of this kind of loss on user latent vector.
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param r, target value
	 * @param lambda, regularization parameter
	 * @return gradients on user latent vector
	 */
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v, double r, double lambda);
}
