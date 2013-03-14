package com.tencent.bi.cf.optimization.gradient.rank;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.jet.math.Mult;
import cern.jet.math.PlusMult;

import com.tencent.bi.cf.optimization.Loss;

/**
 * AUC Loss
 * @author tigerzhong
 * 1/(1+Math.exp(-(uv1-uv2))), r1>r2
 */
public class AUCLoss implements Loss {

	private static final long serialVersionUID = 1L;
	
	/**
	 * Get the loss value
	 * @param u, user latent vector
	 * @param v1, latent vector of item i
	 * @param v2, latent vector of item j
	 * @return loss value
	 */
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v1, DoubleMatrix1D v2) {
		double r1 = u.zDotProduct(v1);
		double r2 = u.zDotProduct(v2);
		return Math.log(1/(1+Math.exp(-(r1-r2))));
	}
	
	/**
	 * Get the gradients
	 * @param u, user latent vector
	 * @param v1, latent vector of item i
	 * @param v2, latent vector of item j
	 * @param c, category; 1: user; 2: v1; 3: v2;
	 * @param lambda, regularization parameter
	 * @return
	 */
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v1, DoubleMatrix1D v2, int c, double lambda) {
		double r_uij = u.zDotProduct(v1) - u.zDotProduct(v2);
		double dr = Math.exp(-r_uij)/(1+Math.exp(-r_uij));
		DoubleMatrix1D dt = new DenseDoubleMatrix1D(u.size());
		DoubleMatrix1D rvec = new DenseDoubleMatrix1D(u.size());
		if(c==1) {
			dt = dt.assign(v1);
			dt = dt.assign(v2, PlusMult.plusMult(-1));
			rvec = rvec.assign(u);
		} else if (c==2){
			dt = dt.assign(u);
			rvec = rvec.assign(v1);
		} else if (c==3){
			dt = dt.assign(u);
			dt = dt.assign(Mult.mult(-1));
			rvec = rvec.assign(v2);
		}
		dt.assign(Mult.mult(dr));
		dt = dt.assign(rvec, PlusMult.plusMult(lambda));
		dt.assign(Mult.mult(-1));
		return dt;
	}
	
/////////////////////////////////////////////////////////////////////////////////////////////
	
	@Deprecated
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
		//Ignore!!
		return 0;
	}

	@Deprecated
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			double r, double lambda) {
		// Ignore!!
		return null;
	}

}
