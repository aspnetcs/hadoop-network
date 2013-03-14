package com.tencent.bi.cf.optimization.gradient.hybrid;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

/**
 * Hybrid Square Loss
 * @author tigerzhong
 *
 */
public class SquareHybridLoss implements Loss {

	private static final long serialVersionUID = 1L;

	/**
	 * Get the loss value
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param B, regression matrix
	 * @param fu, user features
	 * @param fv, item features
	 * @param r, target value
	 * @return loss value
	 */
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double r){
		double p = getPrediction(u, v, B, fu, fv);
		return (p-r)*(p-r);
	}
	
	/**
	 * Get the gradient of latent vectors
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param B, regression matrix
	 * @param fu, user features
	 * @param fv, item features
	 * @param r, target value
	 * @param lambda, regurlarization parameter
	 * @return gradient
	 */
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double r, double lambda){
		int d = u.size();
		DoubleMatrix1D g = DoubleFactory1D.dense.make(d);
		double e = getPrediction(u, v, B, fu, fv) - r;
		for(int i=0;i<d;i++){
			double tg = e*v.get(i) + lambda*u.get(i);
			g.set(i, tg);
		}
		return g;
	}
	
	/**
	 * Get the gradient of regression matrix, B
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param B, regression matrix
	 * @param fu, user features
	 * @param fv, item features
	 * @param r, target value
	 * @param lambda, regurlarization parameter
	 * @return gradient
	 */
	public DoubleMatrix2D getGradientB(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double r, double lambda){
		double e = getPrediction(u, v, B, fu, fv) - r;
		DoubleMatrix2D g = DoubleFactory2D.dense.make(fu.size(),fv.size());
		for(int i=0;i<fu.size();i++)
			for(int j=0;j<fv.size();j++)
				g.set(i, j, e*fu.get(i)*fv.get(j)+lambda*B.getQuick(i,j));
		return g;
	}
	
	/**
	 * Get the prediction value
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param B, regression matrix
	 * @param fu, user features
	 * @param fv, item features
	 * @return prediction value
	 */
	public double getPrediction(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv){
		DoubleMatrix1D tu = new DenseDoubleMatrix1D(fu.size());
		for(int i=0;i<fv.size();i++){
			tu.set(i, fu.zDotProduct(B.viewColumn(i)));
		}
		double opVal = tu.zDotProduct(fv);
		return u.zDotProduct(v) + opVal;
	}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	@Deprecated
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
		//Ignore!!
		return 0;
	}

	@Deprecated
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			double r, double lambda) {
		//Ignore!!
		return null;
	}

}
