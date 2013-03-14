package com.tencent.bi.cf.optimization.gradient.hybrid;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

/**
 * Square Loss Function with Bias for Hybrid MF
 * @author tigerzhong
 *
 */
public class SquareBiasHybridLoss implements Loss {
	/**
	 * Serial number
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * Get the loss value
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param B, regression matrix
	 * @param fu, user feature matrix
	 * @param fv, item feature matrix
	 * @param uBias, user bias
	 * @param vBias, item bias
	 * @param r, target value
	 * @return loss value
	 */
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double uBias, double vBias, double r){
		double p = getPrediction(u, v, B, fu, fv, uBias, vBias);
		return (p-r)*(p-r);
	}
	
	/**
	 * Get the gradient of bias
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param B, regression matrix
	 * @param fu, user feature matrix
	 * @param fv, item feature matrix
	 * @param uBias, user bias
	 * @param vBias, item bias
	 * @param r, target value
	 * @param lambda, reguralization parameter
	 * @return gradient value
	 */
	public double getGradientBias(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double uBias, double vBias, double r, double lambda) {
		double e = getPrediction(u, v, B, fu, fv, uBias, vBias) - r;
		return e + lambda*uBias;
	}
	
	/**
	 * Get the gradient of latent matrix
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param B, regression matrix
	 * @param fu, user feature matrix
	 * @param fv, item feature matrix
	 * @param uBias, user bias
	 * @param vBias, item bias
	 * @param r, target value
	 * @param lambda, reguralization parameter
	 * @return gradient value
	 */
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double uBias, double vBias, double r, double lambda){
		int d = u.size();
		DoubleMatrix1D g = DoubleFactory1D.dense.make(d);
		double e = getPrediction(u, v, B, fu, fv, uBias, vBias) - r;
		for(int i=0;i<d;i++){
			double tg = e*v.get(i) + lambda*u.get(i);
			g.set(i, tg);
		}
		return g;
	}
	
	/**
	 * Get the gradient of regression matrix
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param B, regression matrix
	 * @param fu, user feature matrix
	 * @param fv, item feature matrix
	 * @param uBias, user bias
	 * @param vBias, item bias
	 * @param r, target value
	 * @param lambda, reguralization parameter
	 * @return gradient value
	 */
	public DoubleMatrix2D getGradientB(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double uBias, double vBias, double r, double lambda){
		double e = getPrediction(u, v, B, fu, fv, uBias, vBias) - r;
		DoubleMatrix2D g = DoubleFactory2D.dense.make(fu.size(),fv.size());
		for(int i=0;i<fu.size();i++)
			for(int j=0;j<fv.size();j++)
				g.set(i, j, e*fu.get(i)*fv.get(j)+lambda*B.getQuick(i,j));
		return g;
	}
	
	/**
	 * Get the prediction value, r = uv' + uBias + vBias + f_u*B*f_v
	 * @param u, user latent vector
	 * @param v, item latent vector
	 * @param B, regression matrix
	 * @param fu, user feature matrix
	 * @param fv, item feature matrix
	 * @param uBias, user bias
	 * @param vBias, item bias
	 * @return prediction value
	 */
	public double getPrediction(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double uBias, double vBias){
		//f_u*B*f_v
		DoubleMatrix1D tu = new DenseDoubleMatrix1D(fv.size());
		for(int i=0;i<fv.size();i++)
			tu.set(i, fu.zDotProduct(
					B.viewColumn(i)));
		//uv'
		double opVal = tu.zDotProduct(fv);
		//uv' + uBias + vBias + f_u*B*f_v
		return u.zDotProduct(v) + opVal;// + uBias + vBias;
	}
	
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	@Deprecated
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
		//Ignore!
		return 0;
	}

	@Deprecated
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			double r, double lambda) {
		//Ignore!
		return null;
	}

}
