package com.tencent.bi.cf.optimization.gradient.hybrid;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

/**
 * Regression Square Loss
 * @author tigerzhong
 *
 */
public class SquareRegressionLoss implements Loss {

	private static final long serialVersionUID = 1L;

	/**
	 * Get the loss value
	 * @param B, regression matrix
	 * @param fu, user features
	 * @param fv, item features
	 * @param r, target value
	 * @return loss value
	 */
	public double getValue(DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double r){
		double p = getPrediction(B, fu, fv);
		return (p-r)*(p-r);
	}
	
	/**
	 * Get the gradient of regression matrix, B
	 * @param B, regression matrix
	 * @param fu, user features
	 * @param fv, item features
	 * @param r, target value
	 * @param lambda, regurlarization parameter
	 * @return gradient
	 */
	public DoubleMatrix2D getGradientB(DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv, double r, double lambda){
		double e = getPrediction(B, fu, fv) - r;
		DoubleMatrix2D g = DoubleFactory2D.dense.make(fu.size(),fv.size());
		for(int i=0;i<fu.size();i++)
			for(int j=0;j<fv.size();j++)
				g.set(i, j, e*fu.get(i)*fv.get(j)+lambda*B.getQuick(i,j));
		return g;
	}
	
	/**
	 * Get the prediction value
	 * @param B, regression matrix
	 * @param fu, user features
	 * @param fv, item features
	 * @return prediction value
	 */
	public double getPrediction(DoubleMatrix2D B, DoubleMatrix1D fu, DoubleMatrix1D fv){
		DoubleMatrix1D tu = new DenseDoubleMatrix1D(fu.size());
		for(int i=0;i<fv.size();i++){
			tu.set(i, fu.zDotProduct(B.viewColumn(i)));
		}
		return tu.zDotProduct(fv);
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
