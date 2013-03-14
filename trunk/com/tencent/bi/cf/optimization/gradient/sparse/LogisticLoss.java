package com.tencent.bi.cf.optimization.gradient.sparse;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;

/**
 * Loss using Logistic Transform
 * @author tigerzhong
 *
 */
public class LogisticLoss implements Loss  {

	private static final long serialVersionUID = 1L;
	/**
	 * lower and upper bounds
	 */
	double r0=0.0, r1=1.0;
	/**
	 * Set bounds
	 * @param r0
	 * @param r1
	 */
	public void setBounds(double r0, double r1){
		this.r0 = r0;
		this.r1 = r1;
	}
	
	@Override
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
		double lp = getLogisticValue(u.zDotProduct(v));
		r = (r-r0)/(r1-r0);
		return (lp-r)*(lp-r);
	}

	@Override
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			double r, double lambda) {
		/*
		 p = 1/(1+exp(-uv)); 
		 r = (r-r0)/(r1-r0);
		 */
		double lg = getLogisticValue(u.zDotProduct(v));
		double e = lg - (r-r0)/(r1-r0);
		int d = u.size();
		DoubleMatrix1D g = DoubleFactory1D.dense.make(d);
		for(int i=0;i<d;i++){
			double tg = e*lg*(1-lg)*v.get(i)+lambda*u.get(i);
			g.set(i, tg);
		}
		return g;
	}
	
	/**
	 * Get logistic value
	 * @param v
	 * @return  1/(1+exp(-v))
	 */
	private double getLogisticValue(double v){
		return 1/(1+Math.exp(-v));
	}
	
}
