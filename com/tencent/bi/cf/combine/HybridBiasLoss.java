package com.tencent.bi.cf.combine;

import com.tencent.bi.cf.optimization.Loss;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

/**
 * Square Loss Function with Bias for Hybrid MF
 * 
 * @author tigerzhong
 * 
 */
@SuppressWarnings("unused")
public class HybridBiasLoss implements Loss {
	/**
	 * Serial number
	 */
	private static final long serialVersionUID = 1L;

	public double getValue(double p, double r) {
		return (p - r) * (p - r);
	}

	public double getGradientBias(double bias, double p, double r,
			double lambda, double rb) {
		double e = p - r;
		return rb * e + lambda * bias;
	}

	public double[] getGradient(double[] u, double[] v, double p, double r,
			double lambda) {
		double[] g = new double[u.length];
		double e = p - r;
		for (int i = 0; i < u.length; i++)
			g[i] = e * v[i] + lambda * u[i];
		return g;
	}

	public double[][] getGradientB(double[][] B, double[] fu, double[] fv,
			double p, double r, double lambda, double rf) {
		double e = p - r;
		double[][] g = new double[fu.length][fv.length];
		for (int i = 0; i < fu.length; i++)
			for (int j = 0; j < fv.length; j++)
				g[i][j] = rf * e * fu[i] * fv[j] + lambda * B[i][j];
		return g;
	}

	public double getPrediction(double[] u, double[] v, double[][] B,
			double[] fu, double[] fv, double uBias, double vBias, double ru,
			double ri, double rf) {
		// f_u*B*f_v
		double[] tu = new double[fv.length];
		for (int i = 0; i < fv.length; i++)
			for (int j = 0; j < fu.length; j++)
				tu[i] += fu[j] * B[j][i];
		double fuBv = 0.0;
		for (int i = 0; i < fv.length; i++)
			fuBv += tu[i] * fv[i];
		// uv'
		double uv = 0.0;
		for (int i = 0; i < u.length; i++)
			uv += u[i] * v[i];
		// uv' + uBias + vBias + f_u*B*f_v
		return uv + rf * fuBv + ru * uBias + ri * vBias;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	@Deprecated
	public double getValue(DoubleMatrix1D u, DoubleMatrix1D v, double r) {
		return 0;
	}

	@Deprecated
	public DoubleMatrix1D getGradient(DoubleMatrix1D u, DoubleMatrix1D v,
			double r, double lambda) {
		return null;
	}

}
