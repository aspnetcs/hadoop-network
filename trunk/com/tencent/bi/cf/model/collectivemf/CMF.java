package com.tencent.bi.cf.model.collectivemf;

import org.apache.hadoop.conf.Configuration;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

import com.tencent.bi.cf.optimization.gradient.collective.SGDCollective;
import com.tencent.bi.cf.optimization.gradient.sparse.SquareLoss;
import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * Collective Matrix Factorization
 * R_u = UV'; R_v = US'; R_s = VS'
 * @author tigerzhong
 *
 */
public class CMF extends AbstractCMF {
	/**
	 * Matrix for context
	 */
	protected DenseDoubleMatrix2D S = null;
	/**
	 * Slover
	 */
	protected SGDCollective solver = null;
	/**
	 * Regurlarization parameter for R_v
	 */
	protected double lambdaV = 0.0;
	/**
	 * Regurlarization parameter for R_s
	 */
	protected double lambdaS = 0.0;
	
	@Override
	public void initModel(int m, int n, int s, int d, String solverName, double lambdaV, double lambdaS, double lambda, double learningRate, int numIt, String inPath) throws Exception{
		super.initModel(m, n, d, "", lambda, learningRate, numIt, inPath);
		S = (DenseDoubleMatrix2D) DoubleFactory2D.dense.random(s,d);
		solver = (SGDCollective) Class.forName(solverName).newInstance();
		this.lambdaV = lambdaV;
		this.lambdaS = lambdaS;
	}

	@Override
	public void buildModel() throws Exception {
		SquareLoss lossFunc = new SquareLoss();
		Configuration conf = FileOperators.getConfiguration();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, lambdaV, lambdaS, super.learningRate, super.V.columns());
			solver.update(super.U, super.V, S, super.inputPath, conf.get("hadoop.output.path")+"CMF/"+i+"/");
		}
	}

	@Override
	public double predict(int p, int q, int o) {
		return U.viewRow(p).zDotProduct(V.viewRow(q));	
	}

	@Deprecated
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s) { return 0; 	}

	@Deprecated
	public void predictPair(String inputPath, int numD) throws Exception { 	}

	@Deprecated
	public void predictAll() throws Exception { }
}
