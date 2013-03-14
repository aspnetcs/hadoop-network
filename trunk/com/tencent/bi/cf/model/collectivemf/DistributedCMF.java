package com.tencent.bi.cf.model.collectivemf;

import org.apache.hadoop.conf.Configuration;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

import com.tencent.bi.cf.model.common.DistributedMFPrediction;
import com.tencent.bi.cf.model.common.MatrixInitialization;
import com.tencent.bi.cf.optimization.gradient.collective.DistributedSGDCollective;
import com.tencent.bi.cf.optimization.gradient.sparse.SquareLoss;
import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * Collective Matrix Factorization
 * R_u = UV'; R_v = US'; R_s = VS'
 * @author tigerzhong
 *
 */
public class DistributedCMF extends AbstractCMF {
	/**
	 * Matrix for context
	 */
	protected DenseDoubleMatrix2D S = null;
	/**
	 * Slover
	 */
	protected DistributedSGDCollective solver = null;
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
		super.initModel(-1, n, d, "", lambda, learningRate, numIt, inPath);
		S = (DenseDoubleMatrix2D) DoubleFactory2D.dense.random(s,d);
		solver = (DistributedSGDCollective) Class.forName(solverName).newInstance();
		this.lambdaV = lambdaV;
		this.lambdaS = lambdaS;
		//Initialize U
		Configuration conf = FileOperators.getConfiguration();
		MatrixInitialization obj = new MatrixInitialization();
		obj.init(d, 0, this.inputPath + "MU/", conf.get("hadoop.cache.path")+"U/",true);
		obj.perform(conf); 
	}

	@Override
	public void buildModel() throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		SquareLoss lossFunc = new SquareLoss();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, lambdaV, lambdaS, super.learningRate, super.V.columns());
			solver.update(super.U, super.V, S, super.inputPath, conf.get("hadoop.output.path")+"DistributedCMF/"+i+"/");
		}
	}

	@Override
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s) {
		return u.zDotProduct(v);
	}

	@Override
	public void predictPair(String inputPath, int numD) throws Exception {
		DistributedMFPrediction.distributV();
		Configuration conf = FileOperators.getConfiguration();
		DistributedMFPrediction.predictPair(inputPath, conf.get("hadoop.output.path"), "com.tencent.bi.cf.model.sparsemf.GaussianMF", numD);
	}

	@Override
	public void predictAll() throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		DistributedMFPrediction.predictAll(conf.get("hadoop.output.path"), "com.tencent.bi.cf.model.tensor.DistributedCMF");			
	}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	@Deprecated
	public double predict(int p, int q, int o) {
		return U.viewRow(p).zDotProduct(V.viewRow(q));	
	}
}
