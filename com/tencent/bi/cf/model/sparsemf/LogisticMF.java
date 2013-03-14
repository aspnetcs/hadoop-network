package com.tencent.bi.cf.model.sparsemf;

import org.apache.hadoop.conf.Configuration;

import cern.colt.matrix.DoubleMatrix1D;

import com.tencent.bi.cf.model.AbstractMF;
import com.tencent.bi.cf.model.common.DistributedMFPrediction;
import com.tencent.bi.cf.model.common.MatrixInitialization;
import com.tencent.bi.cf.optimization.gradient.sparse.LogisticLoss;
import com.tencent.bi.utils.hadoop.FileOperators;


/**
 * Matrix Factorization with Logistic Modification
 * @author tigerzhong
 *
 */
public class LogisticMF extends AbstractMF  {
	/**
	 * Upper and lower bounds
	 */
	private double r0=0.0, r1=1.0;
	
	private LogisticLoss lossFunc = new LogisticLoss();
	
	/**
	 * Initialize Model
	 */
	public void initModel(int m, int n, int d, String solverName, double lambda, double learningRate, int numIt, String inPath) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		if(solverName.indexOf("Distributed")>-1){	//For huge U but small V
			super.initModel(-1, n, d, solverName, lambda, learningRate, numIt, inPath);
			MatrixInitialization obj = new MatrixInitialization();
			obj.init(d, 0, inputPath, conf.get("hadoop.cache.path")+"U/",true);
			obj.perform(conf);
		} else super.initModel(m, n, d, solverName, lambda, learningRate, numIt, inPath);	//For small U and V
	}
	
	@Override
	public void buildModel() throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		lossFunc.setBounds(r0, r1);
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, super.numD);
			solver.update(super.U, super.V, super.inputPath, conf.get("hadoop.output.path")+"LogMF/"+i+"/");
		}
		if(solverName.indexOf("Distributed")>-1){	//transform V
			DistributedMFPrediction.distributV();
		}
	}

	@Override
	public double predict(int p, int q, int o) {
		/*1/(1+exp(-uv)*/
		double a = 1.0/(1.0+Math.exp(-U.viewRow(p).zDotProduct(V.viewRow(q))));
		return a*(r1-r0)+r0;
	}

	public void setR0(double r0) {
		this.r0 = r0;
	}

	public void setR1(double r1) {
		this.r1 = r1;
	}
	
	@Override
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s){
		return 1/(1+Math.exp(-u.zDotProduct(v)))*(r1-r0)+r0;
	}
	
	@Override
	public void predictPair(String inputPath, int numD) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		DistributedMFPrediction.predictPair(inputPath, conf.get("hadoop.output.path"), "com.tencent.bi.cf.model.sparsemf.LogMF", numD);
	}

	@Override
	public void predictAll() throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		DistributedMFPrediction.predictAll(conf.get("hadoop.output.path"), "com.tencent.bi.cf.model.sparsemf.LogMF");	
		
	}
}
