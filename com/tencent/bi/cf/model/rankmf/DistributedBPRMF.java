package com.tencent.bi.cf.model.rankmf;

import org.apache.hadoop.conf.Configuration;

import cern.colt.matrix.DoubleMatrix1D;

import com.tencent.bi.cf.model.AbstractMF;
import com.tencent.bi.cf.model.common.DistributedMFPrediction;
import com.tencent.bi.cf.model.common.MatrixInitialization;
import com.tencent.bi.cf.optimization.gradient.rank.AUCLoss;
import com.tencent.bi.cf.optimization.gradient.rank.DistributedSGDBPR;
import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * Distributed Bayesian Personalized Ranking with MF
 * @author tigerzhong
 *
 */
public class DistributedBPRMF extends AbstractMF {
	
	@Override
	public void initModel(int m, int n, int d, String solverName, double lambda, double learningRate, int numIt, String inputPath) throws Exception{
		super.initModel(-1, n, d, "", lambda, learningRate, numIt, inputPath);
		solver = (DistributedSGDBPR) Class.forName(solverName).newInstance();
		//Initialize U
		Configuration conf = FileOperators.getConfiguration();
		MatrixInitialization obj = new MatrixInitialization();
		obj.init(d, 0, inputPath, conf.get("hadoop.cache.path")+"U/",true);
		obj.perform(conf);
	}
	
	@Override
	public void buildModel() throws Exception {
		//Preprocess
		Configuration conf = FileOperators.getConfiguration();
		PairwiseGenerator.getPariwise(super.inputPath, conf.get("hadoop.tmp.path")+"Pairwise/");
		AUCLoss lossFunc = new AUCLoss();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, super.numD);
			solver.update(null, super.V, conf.get("hadoop.tmp.path")+"Pairwise/", conf.get("hadoop.output.path")+"DistributedBPRMF/"+i+"/");
		}
	}

	@Override
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s){
		return u.zDotProduct(v);
	}
	
	@Override
	public void predictPair(String inputPath, int numD) throws Exception {
		DistributedMFPrediction.distributV();
		Configuration conf = FileOperators.getConfiguration();
		DistributedMFPrediction.predictPair(inputPath, conf.get("hadoop.output.path"), "com.tencent.bi.cf.model.rankmf.DistributedBPRMF", numD);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	@Deprecated
	public double predict(int p, int q, int o) {
		//Ignore!!
		return 0.0;
	}

	@Override
	public void predictAll() throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		DistributedMFPrediction.predictAll(conf.get("hadoop.output.path"), "com.tencent.bi.cf.model.rankmf.DistributedBPRMF");			
	}
}
