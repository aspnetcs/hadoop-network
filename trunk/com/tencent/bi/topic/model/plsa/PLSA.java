package com.tencent.bi.topic.model.plsa;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

public abstract class PLSA {
	
	protected DenseDoubleMatrix2D pzw = null;
	
	protected DenseDoubleMatrix2D pzd = null;
	
	protected DenseDoubleMatrix1D pz = null;
	
	protected int m = 0;
	
	protected int n = 0;
	
	protected int z = 0;
	
	protected int numIt = 0;
	
	protected String inputPath = "";
	
	public void initModel(int m, int n, int z, int numIt, String inputPath){
		this.m = m;
		this.n = n;
		this.z = z;
		this.numIt = numIt;
		this.inputPath = inputPath;
		this.pzw = (DenseDoubleMatrix2D) DoubleFactory2D.dense.make(z, m);
		this.pzd = (DenseDoubleMatrix2D) DoubleFactory2D.dense.make(z, n);
		this.pz = (DenseDoubleMatrix1D) DoubleFactory1D.dense.make(z);
	}
	
	public abstract void buildModel() throws Exception;
}
