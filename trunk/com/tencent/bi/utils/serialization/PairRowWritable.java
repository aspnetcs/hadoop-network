package com.tencent.bi.utils.serialization;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.Writable;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

/**
 * Class for Rating-Factors Output
 * [uid1,ri1,uid2,ri2...,v_i]
 * @author tigerzhong
 *
 */
public class PairRowWritable implements Writable {

	/**
	 * ID of each item
	 */
	private long[] rowIDs;
	/**
	 * Rating value of each item
	 */
	private double[] ratings;
	/**
	 * Number of all items
	 */
	private long len;
	/**
	 * Number of items with ratings
	 */
	private int nnz;
	/**
	 * Latent factors
	 */
	private double[] factors;
	/**
	 * Number of latent dimensions
	 */
	private int numD;
	
	@Override
	public void readFields(DataInput input) throws IOException {
		len = input.readLong();
		nnz = input.readInt();
		System.out.println(len+","+nnz);
		rowIDs = new long[nnz];
		ratings = new double[nnz];
		for(int i=0;i<nnz;i++)
			rowIDs[i] = input.readLong();
		for(int i=0;i<nnz;i++)
			ratings[i] = input.readDouble();
		numD = input.readInt();
		factors = new double[numD];
		for(int i=0;i<numD;i++)
			factors[i] = input.readDouble();
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeLong(len);
		output.writeInt(nnz);
		for(int i=0;i<nnz;i++)
			output.writeLong(rowIDs[i]);
		for(int i=0;i<nnz;i++)
			output.writeDouble(ratings[i]);
		output.writeInt(numD);
		for(int i=0;i<numD;i++)
			output.writeDouble(factors[i]);
	}

    public static PairRowWritable read(DataInput in) throws IOException {
    	PairRowWritable w = new PairRowWritable();
        w.readFields(in);
        return w;
    }
    
    public void setFectors(double[] vec){
    	numD = vec.length;
    	factors = vec.clone();
    }
    
    public void setFactors(DoubleMatrix1D vec){
    	numD = vec.size();
    	factors = vec.toArray();
    }
    
    public DoubleMatrix1D getFactors(){
    	DoubleMatrix1D vec = new DenseDoubleMatrix1D(factors);
    	return vec;
    }
    
    public double[] getFactorArray(){
    	return factors;
    }
    
    public void setRating(long id, double val){
    	len = id + 1;
    	nnz = 1;
    	rowIDs = new long[1]; rowIDs[0] = id;
    	ratings = new double[1]; ratings[0] = val;
    }
    
//    public void setRatings(SparseDoubleMatrix1D vector) {
//  		IntArrayList Y = new IntArrayList();
//  		DoubleArrayList R = new DoubleArrayList();
//  		vector.getNonZeros(Y, R);
//    	len = vector.size();
//    	nnz = R.size();
//    	rowIDs = new long[vector.];
//    	rowIDs = Y.elements();
//    	ratings = R.elements();
//    }
    
    public void setRatings(List<Long> ids, List<Double> vals, int size){
    	rowIDs = new long[ids.size()];
    	ratings = new double[vals.size()];
    	for(int i=0;i<ids.size();i++){
    		rowIDs[i] = ids.get(i);
    		ratings[i] = vals.get(i);
    	}
    	nnz = ids.size();
    	len = size;
    }
    
//    public SparseDoubleMatrix1D getRatingVector() {
//    	SparseDoubleMatrix1D vector = new SparseDoubleMatrix1D(len);
//    	for(int i=0; i<nnz; i++)
//    		vector.setQuick((int)rowIDs[i], ratings[i]);
//        return vector;
//    }
    
	public long[] getRowIDs() {
		return rowIDs;
	}

	public void setRowIDs(long[] rowIDs) {
		this.rowIDs = rowIDs;
	}

	public double[] getRatings() {
		return ratings;
	}

	public void setRatings(double[] ratings) {
		this.ratings = ratings;
	}
}
