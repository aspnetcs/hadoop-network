package com.tencent.bi.utils.serialization;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.Writable;

//import cern.colt.list.DoubleArrayList;
//import cern.colt.list.IntArrayList;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
//import cern.colt.matrix.impl.SparseDoubleMatrix1D;

/**
 * {@link Writable} Class for Matrix Row
 * @author tigerzhong
 * @weixue Will it be better to implement as a wrapper of {@link cern.colt.matrix.DoubleMatrix1D}?
 */
public class MatrixRowWritable implements Writable {
	/**
	 * Indices of non-zero elements
	 */
	private long[] rowIDs;
	/**
	 * Values of non-zero elements
	 * @weixue should be renamed to "vector".
	 */
	private double[] vectors;
	/**
	 * Number of dimensions 
	 */
	private long len;
	/**
	 * Number of non-zero elements
	 */
	private int nnz;
	/**
	 * Is sparse vector?
	 */
	private boolean sparse;
	
	@Override
	public void readFields(DataInput input) throws IOException {
		sparse = input.readBoolean();
		len = input.readLong();
		nnz = input.readInt();
		if(sparse){	//sparse
			rowIDs = new long[nnz];
			for(int i=0;i<nnz;i++)
				rowIDs[i] = input.readLong();
		}
		vectors = new double[nnz];
		for(int i=0;i<nnz;i++)
			vectors[i] = input.readDouble();
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeBoolean(sparse);
		output.writeLong(len);
		output.writeInt(nnz);
		if(sparse){	//sparse
			for(int i=0;i<nnz;i++)
				output.writeLong(rowIDs[i]);
		}
		for(int i=0;i<nnz;i++)
			output.writeDouble(vectors[i]);
	}
	
	/**
	 * Read data from datainput
	 * @param in
	 * @return writable object
	 * @throws IOException
	 */
    public static MatrixRowWritable read(DataInput in) throws IOException {
    	MatrixRowWritable w = new MatrixRowWritable();
        w.readFields(in);
        return w;
    }
    
    /**
     * Set one sparse element
     * @param id, index
     * @param val, value
     */
    public void set(long id, double val) {
    	len = id + 1;
    	nnz = 1;
    	rowIDs = new long[1]; rowIDs[0] = id;
    	vectors = new double[1]; vectors[0]=val;
    	sparse = true;
    }
    
//    /**
//     * Set a sparse vector
//     * @param vector
//     */
//    public void set(SparseDoubleMatrix1D vector) {
//  		IntArrayList Y = new IntArrayList();
//  		DoubleArrayList R = new DoubleArrayList();
//  		vector.getNonZeros(Y, R);
//    	len = vector.size();
//    	nnz = R.size();
//    	int[] tmp = Y.elements();
//    	for(int i=0;i<tmp.length;i++)
//    		rowIDs[i] = tmp[i]; 
//    	vectors = R.elements();
//    	sparse = true;
//    }
    
    public void set(long[] ids, double[] vector, int l){
    	vectors = vector.clone();
    	rowIDs = ids.clone();
    	len = l;
    	nnz = ids.length;
    	sparse = true;
    }
    
    public void set(List<Long> ids, List<Double> vector, int l){
    	vectors = new double[vector.size()];
    	rowIDs = new long[ids.size()];
    	for(int i=0;i<ids.size();i++){
    		vectors[i] = vector.get(i);
    		rowIDs[i] =  ids.get(i);
    	}
    	len = l;
    	nnz = ids.size();
    	sparse = true;
    }
    
    /**
     * Set a dense vector
     * @param vector
     */
    public void set(DenseDoubleMatrix1D vector) {
    	vectors = vector.toArray();
    	len = vector.size();
    	nnz = vector.size();
    	sparse = false;
    }
    
    public void set(double[] vector){
    	vectors = vector.clone();
    	len = vector.length;
    	nnz = vector.length;
    	sparse = false;
    }
    
    public void set(List<Double> vector){
    	vectors = new double[vector.size()];
    	for(int i=0;i<vector.size();i++)
    		vectors[i] = vector.get(i);
    }
    
    public void set(double v){
    	vectors = new double[1];
    	vectors[0] = v;
    	len = 1; nnz = 1;
    	sparse = false;
    }
    
//    /**
//     * Get the sparse vector
//     * @return
//     */
//    public SparseDoubleMatrix1D getSparseVector() {
//    	SparseDoubleMatrix1D vector = new SparseDoubleMatrix1D((int)len);
//    	for(int i=0; i<nnz; i++)
//    		vector.setQuick((int) rowIDs[i], vectors[i]);
//        return vector;
//    }
    
    /**
     * Get the dense vector
     * @return
     */
    public DenseDoubleMatrix1D getDenseVector() {
    	DenseDoubleMatrix1D vector = new DenseDoubleMatrix1D((int)len);
    	for(int i=0; i<len; i++)
    		vector.setQuick(i, vectors[i]);
        return vector;
    }

    public double[] getVector(){
    	return vectors.clone();
    }
    
    public double[] viewVector(){
    	return vectors;
    }
    
    public long[] getIDs() {
    	return rowIDs.clone();
    }
    
	public long getLen() {
		return len;
	}

	public void setLen(long len) {
		this.len = len;
	}

	public int getNnz() {
		return nnz;
	}

	public void setNnz(int nnz) {
		this.nnz = nnz;
	}

	public boolean isSparse() {
		return sparse;
	}

	public void setSparse(boolean sparse) {
		this.sparse = sparse;
	}
	
	public double getFirstVal(){
		return vectors[0];
	}
	
	public long getFirstID(){
		return rowIDs[0];
	}
	
	@Override
	public String toString(){
		StringBuilder line = new StringBuilder("");
		if(sparse){
			for(int i=0;i<nnz;i++){
				if(i!=0) line.append(",");
				line.append(rowIDs[i]);
				line.append(",");
				line.append(vectors[i]);
			}
		} else {
			for(int i=0;i<len;i++){
				if(i!=0) line.append(",");
				line.append(vectors[i]);
			}
		}
		return line.toString();
	}
}
