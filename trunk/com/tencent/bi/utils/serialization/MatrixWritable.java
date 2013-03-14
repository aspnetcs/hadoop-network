package com.tencent.bi.utils.serialization;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

/**
 * Writable for matrix
 * @author tigerzhong
 *
 */
public class MatrixWritable implements Writable {
	/**
	 * Number of columns
	 */
	private int numCols;
	/**
	 * Number for rows
	 */
	private int numRows;
	/**
	 * Number of non-zero elements in matrix
	 */
	private int nnz;
	/**
	 * sparse or dense?
	 */
	private boolean sparse;
	/**
	 * Value list, triplet format
	 */
	private double[] vals;
	/**
	 * Row ID list
	 */
	private long[] rowIDs;
	/**
	 * Column ID list
	 */
	private long[] colIDs;
		
	@Override
	public void readFields(DataInput input) throws IOException {
		sparse = input.readBoolean();
		numRows = input.readInt();
		numCols = input.readInt();
		if(!sparse){
			int pt = 0;
			vals = new double[numRows*numCols];
			for(int i=0;i<numRows;i++)
				for(int j=0;j<numCols;j++)
					vals[pt++] = input.readDouble();
		} else {
			nnz = input.readInt();
			for(int i=0;i<nnz;i++){
				rowIDs[i] = input.readLong();
				colIDs[i] = input.readLong();
				vals[i] = input.readDouble();
			}
		}
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeBoolean(sparse);
		output.writeInt(numRows);
		output.writeInt(numCols);
		if(!sparse){
			for(double v : vals){
				output.writeDouble(v);
			}
		} else {
			output.writeInt(nnz);
			for(int i=0;i<nnz;i++){
				output.writeLong(rowIDs[i]);
				output.writeLong(colIDs[i]);
				output.writeDouble(vals[i]);
			}
		}
	}

}
