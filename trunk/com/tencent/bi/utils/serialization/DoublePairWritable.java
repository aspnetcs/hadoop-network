package com.tencent.bi.utils.serialization;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class DoublePairWritable implements Writable{

	private double k1;
	
	private double k2;
	
	@Override
	public void readFields(DataInput in) throws IOException {
		k1 = in.readDouble();
		k2 = in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(k1);
		out.writeDouble(k2);
	}
	
	public void set(double k1, double k2){
		this.k1 = k1;
		this.k2 = k2;
	}
	
	public double getFirst(){
		return k1;
	}

	public double getSecond(){
		return k2;
	}
	
	public String toString(){
		return k1+","+k2;
	}
}
