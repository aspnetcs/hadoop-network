package com.tencent.bi.utils.serialization;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;

public class KeyPairWritable implements Writable, WritableComparable<KeyPairWritable>{

	private long k;
	
	private double v;
	
	@Override
	public void readFields(DataInput in) throws IOException {
		k = in.readLong();
		v = in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(k);
		out.writeDouble(v);
	}
	
	public void set(long k, double v){
		this.k = k;
		this.v = v;
	}
	
	public long getKey(){
		return k;
	}

	public double getValue(){
		return v;
	}

	public String toString(){
		return k+","+v;
	}

	@Override
	public int compareTo(KeyPairWritable other) {
		if(this.k==other.getValue()){
			return Double.compare(this.v, other.getValue());
		}
		else return this.k>other.getValue() ? 1 : -1;
	}
}
