package com.tencent.bi.utils.serialization;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.*;

/**
 * Serialization object for triplet
 * @author tigerzhong
 *
 */
public class TripletWritable implements Writable {

	protected long p;
	
	protected double s;
	
	protected double r;
	
	@Override
	public void readFields(DataInput in) throws IOException {
		p = in.readLong();
		s = in.readDouble();
		r = in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(p);
		out.writeDouble(s);
		out.writeDouble(r);
	}

	public void set(long p, double s, double r){
		this.p = p;
		this.s = s;
		this.r = r;
	}
	
	public long getP() {
		return p;
	}

	public void setP(long p) {
		this.p = p;
	}

	public double getS() {
		return s;
	}

	public void setS(double s) {
		this.s = s;
	}

	public double getR() {
		return r;
	}

	public void setR(double r) {
		this.r = r;
	}

	@Override
	public String toString(){
		return this.p+","+this.s+","+this.r;
	}
}
