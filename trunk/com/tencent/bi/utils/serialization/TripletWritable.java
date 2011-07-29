package com.tencent.bi.utils.serialization;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.*;

public class TripletWritable implements Writable {

	protected int p;
	
	protected int q;
	
	protected double r;
	
	@Override
	public void readFields(DataInput in) throws IOException {
		p = in.readInt();
		q = in.readInt();
		r = in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(p);
		out.writeInt(q);
		out.writeDouble(r);
	}

	public int getP() {
		return p;
	}

	public void setP(int p) {
		this.p = p;
	}

	public int getQ() {
		return q;
	}

	public void setQ(int q) {
		this.q = q;
	}

	public double getR() {
		return r;
	}

	public void setR(double r) {
		this.r = r;
	}

}
